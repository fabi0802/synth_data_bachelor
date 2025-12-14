import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE
import numpy as np

def kmeans_cluster_bestellungen(df: pd.DataFrame, random_seed=42, k=2) -> pd.DataFrame:
    '''
    Kategorisierung der Bestellungen mittels kMean-clustering
    
    Args:
        df (pd.DataFrame): Original df
        random_seed (Integer): Ist auf 42 festgeschrieben, Seed für das Cluster
        k (Integer): Ist auf 3 festgeschrieben, nach Test in find_best_k.ipynb, Anzahl Cluster
         
    Returns:
        df: Gruppiertes DataFrame auf Bestellebene und cluster Kennzeichen
    '''
    df_bestellugnen = df.copy()

    df_bestellugnen['order_id'] = df_bestellugnen.groupby(['Marktnummer', 'Datum']).ngroup()

    df_bestellugnen['Sortiment'] = df_bestellugnen['Artikelnummer'].astype(str).str[2:4]

    df_bestellugnen['Wochentag'] = pd.to_datetime(df_bestellugnen['Datum']).dt.weekday

    # Gruppierung auf Bestellebene 
    df_bestellugnen = df_bestellugnen.groupby('order_id').agg({
        'Marktnummer': 'first',
        'Datum': 'first',
        'Wochentag': 'first',       # An welchem Wochentag wurde bestellt (Mo., Di, etc.)?
        'Artikelnummer': 'count',   # Wie viele Bestellpositionen existieren in der Bestellung?
        'Sortiment': 'nunique'      # Wie viele unterschiedliche Sortimente existieren in der Bestellung?
    }).rename(columns={
        'Artikelnummer': 'orderlines',
        'Sortiment': 'diff_sortimente'  # different_sortimente
    }).reset_index()

    labels = df_bestellugnen[['Wochentag', 'orderlines', 'diff_sortimente']].copy()

    # Standardisierung der verschiedene Metriken mit StandardScalar
    scalar = StandardScaler()
    x_scaled_k2 = scalar.fit_transform(labels)

    # Clustering auf Bestellebene
    model = KMeans(n_clusters=k, random_state=random_seed)
    predicted_labels_k2 = model.fit_predict(x_scaled_k2)

    df_bestellugnen['cluster'] = predicted_labels_k2

    return df_bestellugnen

def synth_bestellungen(df_bestellungen: pd.DataFrame, df_maerkte: pd.DataFrame, synth_maerkte: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic orders based on clustered order and market data using Gaussian copula models.

    This function creates synthetic order data by first merging market cluster information with order clusters,
    computing the distribution of order clusters within market clusters, training Gaussian multivariate models
    for each order cluster, and then sampling synthetic orders for each synthetic market while preserving
    the cluster relationships and temporal patterns.

    Args:
        df_bestellungen (pd.DataFrame): Clustered order data from kmeans_cluster_bestellungen, containing
            order-level aggregations with columns like 'Marktnummer', 'Datum', 'Wochentag', 'orderlines',
            'diff_sortimente', and 'cluster'.
        df_maerkte (pd.DataFrame): Clustered market data from kmeans_cluster_maerkte, containing at least
            'Marktnummer' and 'cluster' columns.
        synth_maerkte (pd.DataFrame): Synthetic market data from synth_maerkte, containing 'Marktnummer',
            'cluster', and 'orders' columns representing the number of orders to generate per market.

    Returns:
        pd.DataFrame: Synthetic orders DataFrame with columns including 'Datum', 'Wochentag', 'orderlines',
            'diff_sortimente', 'Marktnummer', 'cluster_bestellungen', and 'cluster_markt'. The synthetic
            data preserves the statistical properties and relationships of the original clustered data.

    Notes:
        - Uses GaussianMultivariate with GaussianKDE for modeling multivariate distributions.
        - Ensures order counts match the specified 'orders' per market by adjusting for rounding errors.
        - Maps synthetic dates to the nearest real dates from the training data to maintain temporal validity.
        - Clips synthetic values to reasonable bounds (e.g., orderlines >= 1, weekdays 0-6).
    """
    # Markt Cluster an die Bestellcluster dazuholen bspw.  Marktcluster 0 = (80% Betellcluster 0 & 20% Bestellcluster 1)
    df_bestellungen = pd.merge(
        df_bestellungen,
        df_maerkte[['Marktnummer', 'cluster']],
        on='Marktnummer',
        how='left',
        suffixes=('_bestellungen', '_markt')
    ).reset_index(drop=True)

    # Matrix Tabelle zur Häufigkeit der Bestellclsuter in den Marktclustern
    kreuztabelle = pd.crosstab(
        df_bestellungen['cluster_markt'],
        df_bestellungen['cluster_bestellungen'],
        normalize='index'
    )

    # Iterieren über die for-Schleife
    cluster_bestellungen_unq = df_bestellungen['cluster_bestellungen'].unique()
    
    # Speichern der models zum generieren
    sampling_models = {}

    # Für jedes Bestell Cluster ein Model trainieren und abspeichern
    for cluster_id in cluster_bestellungen_unq:

        df_bestellungen_cluster = df_bestellungen[df_bestellungen['cluster_bestellungen'] == cluster_id].copy()

        df_bestellungen_cluster['Datum'] = pd.to_datetime(df_bestellungen_cluster['Datum'])
        df_bestellungen_cluster['Wochentag'] = pd.to_datetime(df_bestellungen_cluster['Datum']).dt.weekday
        min_date = df_bestellungen_cluster['Datum'].min()
        df_bestellungen_cluster['Datum_num'] = (df_bestellungen_cluster['Datum'] - min_date).dt.days

        cols_for_model = df_bestellungen_cluster[['Datum_num', 'Wochentag', 'orderlines', 'diff_sortimente']]

        model= GaussianMultivariate(distribution=GaussianKDE)
        model.fit(cols_for_model)

        sampling_models[cluster_id] = {
            'model': model,
            'min_date': min_date,
            'dates_unq': df_bestellungen_cluster['Datum'].unique()
        }

    # Speichern der synthetischen Bestellungen
    synthetic_parts = []

    # Iterierung über jeden einzelnen Markt / Zeile
    for _, markt in synth_maerkte.iterrows():

        markt_nr = markt['Marktnummer']
        markt_cluster = markt['cluster']
        orders = int(markt['orders'])

        # Anteile des ausgewählten Bestell Cluster am Markt Cluster bspw. Markt Cluster 0 => 80% Bestell Cluster 0 & 20% Bestell Cluster 1 
        anteile = kreuztabelle.loc[markt_cluster]

        # Speichern der Anzahl von Bestellungen pro Markt Cluster & Überprüfunng der Gesamtanzahl Bestellungen
        anzahl_pro_cluster = {}
        rest = orders

        # Iterierung über die relativen Anteile pro Markt Cluster pro Bestell Cluster
        for bestellcluster, anteil in anteile.items():
            n_orders_cluster = int(round(anteil * orders))
            anzahl_pro_cluster[bestellcluster] = n_orders_cluster
            rest -= n_orders_cluster

        # Korriegierung bei Rundungsfehlern (bspw. orders = 7 & Verhältnis Bestell Cluster 50% / 50% => 2 x round(3,5) => 8 != 7)
        if rest > 0:
            anzahl_pro_cluster[anteile.idxmax()] += 1
        elif rest < 0:
            anzahl_pro_cluster[anteile.idxmin()] -= 1


        for cluster_id, n_orders_cluster in anzahl_pro_cluster.items():

            if n_orders_cluster <= 0:
                continue

            pack = sampling_models[cluster_id]
            model = pack['model']
            min_date = pack['min_date']
            dates_unq = pd.to_datetime(pack['dates_unq'])

            # Bestellungen ziehen
            synth = model.sample(int(n_orders_cluster))

            # Aufräumen / Plausibilisierung
            synth['orderlines'] = synth['orderlines'].round().astype(int).clip(lower=1)
            synth['diff_sortimente'] = synth['diff_sortimente'].round().astype(int).clip(lower=1)
            synth['Wochentag'] = synth['Wochentag'].round().astype(int).clip(0, 6)

            # Datum zurück auf echte Tage mappen
            daten_num = synth['Datum_num'].values
            echte_daten = []
            allowed_numeric = (dates_unq - min_date).days

            for x in daten_num:
                idx = np.argmin(np.abs(allowed_numeric - x))
                echte_daten.append(dates_unq[idx])

            synth['Datum'] = pd.to_datetime(echte_daten).date

            # Markt-Zuordnung
            synth['Marktnummer'] = markt_nr
            synth['cluster_bestellungen'] = cluster_id
            synth['cluster_markt'] = markt_cluster

            synth = synth.drop(columns=['Datum_num'])

            synthetic_parts.append(synth)

    synthetic_bestellungen = pd.concat(synthetic_parts, ignore_index=True)

    # Fake Bestellnummern generieren
    synthetic_bestellungen['order_id'] = np.arange(1, len(synthetic_bestellungen) + 1)

    return synthetic_bestellungen


