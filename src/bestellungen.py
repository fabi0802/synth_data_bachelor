import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE
import numpy as np

def kmeans_cluster_bestellungen(df: pd.DataFrame, random_seed=42, k=2) -> pd.DataFrame:
    '''
    Gruppierung der originalen Bestellpositionen auf Bestellebene.

    Anschließendes standardisieren mittels Standardscalar() und clustering mittels kMeans, vorgegebenen Seeds + Cluster Anzahl.
    
    Args:
        df (pd.DataFrame): Originale Bestellpositionen
        random_seed (Integer): Startwert für den Zufallsgenerator (Reproduzierungsgründe)
        k (Integer): Anzahl der zu erzeugende Cluster. Festgesetzt auf k=2 nach Erkenntnissen aus find_best_k.ipynb
         
    Returns:
        df_bestellungen (DataFrame): Auf Bestelleben geclusterte Bestellpositionen mit den Kennzahlen diff_article, avg_kolli, orders und Informationen über das zugeordnete cluster 
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
    """Generiert synthetische Bestelldaten basierend auf historischen Verteilungen und Markt-Clustern.

        Zu Beginn wird an den gruppierten Bestelldatensatz aus k_means_cluster_bestellungen die Marktcluster aus k_means_cluster_maerkte angefügt.
        Dabei wird die Wahrscheinlichkeit, dass ein bestimmtes Bestell-Cluster in einem Markt-Cluster auftritt, über eine
        Kreuztabelle berücksichtigt.

        Pro Bestellcluster werden die Spalten 'Datum_num', 'Wochentag', 'orderlines' & 'diff_sortimente' als Grundlage für die Generierung mittels GaussianKDE verwendet.

        Für jede Zeile/ synthetischen Markt aus synth_maerkte wird ein passendes sampling gezogen unter Betrachtung der bestellcluster und marktcluster

        Rückrechnung von numerischen datumswerten auf echte Kalendertage

        Args:
            df_bestellungen (pd.DataFrame): Cluster Bestelldatensatz aus kmeans_cluster_bestellungen
            df_maerkte (pd.DataFrame): Cluster Marktdatensatz aus kmeans_cluster_maerkte
            synth_maerkte (pd.DataFrame): Synthetischer Marktdatensatz

        Returns:
            synthetic_bestellungen (pd.DataFrame): Synthetischer Bestelldatensatz
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

    # Eindeutige cluster_ids bestimmen
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

def synth_bestellungen_custom(df_bestellungen: pd.DataFrame, df_maerkte: pd.DataFrame, synth_maerkte: pd.DataFrame, bestellmix_override: dict) -> pd.DataFrame:
    """Generiert synthetische Bestelldaten basierend auf historischen Verteilungen und Markt-Clustern.

        Zu Beginn wird an den gruppierten Bestelldatensatz aus k_means_cluster_bestellungen die Marktcluster aus k_means_cluster_maerkte angefügt.
        Dabei wird die Wahrscheinlichkeit, dass ein bestimmtes Bestell-Cluster in einem Markt-Cluster auftritt, über eine
        Kreuztabelle berücksichtigt.

        Pro Bestellcluster werden die Spalten 'Datum_num', 'Wochentag', 'orderlines' & 'diff_sortimente' als Grundlage für die Generierung mittels GaussianKDE verwendet.

        Für jede Zeile/ synthetischen Markt aus synth_maerkte wird ein passendes sampling gezogen unter Betrachtung der bestellcluster und marktcluster

        Rückrechnung von numerischen datumswerten auf echte Kalendertage

        Args:
            df_bestellungen (pd.DataFrame): Cluster Bestelldatensatz aus kmeans_cluster_bestellungen
            df_maerkte (pd.DataFrame): Cluster Marktdatensatz aus kmeans_cluster_maerkte
            synth_maerkte (pd.DataFrame): Synthetischer Marktdatensatz

        Returns:
            synthetic_bestellungen (pd.DataFrame): Synthetischer Bestelldatensatz
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

    # Eindeutige cluster_ids bestimmen
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
        anteile = bestellmix_override[markt_cluster]

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
            anzahl_pro_cluster[max(anteile, key=anteile.get)] += 1
        elif rest < 0:
            anzahl_pro_cluster[min(anteile, key=anteile.get)] -= 1

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

