import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE
import numpy as np

def kmeans_cluster_maerkte(df:pd.DataFrame, random_seed=42, k=3) -> pd.DataFrame:
    '''Gruppierung der originalen Bestellpositionen auf Markebene.
    
    Anschließendes standardisieren mittels Standardscalar() und clustering mittels kMeans, vorgegebenen Seeds + Cluster Anzahl.
    
    Args:
        df (DataFrame): Originale Bestellpositionen
        random_seed (Integer): Startwert für den Zufallsgenerator (Reproduzierungsgründe)
        k (Integer): Anzahl der zu erzeugende Cluster. Festgesetzt auf k=3 nach Erkenntnissen aus find_best_k.ipynb
    
    Returns:
        df_maerkte (DataFrame): Auf Marktebene geclusterte Bestellpositionen mit den Kennzahlen diff_article, avg_kolli, orders und Informationen über das zugeordnete cluster 
    '''
    df_maerkte = df.copy()

    # Erzeugung einer Fake "order_id" mit der Annahme, dass max einmal pro Tag bestellt wird
    df_maerkte['order_id'] = df_maerkte.groupby(['Marktnummer', 'Datum']).ngroup()

    # Gruppierung auf Marktebene 
    df_maerkte = df_maerkte.groupby('Marktnummer').agg({
        'Artikelnummer': 'nunique',     # Anzahl unterschiedlicher Artikel, welche der Markt bestellt hat
        'MengeInKolli': 'median',       # Durchschnittliche bestellte Kolli Menge pro Markt
        'order_id': 'nunique'           # Anzahl an Bestellungen, welche der Markt getätigt hat
    }).rename(columns={
        'Artikelnummer': 'diff_article',    # different_article
        'MengeInKolli': 'avg_kolli',        # avergae_kolli
        'order_id': 'orders'
    }).reset_index()

    labels = df_maerkte[['diff_article', 'avg_kolli', 'orders']].copy()

    # Standardisierung der verschiedene Variablen mit StandardScalar
    scalar = StandardScaler()
    x_scaled_k3 = scalar.fit_transform(labels)

    # Clustering auf Marktebene
    model = KMeans(n_clusters=k, random_state=random_seed)
    predicted_labels_k3 = model.fit_predict(x_scaled_k3)

    df_maerkte['cluster'] = predicted_labels_k3

    return df_maerkte

def synth_maerkte(df_maerkte: pd.DataFrame, maerkte_count: int) -> pd.DataFrame:
    '''Generierung von synthetischen Märkten auf Basis der geclusterten Märkte in kmeans_cluster_maerkte.

    Für jedes cluster wird die relationale Bedeutung der cluster berechnet um eine passende Anzahl an Märkten pro cluster zu generieren.
    Es werden die Spalten 'diff_article', 'avg_kolli' & 'orders' als Grundlage für die Generierung mittels GaussianKDE verwendet.
    Anschließend werden die neu erzeugten Werte gerundet, geclipped um Plausibilität der Werte zu gewährleisten
    
    Args:
        df_maerkte (DataFrame): Cluster Marktdatensatz aus kmeans_cluster_maerkte
        maerkte_count (Integer): Anzahl zu generierenden Märkten 
    
    Returns:
        synthetic_maerkte (DataFrame): Synthetischer Marktdatensatz'''

    allowed_cluster = df_maerkte['cluster'].unique()
    allowed_share = df_maerkte['cluster'].value_counts(normalize=True)

    synthetic_parts = []

    for cluster_id in allowed_cluster:

        df_maerkte_cluster = df_maerkte[df_maerkte['cluster'] == cluster_id]
        
        # Wie war das Verhältnis der cluster beim originalen Datensatz und wie wäre das bei den zu erzeugendn Märkten?
        n_cluster = int(round(maerkte_count * allowed_share.loc[cluster_id]))

        # Columns für Generierung
        cols_for_model = df_maerkte_cluster[['diff_article', 'avg_kolli', 'orders']]

        # Modell trainieren
        model_maerkte = GaussianMultivariate(distribution=GaussianKDE)
        model_maerkte.fit(cols_for_model)

        # Maerkte generieren
        synthetic_maerkte = model_maerkte.sample(n_cluster)

        # Plausibilität gewährleisten bei orders, avg_kolli & diff_article
        synthetic_maerkte['orders'] = synthetic_maerkte['orders'].round().astype(int).clip(lower=1)
        synthetic_maerkte['avg_kolli'] = synthetic_maerkte['avg_kolli'].round().astype(float).clip(lower=1)
        synthetic_maerkte['diff_article'] = synthetic_maerkte['diff_article'].round().astype(int).clip(lower=1)

        synthetic_maerkte['cluster'] = cluster_id

        synthetic_parts.append(synthetic_maerkte)
    
    synthetic_maerkte = pd.concat(synthetic_parts)

    synthetic_maerkte = synthetic_maerkte.reset_index(drop=True)

    # Fake_Marktnummer generieren
    synthetic_maerkte['Marktnummer'] = np.arange(1, len(synthetic_maerkte) + 1)

    # Sortierung
    synthetic_maerkte = synthetic_maerkte[['Marktnummer', 'diff_article', 'avg_kolli', 'orders', 'cluster']]

    return synthetic_maerkte

def synth_maerkte_custom(df_maerkte: pd.DataFrame, maerkte_count: int, cluster_0_rel: float, cluster_1_rel: float, cluster_2_rel: float) -> pd.DataFrame:
    '''Generierung von synthetischen Märkten auf Basis der geclusterten Märkte in kmeans_cluster_maerkte und eigenen Cluster Angaben.

    Für jedes cluster wird auf Basis der cluster eingaben Märkten pro cluster zu generiert.
    Es werden die Spalten 'diff_article', 'avg_kolli' & 'orders' als Grundlage für die Generierung mittels GaussianKDE verwendet.
    Anschließend werden die neu erzeugten Werte gerundet, geclipped um Plausibilität der Werte zu gewährleisten
    
    Args:
        df_maerkte (DataFrame): Cluster Marktdatensatz aus kmeans_cluster_maerkte
        maerkte_count (Integer): Anzahl zu generierenden Märkten 
        cluster_0_rel (Float): Relativen Anteil an der Gesamtheit an der zu generierenden Märktenvon cluster 0
        cluster_1_rel (Float): Relativen Anteil an der Gesamtheit an der zu generierenden Märktenvon cluster 1
        cluster_2_rel (Float): Relativen Anteil an der Gesamtheit an der zu generierenden Märktenvon cluster 2
    
    Returns:
        synthetic_maerkte (DataFrame): Synthetischer Marktdatensatz'''

    allowed_cluster = df_maerkte['cluster'].unique()

    allowed_share = {
        0: cluster_0_rel,
        1: cluster_1_rel,
        2: cluster_2_rel
    }

    synthetic_parts = []

    for cluster_id in allowed_cluster:

        df_maerkte_cluster = df_maerkte[df_maerkte['cluster'] == cluster_id]
        
        # Wie war das Verhältnis der cluster beim originalen Datensatz und wie wäre das bei den zu erzeugendn Märkten?
        n_pro_cluster = int(round(maerkte_count * allowed_share[cluster_id]))

        # Columns für Generierung
        cols_for_model = df_maerkte_cluster[['diff_article', 'avg_kolli', 'orders']]

        # Modell trainieren
        model_maerkte = GaussianMultivariate(distribution=GaussianKDE)
        model_maerkte.fit(cols_for_model)

        # Maerkte generieren
        synthetic_maerkte = model_maerkte.sample(n_pro_cluster)

        # Plausibilität gewährleisten bei orders, avg_kolli & diff_article
        synthetic_maerkte['orders'] = synthetic_maerkte['orders'].round().astype(int).clip(lower=1)
        synthetic_maerkte['avg_kolli'] = synthetic_maerkte['avg_kolli'].round().astype(float).clip(lower=1)
        synthetic_maerkte['diff_article'] = synthetic_maerkte['diff_article'].round().astype(int).clip(lower=1)

        synthetic_maerkte['cluster'] = cluster_id

        synthetic_parts.append(synthetic_maerkte)
    
    synthetic_maerkte = pd.concat(synthetic_parts)

    synthetic_maerkte = synthetic_maerkte.reset_index(drop=True)

    # Fake_Marktnummer generieren
    synthetic_maerkte['Marktnummer'] = np.arange(1, len(synthetic_maerkte) + 1)

    # Sortierung
    synthetic_maerkte = synthetic_maerkte[['Marktnummer', 'diff_article', 'avg_kolli', 'orders', 'cluster']]

    return synthetic_maerkte





