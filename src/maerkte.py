import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE
import numpy as np

def kmeans_cluster_maerkte(df:pd.DataFrame, random_seed=42, k=3) -> pd.DataFrame:
    '''Gruppierung und Clustering auf Marktebene.
    
    Args:
        df (DataFrame): Vollständiges Orderline Dataframe.
        random_seed (Integer): Ist auf 42 festgeschrieben, Seed für das Cluster
        k (Integer): Ist auf 3 festgeschrieben, nach Test in find_best_k.ipynb, Anzahl Cluster
    
    Returns:
        df: Gruppiertes DataFrame auf Marktebene und cluster Kennzeichen 
    '''
    df_maerkte = df.copy()

    df_maerkte['order_id'] = df_maerkte.groupby(['Marktnummer', 'Datum']).ngroup()

    # Gruppierung auf Marktebene 
    df_maerkte = df_maerkte.groupby('Marktnummer').agg({
        'Artikelnummer': 'nunique',     # Wie viele verschiedene Artikel hat Markt X über Zeitraum Y bestellt?
        'MengeInKolli': 'median',       # Wie viele Kolli bestellt der Markt im Durchschnitt pro Bestellposition?
        'order_id': 'nunique'           # Wie viele Bestellungen hat der Markt X über Zeitraum Y getätigt?
    }).rename(columns={
        'Artikelnummer': 'diff_article',    # different_article
        'MengeInKolli': 'avg_kolli',        # avergae_kolli
        'order_id': 'orders'
    }).reset_index()

    labels = df_maerkte[['diff_article', 'avg_kolli', 'orders']].copy()

    # Standardisierung der verschiedene Metriken mit StandardScalar
    scalar = StandardScaler()
    x_scaled_k3 = scalar.fit_transform(labels)

    # Clustering auf Marktebene
    model = KMeans(n_clusters=k, random_state=random_seed)
    predicted_labels_k3 = model.fit_predict(x_scaled_k3)

    df_maerkte['cluster'] = predicted_labels_k3

    return df_maerkte

def synth_maerkte(df_maerkte: pd.DataFrame, maerkte_count: int) -> pd.DataFrame:
    '''Generierung von synthetischen Märkten.
    
    Args:
        df_maerkte (DataFrame): Gruppierter Marktdatensatz aus kmeans_cluster_maerkte
        maerkte_count (Integer): Anzahl zu generierenden Märkten 
    
    Returns:
        df: Synthetischer Marktdatensatz'''

    allowed_cluster = df_maerkte['cluster'].unique()
    allowed_share = df_maerkte['cluster'].value_counts(normalize=True)

    synthetic_parts = []

    for cluster_id in allowed_cluster:

        df_maerkte_cluster = df_maerkte[df_maerkte['cluster'] == cluster_id]
        
        # Relationalisierung der zu generierenden Märkten
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





