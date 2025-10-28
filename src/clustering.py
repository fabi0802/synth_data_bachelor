import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RANDOM_SEED = 42

def sampling(df:pd.DataFrame, SAMPLING_NR:int) -> pd.DataFrame:
    '''Erstellung einer Bestell_ID und extrahiert eine Stichprobe.
    
    Args:
        df (DataFrame): Vollständiges Orderline Dataframe.
        SAMPLING_NR (Integer): Anzahl der Stichprobe 
    
    Returns:
        df: Stichprobe vom Original mit erstellter Order_ID '''

    df['order_id'] = df.groupby(['Marktnummer', 'Datum']).ngroup()
    
    df = df.sample(n=SAMPLING_NR, random_state=RANDOM_SEED)

    return df


def kmean_cluster(df_sample:pd.DataFrame) -> pd.DataFrame:
    '''Kategorisierung der Bestellungen mittels kMean-clustering
    
    Args:
        df_sample (pd.DataFrame): Stichproben DataFrame aus def sampling
         
    Returns:
        df_sample: Stichprobe mit zugehöriger Kategorisierung '''

    # Encoding als Vorbereitung für den StandardScalar
    df_cluster = df_sample.copy()

    le = LabelEncoder()
    df_cluster['Marktnummer_enc'] = le.fit_transform(df_cluster['Marktnummer'])

    df_cluster['Markierung_enc'] = le.fit_transform(df_cluster['Markierung'])

    # Sortiments Kategorie
    df_cluster['Sortiment'] = df_cluster['Artikelnummer'].astype(str).str[2:4]

    # Aggregation auf Bestellkopfebene
    cluster_feautures = df_cluster.groupby('order_id').agg({
        'Marktnummer_enc': 'first',
        'Sortiment': 'nunique',
        'MengeInKolli': 'mean',
        'Markierung_enc': 'first',
        }).reset_index()

    # Eingabedaten standardisieren für eine ausgewogene Skalierung
    scalar = StandardScaler()
    x_scaled = scalar.fit_transform(cluster_feautures)
    
    kmeans = KMeans(n_clusters=5, random_state=RANDOM_SEED)

    cluster_feautures['cluster'] = kmeans.fit_predict(x_scaled)

    # Verknüpfung cluster an die orderlines
    df_sample = pd.merge(
    df_sample,
    cluster_feautures[['order_id', 'cluster']],
    how='left',
    on='order_id')
    
    return df_sample


def n_orders(df_sample: pd.DataFrame) -> int:
    ''' Ermittlung der Zielbestellungen pro Tag
    
    Args:
        df_sample (pd.DataFrame): Stichproben DataFrame 
         
    Returns:
        n_orders_tag: Anzahl Bestellungen pro Tag '''

    ziel_orders = df_sample.groupby('Datum')['order_id'].nunique().mean()

    tage = df_sample['Datum'].nunique()

    n_orders_tag = int(ziel_orders * tage)

    return n_orders_tag
