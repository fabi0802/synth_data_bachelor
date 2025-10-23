import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE

def generate_orderlines(df_sample:pd.DataFrame, synthetic_orders:pd.DataFrame):
      
    # DataFrame auf relevante Spalten reduzieren
    df_sample = df_sample[['Artikelnummer', 'MengeInKolli', 'Markierung', 'cluster']]

    # Vorbereitung auf cluster separate Generierung
    allowed_cluster = df_sample['cluster'].unique()

    # Kategorische Spalte encoden
    le_artikel = LabelEncoder()
    df_sample['Artikelnummer_enc'] = le_artikel.fit_transform(df_sample['Artikelnummer'])

    le_markierung = LabelEncoder()
    df_sample['Markierung_enc'] = le_markierung.fit_transform(df_sample['Markierung'])

    # Orderlines pro Cluster
    cluster_count = synthetic_orders.groupby('cluster')['orderlines'].sum()

    # Liste zum speichern der Orderlines
    synthetic_positions_list = []

    # Cluster separate Generierung
    for cluster in allowed_cluster:
        cluster_df = df_sample[df_sample['cluster'] == cluster]
        n_orderlines = cluster_count.get(cluster, 0)

        # Modell trainieren
        model = GaussianMultivariate(distribution=GaussianKDE)
        model.fit(cluster_df[['Markierung_enc', 'Artikelnummer_enc', 'MengeInKolli']])

        # Bestellzeilen generieren
        synthetic_cluster_positions = model.sample(n_orderlines)

      # Rücktransformation der kategorischen Spalten
        synthetic_cluster_positions['Artikelnummer_enc'] = synthetic_cluster_positions['Artikelnummer_enc'].clip(
            lower=0, upper=len(le_artikel.classes_) - 1).round().astype(int)
        synthetic_cluster_positions['Artikelnummer'] = le_artikel.inverse_transform(synthetic_cluster_positions['Artikelnummer_enc'])

        synthetic_cluster_positions['Markierung_enc'] = synthetic_cluster_positions['Markierung_enc'].clip(
            lower=0, upper=len(le_markierung.classes_) - 1).round().astype(int)
        synthetic_cluster_positions['Markierung'] = le_markierung.inverse_transform(synthetic_cluster_positions['Markierung_enc'])

        
        # Plausibilität bei den Bestellzeilen
        synthetic_cluster_positions['MengeInKolli'] = synthetic_cluster_positions['MengeInKolli'].clip(lower=1).round()

        synthetic_cluster_positions['cluster_v3'] = cluster

        # Reduzierung auf benötigte Spalten
        synthetic_cluster_positions = synthetic_cluster_positions[['Artikelnummer', 'Markierung', 'MengeInKolli', 'cluster_v3']]

        # Verknüpfung der seperat generierten Orderlines in einer List
        synthetic_positions_list.append(synthetic_cluster_positions)

    # Verknüpfugn der List zu einem DataFrame
    if synthetic_positions_list:
        synthetic_positions_df = pd.concat(synthetic_positions_list, ignore_index=True)
    else:
        synthetic_positions_df = pd.DataFrame(columns=['Artikelnummer', 'Markierung', 'MengeInKolli', 'cluster_v3'])

    return synthetic_positions_df

