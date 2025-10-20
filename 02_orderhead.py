import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE

n_orders_tag = 1

def generate_orders(df_sample: pd.DataFrame, n_orders = n_orders_tag):
    ''' Generierung und Anpassung der Orderköpfe
    
    Args:
        df_sample (pd.DataFrame): Stichproben DataFrame
        n_orders (integer): Anzahl an Orders die generiert werden sollen 
         
    Returns:
        synthetic_orders: Synthetische Bestellköpfe'''
    
    # Gruppierung auf Bestellebene
    order_df = df_sample.groupby('order_id').agg({
        'Marktnummer': 'first',
        'Datum': 'first',
        'Artikelnummer': 'count', # Bestellzeilen zusammenfassen
        'cluster': 'first'  
    }).rename(columns={'Artikelnummer': 'orderlines'}).reset_index()

    # Vorbereitung auf Nearest-Neighbour Method
    allowed_dates = df_sample['Datum'].unique()
    allowed_cluster = df_sample['cluster'].unique()

    # Differenz aus geringstem und aktuellem Tag
    order_df['Wochentag'] = pd.to_datetime(order_df['Datum']).dt.weekday
    order_df['Datum'] = pd.to_datetime(order_df['Datum'])
    min_date = order_df['Datum'].min()
    order_df['Datum'] = (order_df['Datum'] - min_date).dt.days

    # Kategorische Spalte encoden
    le_markt = LabelEncoder()
    order_df['Marktnummer_enc'] = le_markt.fit_transform(order_df['Marktnummer'])

    # Modell traineren
    model_order = GaussianMultivariate(distribution=GaussianKDE)
    model_order.fit(order_df[['Marktnummer_enc', 'Datum', 'Wochentag', 'orderlines', 'cluster']])

    # Bestellköpfe generieren
    synthetic_orders = model_order.sample(n_orders)

    # Rücktransformieren der kategorischen Spalte
    synthetic_orders['Marktnummer_enc'] = synthetic_orders['Marktnummer_enc'].clip(
        lower=0,
        upper=len(le_markt.classes_) - 1
    ).round().astype(int)
    synthetic_orders['Marktnummer'] = le_markt.inverse_transform(synthetic_orders['Marktnummer_enc'])
    synthetic_orders.drop(columns=['Marktnummer_enc'], inplace=True)

    # Plausibilität bei den Bestellzeilen
    synthetic_orders['orderlines'] = synthetic_orders['orderlines'].round().astype(int).clip(lower=1)

    # Rücktransformieren der Datums Spalte
    synthetic_orders['Datum_numeric'] = synthetic_orders['Datum']
    synthetic_orders['Datum'] = pd.to_datetime(synthetic_orders['Datum'], unit='D', origin=min_date)

    # Nearest-Neighbor-Mapping auf reale Tage
    allowed_dates = pd.to_datetime(sorted(df['Datum'].unique()))
    allowed_numeric = (allowed_dates - min_date).days.values

    synthetic_orders['Datum'] = synthetic_orders['Datum_numeric'].apply(
        lambda x: allowed_dates[np.argmin(np.abs(allowed_numeric - x))]
    ).dt.date

    # Nearest-Neighbour-Mapping auf reale cluster
    synthetic_orders['cluster_v2'] = synthetic_orders['cluster'].apply(
        lambda x: allowed_cluster[np.argmin(np.abs(allowed_cluster - x))]
    )

    synthetic_orders.drop(columns=['Datum_numeric', 'Wochentag', 'cluster'], inplace=True)

    synthetic_orders = synthetic_orders.rename(columns={'cluster_v2': 'cluster'})

    return synthetic_orders


