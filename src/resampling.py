import pandas as pd
import random
from src.maerkte import kmeans_cluster_maerkte

def resampling_hochrechnung(df: pd.DataFrame, anzahl_neue_maerkte: int, seed: int = 42) -> pd.DataFrame:
    ''' 
    Erstellt aus zufällig gezogegen Märkten und deren Bestellungen eine neues DataFrame
    mit Märkten und Bestellpositionen

    Args:
        df (DataFrame): Original DataFrame mit allen Bestellpositionen
        anzahl_neue_maerkte (Integer): Anzahl neu zu generierende Märkte
        seed (Integer): Startpunkt für den Zufallsgenerator

    Returns:
        neue_bestellpositionen (DataFrame): DataFrame mit neu generierten Bestellungen
    '''

    # Clusterbildung mit kmeans
    df_cluster = kmeans_cluster_maerkte(df)

    cluster_share = df_cluster["cluster"].value_counts(normalize=True)

    random.seed(seed)

    neue_maerkte = []
    neue_marktnummer = 1

    for cluster_id, anteil in cluster_share.items():
        maerkte_im_cluster = df_cluster.loc[df_cluster["cluster"] == cluster_id, "Marktnummer"].tolist()

        n = round(anzahl_neue_maerkte * anteil)

        gezogene_maerkte = random.choices(maerkte_im_cluster, k=n)

        for quell_markt in gezogene_maerkte:
            neue_maerkte.append({
                "Marktnummer_neu": neue_marktnummer,
                "quell_markt": quell_markt,
                "cluster": cluster_id
            })
            neue_marktnummer += 1

    maerkte_df = pd.DataFrame(neue_maerkte)

    neue_bestellpositionen = maerkte_df.merge(
        df,
        left_on="quell_markt",
        right_on="Marktnummer",
        how="left"
    )

    neue_bestellpositionen = neue_bestellpositionen.drop(columns=["Marktnummer"])

    neue_bestellpositionen = neue_bestellpositionen.rename(columns={
        "Marktnummer_neu": "Marktnummer"
        }
    )

    return neue_bestellpositionen

def resampling_maerkte(df_resampling: pd.DataFrame) -> pd.DataFrame:

    df_maerkte = df_resampling.copy()

    df_maerkte['order_id'] = df_maerkte.groupby(['Marktnummer', 'Datum']).ngroup()

    df_maerkte = df_maerkte.groupby('Marktnummer').agg({
        'cluster': 'first',
        'Artikelnummer': 'nunique',     # Anzahl unterschiedlicher Artikel, welche der Markt bestellt hat
        'MengeInKolli': 'mean',       # Durchschnittliche bestellte Kolli Menge pro Markt
        'order_id': 'nunique',           # Anzahl an Bestellungen, welche der Markt getätigt hat
    }).rename(columns={
        'Artikelnummer': 'diff_article',    # different_article
        'MengeInKolli': 'avg_kolli',        # avergae_kolli
        'order_id': 'orders'
    }).reset_index()

    return df_maerkte

def resampling_bestellungen(df_resampling: pd.DataFrame) -> pd.DataFrame:

    df_bestellugnen = df_resampling.copy()

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

    return df_bestellugnen