import pandas as pd
import random

# zwei naiv methoden 
# 1. erste voll kommen naiv reproduzierbarkeit von ist
# 2. fair naiv einfluss auf den soll durch cluster blablabla

def naive_hochrechnung(df:pd.DataFrame, anzahl_neue_maerkte:int, seed:int) -> pd.DataFrame:
    ''' 
    Erstellt aus zufällig gezogegen Märkten und deren Bestellungen eine neues DataFrame
    mit Märkten und Bestellpositionen ohne Beachtung

    Args:
        df (DataFrame): Original DataFrame mit allen Bestellpositionen
        anzahl_neue_maerkte (Integer): Anzahl neu zu generierende Märkte
        seed (Integer): Startpunkt für den Zufallsgenerator

    Returns:
        neue_bestellpositionen (DataFrame): DataFrame mit neu generierten Bestellungen
    '''
    df_copy = df.copy()

    # Random Ziehung von X-Märkten und Speicherung als neues df
    random.seed(seed)
    neue_maerkte = random.choices(df_copy['Marktnummer'].unique(), k=anzahl_neue_maerkte)
    neue_maerkte = pd.DataFrame({'quell_markt':neue_maerkte,
                                 'Marktnummer': range(1, anzahl_neue_maerkte + 1)})

    # Join der ehemaligen orderlines
    neue_bestellpositionen = pd.merge(
        neue_maerkte,
        df,
        on='Marktnummer',
        how='left'
    )

    neue_bestellpositionen = neue_bestellpositionen.drop(columns=['quell_amrkt', 'Marktnummer_y'])
    neue_bestellpositionen = neue_bestellpositionen.rename(columns={'Marktnummer_x': 'Marktnummer'})

    return neue_bestellpositionen

def naive_hochrechnung_cluster(df: pd.DataFrame, anzahl_neue_maerkte: int, seed: int) -> pd.DataFrame:
    ''' 
    Erstellt aus zufällig gezogegen Märkten und deren Bestellungen eine neues DataFrame
    mit Märkten und Bestellpositionen ohne Beachtung

    Args:
        df (DataFrame): Original DataFrame mit allen Bestellpositionen
        anzahl_neue_maerkte (Integer): Anzahl neu zu generierende Märkte
        seed (Integer): Startpunkt für den Zufallsgenerator

    Returns:
        neue_bestellpositionen (DataFrame): DataFrame mit neu generierten Bestellungen
    '''
    