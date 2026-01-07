import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def abc_klassifizierung(a) -> str:
    '''
    Klassifiziert die kummulierten relativen Anteile mit den Buchstaben A,B oder C je nachdem welchen Wert sie haben
    
    Args:
        a (Series): Eingabespalte mit den Werten

    Returns:
        A,B,C (String): Gibt Wert zurück, klassische ABC-Analyse
    '''
    if a >= 0 and a < 0.85:
        return 'A'
    elif a >=0.85 and a < 0.95:
        return 'B'
    elif a >=0.95:
        return 'C'

def abc_analyse(orig_df:pd.DataFrame, synth_df:pd.DataFrame) -> pd.DataFrame:
    '''
    Erstellt im ersten Schritt eine mengenbasierte ABC-Analyse auf Basis der original Bestellpositionen.
    Im zweiten Schritt wird eine mengenbasierte ABC-Analyse auf Basis der verknüpften original und synthetischen Bestellpositionen durchgeführt
    Im letzten Schritt werden beiden Analysen gemerged für einen Vergleich der Ergebnisse

    Args:
        orig_df (DataFrame): Original Bestellpositionen
        synth_df (DataFrame): Synthetische Bestellpositionen

    Returns:
        abc (DataFrame): DataFrame mit den Spalten Artikelnummer, MengenInKolli, anteil und abc_klasse für das jeweilige df
    '''

    # ABC-Analyse auf Original-Daten
    orig_abc = (
        orig_df.groupby('Artikelnummer')['MengeInKolli'].sum()
        .sort_values(ascending=False).reset_index()
    )
    orig_abc['anteil'] = orig_abc['MengeInKolli'] / orig_abc['MengeInKolli'].sum()
    orig_abc['kummuliert'] = orig_abc['anteil'].cumsum()
    orig_abc['abc_klasse'] = orig_abc['kummuliert'].apply(abc_klassifizierung)

    # Verknüpfung Synth & Original-Daten + ABC-Analyse
    orig_df_reduziert = orig_df[['Datum', 'Marktnummer', 'Artikelnummer', 'MengeInKolli']].copy()
    synth_df_reduziert = synth_df[['Datum', 'Marktnummer', 'Artikelnummer', 'MengeInKolli']].copy()

    df_combined = pd.concat([orig_df_reduziert, synth_df_reduziert])

    new_abc = (
        df_combined.groupby('Artikelnummer')['MengeInKolli'].sum()
        .sort_values(ascending=False).reset_index()
    )
    new_abc['anteil'] = new_abc['MengeInKolli'] / new_abc['MengeInKolli'].sum()
    new_abc['kummuliert'] = new_abc['anteil'].cumsum()
    new_abc['abc_klasse'] = new_abc['kummuliert'].apply(abc_klassifizierung)

    # Merge neue ABC-Analyse an alte ABC-Analyse
    abc = pd.merge(
        orig_abc[['Artikelnummer', 'MengeInKolli', 'anteil', 'kummuliert', 'abc_klasse']],
        new_abc[['Artikelnummer', 'MengeInKolli', 'anteil', 'kummuliert', 'abc_klasse']],
        how='left',
        on='Artikelnummer',
        suffixes=('_orig', '_synth&orig')
    ).reset_index()

    return abc


