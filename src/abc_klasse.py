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

def draw_x_cutoff(ax, x, y=0.02, label=None, linestyle='--'):
    """
    Zeichnet eine vertikale Linie bei x und beschriftet sie.

    Args:
        ax       : matplotlib axis
        x        : x-Position (Artikelrang)
        y        : y-Position für Text (0–1 in Achsenkoordinaten)
        label    : Textbeschriftung (z.B. 'A-Grenze: 1200')
        linestyle: Linienart
    """
    ax.axvline(x, linestyle=linestyle, linewidth=1)

    if label is None:
        label = f"x = {x}"

    ax.text(
        x, y,
        label,
        rotation=90,
        va='bottom',
        ha='right',
        fontsize=9
    )

def cutoff_x(kumuliert_series, y):
    """kleinster Rang, bei dem kumuliert >= y"""
    arr = kumuliert_series.to_numpy()
    idx0 = int(np.searchsorted(arr, y, side="left"))
    return min(idx0 + 1, len(arr))  # Rang (1-basiert)

def abc_analyse_plot(abc_df):

    jetzt = datetime.now()
    df = abc_df.copy()

    # ORIGINAL
    df_orig = (
        df[['Artikelnummer', 'anteil_orig', 'kummuliert_orig']]
        .sort_values('anteil_orig', ascending=False)
        .reset_index(drop=True)
    )
    df_orig['rang'] = np.arange(1, len(df_orig) + 1)

    # WACHSTUM (orig + synth)
    df_growth = (
        df[['Artikelnummer', 'anteil_synth&orig', 'kummuliert_synth&orig']]
        .sort_values('anteil_synth&orig', ascending=False)
        .reset_index(drop=True)
    )
    df_growth['rang'] = np.arange(1, len(df_growth) + 1)

    # Cutoff-Punkte bestimmen
    xA_o = cutoff_x(df_orig['kummuliert_orig'], 0.85)
    xB_o = cutoff_x(df_orig['kummuliert_orig'], 0.95)

    xA_g = cutoff_x(df_growth['kummuliert_synth&orig'], 0.85)
    xB_g = cutoff_x(df_growth['kummuliert_synth&orig'], 0.95)

    # Normaler Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_orig['rang'], df_orig['kummuliert_orig'], label='Originale Märkte')
    plt.plot(df_growth['rang'], df_growth['kummuliert_synth&orig'], label='Originale + synthetische Märkte')

    ax = plt.gca()

    # ORIGINAL
    draw_x_cutoff(
        ax,
        xA_o,
        label=f"A-Grenze (Orig): {xA_o}",
        linestyle='-'
    )
    draw_x_cutoff(
        ax,
        xB_o,
        label=f"B-Grenze (Orig): {xB_o}",
        linestyle='-'
    )

    # WACHSTUM
    draw_x_cutoff(
        ax,
        xA_g,
        label=f"A-Grenze (Wachstum): {xA_g}",
        linestyle='--'
    )
    draw_x_cutoff(
        ax,
        xB_g,
        label=f"B-Grenze (Wachstum): {xB_g}",
        linestyle='--'
    )

    # Y ABC-Grenzen bestimmen
    plt.axhline(0.85, linestyle='--')
    plt.axhline(0.95, linestyle='--')
    plt.text(0.01, 0.86, 'A-Grenze (85%)', transform=plt.gca().transAxes)
    plt.text(0.01, 0.96, 'B-Grenze (95%)', transform=plt.gca().transAxes)

    plt.xlabel('Artikelrang')
    plt.ylabel('Kumulierter Mengenanteil')
    plt.title('Veränderung der ABC-Kurve durch Marktwachstum')
    plt.legend()
    plt.ylim(0, 1.02)
    plt.tight_layout()
    
    plt.savefig(f'reports/figures/auswertung_abc_analyse{jetzt:%Y_%m_%d_%H_%M}.png')

    return

def abc_zusammenfassung_neu(orig_df:pd.DataFrame, synth_df:pd.DataFrame):
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

    # Zusammenfassung pro Durchlauf
    new_abc_summary = new_abc['abc_klasse'].value_counts()

    return new_abc_summary
