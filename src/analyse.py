import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.stats import chi2_contingency
from datetime import datetime

def vergleich_real_synth_maerkte(real_maerkte_v2: pd.DataFrame, synth_maerkte_v2: pd.DataFrame) -> pd.DataFrame:
    '''
    Vergleicht Lageparameter (Median), Streuung (Standardabweichung), Verteilung (KS-Test) und Abweichung (Wasserstein-Test)
    zwischen den realen Märkten und den synthetischen Märkten pro generierten cluster
    Kennzahlen hängen teilweise von der Anzahl der generierten Märkten ab

    Args:
        real_maerkte_v2 (DataFrame): Reale Cluster Märkte aus der Funktion kmeans_cluster_maerkte
        synth_maerkte_v2 (DataFrame): Synthetische Cluster Märkte aus der Funktion synth_maerkte oder synth_maerkte_custom (keine Gewährleistung)
    
    Returns:
        df_auswertung (DataFrame): Auswertungen Abgleich pro cluster zwischen real und synthetischen Märkten
    '''
    # Kennzahlen welche verglichen werden sollen
    unq_variable = ['diff_article', 'orders', 'avg_kolli']
    
    # Cluster über die iteriert werden soll
    unq_cluster = real_maerkte_v2['cluster'].unique()

    rows = []

    # Für jede Variable
    for variable in unq_variable:

        # Für jedes Cluster
        for cluster in unq_cluster:
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "median",
                "real": real_maerkte_v2.loc[real_maerkte_v2['cluster'] == cluster][variable].median(),
                "synth": synth_maerkte_v2.loc[synth_maerkte_v2['cluster'] == cluster][variable].median()
            })

        for cluster in unq_cluster: 
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "stdv",
                "real": real_maerkte_v2.loc[real_maerkte_v2['cluster'] == cluster][variable].std(),
                "synth": synth_maerkte_v2.loc[synth_maerkte_v2['cluster'] == cluster][variable].std()
            })

        for cluster in unq_cluster: 
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "Wasserstein Matrik",
                "real&synth": wasserstein_distance(
                real_maerkte_v2.loc[real_maerkte_v2['cluster'] == cluster][variable],
                synth_maerkte_v2.loc[synth_maerkte_v2['cluster'] == cluster][variable]
                )    
            })
        
        for cluster in unq_cluster:
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "KS-Test",
                "real&synth": ks_2samp(
                real_maerkte_v2.loc[real_maerkte_v2['cluster'] == cluster][variable],
                synth_maerkte_v2.loc[synth_maerkte_v2['cluster'] == cluster][variable]
                )    
            })

    df_auswertung = pd.DataFrame(rows)

    return df_auswertung

def vergleich_real_synth_maerkte_visual(real_maerkte: pd.DataFrame, synth_maerkte: pd.DataFrame):
    '''
    Vergleicht die Verteilungen zwischen den realen und synthetischen Märkten für die Variablen diff_article, avg_kolli, orders in einem Histogramm
    Plots hängen von der Anzahl der generierten Märkten ab

    Args:
        real_maerkte (DataFrame): Reale Cluster Märkte aus der Funktion kmeans_cluster_maerkte
        synth_maerkte (DataFrame): Synthetische Cluster Märkte aus der Funktion synth_maerkte oder synth_maerkte_custom (keine Gewährleistung)
    
    Returns:
        auswertung_maerkte (png): Histogramm Abgleich für die Variablen diff_article, avg_kolli, orders
    '''
    jetzt = datetime.now()

    real_maerkte['Kategorie'] = "real"
    synth_maerkte['Kategorie'] = "synth"

    maerkte = pd.concat([real_maerkte, synth_maerkte])

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes = axes.flatten()

    sns.histplot(data=maerkte, x="diff_article", hue="Kategorie", ax=axes[0], stat='density', legend=True)
    axes[0].set_title("Verteilung 'diff_article' zwischen realen und synthtischen Märkten")

    sns.histplot(data=maerkte, x="avg_kolli", hue="Kategorie", ax=axes[1], stat='density', legend=True)
    axes[1].set_title("Verteilung 'avg_kolli' zwischen realen und synthtischen Märkten")

    sns.histplot(data=maerkte, x="orders", hue="Kategorie", ax=axes[2], stat='density', legend=True)
    axes[2].set_title("Verteilung 'orders' zwischen realen und synthtischen Märkten")

    fig.savefig(f'reports/figures/auswertung_maerkte{jetzt:%Y_%m_%d_%H_%M}.png')

    return

def vergleich_real_synth_bestellungen(real_bestellungen_v2: pd.DataFrame, synth_bestellungen_v2: pd.DataFrame) -> pd.DataFrame:
    '''
    Vergleicht Lageparameter (Median, Mean), Streuung (Standardabweichung), Verteilung (KS-Test) und Abweichung (Wasserstein-Test)
    zwischen den realen Bestellungen und den synthetischen Betellungen pro generierten cluster
    Kennzahlen hängen teilweise von der Anzahl der generierten Märkten ab

    Args:
        real_bestellungen_v2 (DataFrame): Reale Cluster Bestellungen aus der Funktion kmeans_cluster_bestellungen
        synth_bestellungen_v2 (DataFrame): Synthetische Cluster Bestellungen aus der Funktion synth_bestellungen oder synth_bestellungen_custom (keine Gewährleistung)
    
    Returns:
        df_auswertung (DataFrame): Auswertungen Abgleich pro cluster zwischen real und synthetischen Bestellungen
    '''

    # Vorbereitung synth_bestellungen
    synth_bestellungen_v2 = synth_bestellungen_v2.rename(columns={
        'cluster_bestellungen': 'cluster'
    })

    # Kennzahlen welche verglichen werden sollen
    unq_variable = ['Wochentag', 'orderlines', 'diff_sortimente']
    unq_cluster = real_bestellungen_v2['cluster'].unique()

    rows = []

    # Für jede Variable
    for variable in unq_variable:

        # Für jedes cluster
        for cluster in unq_cluster:
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "median",
                "real": real_bestellungen_v2.loc[real_bestellungen_v2['cluster'] == cluster][variable].median(),
                "synth": synth_bestellungen_v2.loc[synth_bestellungen_v2['cluster'] == cluster][variable].median()
            })
        
        for cluster in unq_cluster:
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "mean",
                "real": real_bestellungen_v2.loc[real_bestellungen_v2['cluster'] == cluster][variable].mean(),
                "synth": synth_bestellungen_v2.loc[synth_bestellungen_v2['cluster'] == cluster][variable].mean()
            })

        for cluster in unq_cluster:
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "stdv",
                "real": real_bestellungen_v2.loc[real_bestellungen_v2['cluster'] == cluster][variable].std(),
                "synth": synth_bestellungen_v2.loc[synth_bestellungen_v2['cluster'] == cluster][variable].std()
            })

        for cluster in unq_cluster: 
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "Wasserstein Matrik",
                "real&synth": wasserstein_distance(
                real_bestellungen_v2.loc[real_bestellungen_v2['cluster'] == cluster][variable],
                synth_bestellungen_v2.loc[synth_bestellungen_v2['cluster'] == cluster][variable]
                )    
            })

        for cluster in unq_cluster:
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "KS-Test",
                "real&synth": ks_2samp(
                real_bestellungen_v2.loc[real_bestellungen_v2['cluster'] == cluster][variable],
                synth_bestellungen_v2.loc[synth_bestellungen_v2['cluster'] == cluster][variable]
                )    
            })

    df_auswertung = pd.DataFrame(rows)

    return df_auswertung

def vergleich_real_synth_bestellungen_visual(real_bestellungen: pd.DataFrame, synth_bestellungen:pd.DataFrame):
    '''
    Vergleicht die Verteilungen zwischen den realen und synthetischen Bestellungen für die Variablen Wochentag, orderlines, diff_sortimente in einem Histogramm
    Plots hängen von der Anzahl der generierten Märkten ab

    Args:
        real_bestellungen (DataFrame): Reale Cluster Bestellungen aus der Funktion kmeans_cluster_bestellungen
        synth_bestellungen (DataFrame): Synthetische Cluster Bestellungen aus der Funktion synth_bestellungen oder synth_bestellungen_custom (keine Gewährleistung)
    
    Returns:
        auswertung_bestellungen (png): Histogramm Abgleich für die Variablen Wochentag, orderlines, diff_sortimente
    '''
    jetzt = datetime.now()

    # Synth Bestellungen bereinigen
    synth_bestellungen = synth_bestellungen.drop(columns=['cluster_markt']).rename(columns={
        'cluster_bestellungen': 'cluster'
    }).reset_index()

    real_bestellungen['Kategorie'] = "real"
    synth_bestellungen['Kategorie'] = "synth"

    bestellungen = pd.concat([real_bestellungen, synth_bestellungen])

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes = axes.flatten()

    sns.histplot(data=bestellungen, x="Wochentag", hue="Kategorie", ax=axes[0], stat='density', legend=True)
    axes[0].set_title("Verteilung 'Wochentagen' zwischen realen und synthtischen Bestellungen")

    sns.histplot(data=bestellungen, x="orderlines", hue="Kategorie", ax=axes[1], stat='density', legend=True)
    axes[1].set_title("Verteilung 'orderlines' zwischen realen und synthtischen Bestellungen")

    sns.histplot(data=bestellungen, x="diff_sortimente", hue="Kategorie", ax=axes[2], stat='density', legend=True)
    axes[2].set_title("Verteilung 'diff_sortimente' zwischen realen und synthtischen Bestellungen")

    fig.savefig(f'reports/figures/auswertung_bestellungen{jetzt:%Y_%m_%d_%H_%M}.png')

    return

def vergleich_real_synth_bestellpositionen(real_bestellpositionen_v2: pd.DataFrame, synth_bestellpositionen_v2: pd.DataFrame) -> pd.DataFrame:
    '''
    Vergleicht Lageparameter (Median, Mean), Streuung (Standardabweichung), Verteilung (KS-Test) und Abweichung (Wasserstein-Test)
    zwischen den realen Bestellpositionen und den synthetischen Bestellpositionen 
    Kennzahlen hängen teilweise von der Anzahl der generierten Märkten ab

    Args:
        real_bestellpositionen_v2 (DataFrame): Reale Cluster Bestellpositionen aus dem original df
        synth_bestellpositionen_v2 (DataFrame): Synthetische Cluster Bestellpositionen aus der Funktion synth_bestellpositionen
    
    Returns:
        df_auswertung (DataFrame): Auswertungen Abgleich zwischen real und synthetischen bestellpositionen
    '''
    statistik = []

    statistik.append({
        "variable": "MengeInKolli",
        "metrik": "mean",
        "real": real_bestellpositionen_v2['MengeInKolli'].mean(),
        "synth": synth_bestellpositionen_v2['MengeInKolli'].mean(),
    })

    statistik.append({
        "variable": "MengeInKolli",
        "metrik": "median",
        "real": real_bestellpositionen_v2['MengeInKolli'].median(),
        "synth": synth_bestellpositionen_v2['MengeInKolli'].median(),
    })

    statistik.append({
        "variable": "MengeInKolli",
        "metrik": "stdv",
        "real": real_bestellpositionen_v2['MengeInKolli'].std(),
        "synth": synth_bestellpositionen_v2['MengeInKolli'].std(),
    })

    statistik.append({
        "variable": "MengeInKolli",
        "metrik": "Wasserstein Metrik",
        "real&synth": wasserstein_distance(
            real_bestellpositionen_v2["MengeInKolli"],
            synth_bestellpositionen_v2["MengeInKolli"]
            )    
    })
    
    statistik.append({
        "variable": "MengeInKolli",
        "metrik": "Wasserstein Metrik",
        "real&synth": ks_2samp(
            real_bestellpositionen_v2["MengeInKolli"],
            synth_bestellpositionen_v2["MengeInKolli"]
            )    
    })
    
    statistik.append({
        "variable": "Artikelnummer",
        "metrik": "nunique",
        "real": real_bestellpositionen_v2['Artikelnummer'].nunique(),
        "synth": synth_bestellpositionen_v2['Artikelnummer'].nunique(),
    })

    statistik.append({
        "variable": "Artikelnummer",
        "metrik": "mode",
        "real": real_bestellpositionen_v2['Artikelnummer'].mode(),
        "synth": synth_bestellpositionen_v2['Artikelnummer'].mode(),
    })

    df_auswertung = pd.DataFrame(statistik)
    
    return df_auswertung

def zusammenfassung_analysen_roh(real_maerkte_v2: pd.DataFrame,synth_maerkte_v2: pd.DataFrame,real_bestellungen_v2: pd.DataFrame,synth_bestellungen_v2: pd.DataFrame,real_bestellpositionen_v2: pd.DataFrame,synth_bestellpositionen_v2: pd.DataFrame) -> pd.DataFrame:

    df_maerkte = vergleich_real_synth_maerkte(real_maerkte_v2, synth_maerkte_v2).copy()
    df_maerkte["ebene"] = "maerkte"

    df_bestellungen = vergleich_real_synth_bestellungen(real_bestellungen_v2, synth_bestellungen_v2).copy()
    df_bestellungen["ebene"] = "bestellungen"

    df_bestellpos = vergleich_real_synth_bestellpositionen(real_bestellpositionen_v2, synth_bestellpositionen_v2).copy()
    df_bestellpos["ebene"] = "bestellpositionen"

    # concat nimmt automatisch die Union aller Spalten; fehlende werden NaN
    return pd.concat([df_maerkte, df_bestellungen, df_bestellpos], ignore_index=True, sort=False)