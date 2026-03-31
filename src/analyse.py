import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from datetime import datetime

def vergleich_real_synth_maerkte(real_maerkte_v2: pd.DataFrame, synth_maerkte_v2: pd.DataFrame, resampling_maerkte_v2: pd.DataFrame) -> pd.DataFrame:
    '''
    Vergleicht Lageparameter (Median), Streuung (Standardabweichung), Verteilung (KS-Test) und Abweichung (Wasserstein-Test)
    zwischen den realen Märkten und den synthetischen Märkten pro generierten cluster
    Kennzahlen hängen teilweise von der Anzahl der generierten Märkten ab

    Args:
        real_maerkte_v2 (DataFrame): Reale Cluster Märkte aus der Funktion kmeans_cluster_maerkte
        synth_maerkte_v2 (DataFrame): Synthetische Cluster Märkte aus der Funktion synth_maerkte oder synth_maerkte_custom (keine Gewährleistung)
        resampling_maerkte_v2 (DataFrame): Resamplet Cluster Märkte aus der Funktion resampling_maerkte
    
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
                "synth": synth_maerkte_v2.loc[synth_maerkte_v2['cluster'] == cluster][variable].median(),
                "resample": resampling_maerkte_v2.loc[resampling_maerkte_v2['cluster'] == cluster][variable].median(),
            })

        for cluster in unq_cluster: 
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "stdv",
                "real": real_maerkte_v2.loc[real_maerkte_v2['cluster'] == cluster][variable].std(),
                "synth": synth_maerkte_v2.loc[synth_maerkte_v2['cluster'] == cluster][variable].std(),
                "resample": resampling_maerkte_v2.loc[resampling_maerkte_v2['cluster'] == cluster][variable].std()
            })

        for cluster in unq_cluster: 
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "Wasserstein Matrik",
                "real&synth": wasserstein_distance(
                real_maerkte_v2.loc[real_maerkte_v2['cluster'] == cluster][variable],
                synth_maerkte_v2.loc[synth_maerkte_v2['cluster'] == cluster][variable]
                ),
                "real&resample": wasserstein_distance(
                real_maerkte_v2.loc[real_maerkte_v2['cluster'] == cluster][variable],
                resampling_maerkte_v2.loc[resampling_maerkte_v2['cluster'] == cluster][variable]
                ),    
            })
        
        for cluster in unq_cluster:
            rows.append({
                "cluster": cluster,
                "variable": variable,
                "metrik": "KS-Test",
                "real&synth": ks_2samp(
                real_maerkte_v2.loc[real_maerkte_v2['cluster'] == cluster][variable],
                synth_maerkte_v2.loc[synth_maerkte_v2['cluster'] == cluster][variable]
                ),
                "real&resample": ks_2samp(
                real_maerkte_v2.loc[real_maerkte_v2['cluster'] == cluster][variable],
                resampling_maerkte_v2.loc[resampling_maerkte_v2['cluster'] == cluster][variable]
                ),     
            })

    df_auswertung = pd.DataFrame(rows)

    return df_auswertung

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

def vergleich_real_synth_bestellpositionen(real_bestellpositionen_v2: pd.DataFrame, synth_bestellpositionen_v2: pd.DataFrame, resampling_bestellpositionen_v2: pd.DataFrame) -> pd.DataFrame:
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
        "resample": resampling_bestellpositionen_v2["MengeInKolli"].mean(),
    })

    statistik.append({
        "variable": "MengeInKolli",
        "metrik": "median",
        "real": real_bestellpositionen_v2['MengeInKolli'].median(),
        "synth": synth_bestellpositionen_v2['MengeInKolli'].median(),
        "resample": resampling_bestellpositionen_v2["MengeInKolli"].median(),
    })

    statistik.append({
        "variable": "MengeInKolli",
        "metrik": "stdv",
        "real": real_bestellpositionen_v2['MengeInKolli'].std(),
        "synth": synth_bestellpositionen_v2['MengeInKolli'].std(),
        "resample": resampling_bestellpositionen_v2["MengeInKolli"].std(),
    })

    statistik.append({
        "variable": "MengeInKolli",
        "metrik": "Wasserstein Metrik",
        "real&synth": wasserstein_distance(
            real_bestellpositionen_v2["MengeInKolli"],
            synth_bestellpositionen_v2["MengeInKolli"]
            ),
        "real&resample": wasserstein_distance(
            real_bestellpositionen_v2["MengeInKolli"],
            resampling_bestellpositionen_v2["MengeInKolli"]
            ),  
    })
    
    statistik.append({
        "variable": "MengeInKolli",
        "metrik": "Wasserstein Metrik",
        "real&synth": ks_2samp(
            real_bestellpositionen_v2["MengeInKolli"],
            synth_bestellpositionen_v2["MengeInKolli"]
            ),
        "real&resample": ks_2samp(
            real_bestellpositionen_v2["MengeInKolli"],
            resampling_bestellpositionen_v2["MengeInKolli"]
            )      
    })
    
    statistik.append({
        "variable": "Artikelnummer",
        "metrik": "nunique",
        "real": real_bestellpositionen_v2['Artikelnummer'].nunique(),
        "synth": synth_bestellpositionen_v2['Artikelnummer'].nunique(),
        "resample": resampling_bestellpositionen_v2['Artikelnummer'].nunique(),
    })

    statistik.append({
        "variable": "Artikelnummer",
        "metrik": "mode",
        "real": real_bestellpositionen_v2['Artikelnummer'].mode(),
        "synth": synth_bestellpositionen_v2['Artikelnummer'].mode(),
        "resample": resampling_bestellpositionen_v2['Artikelnummer'].mode(),
    })

    df_auswertung = pd.DataFrame(statistik)
    
    return df_auswertung

def zusammenfassung_analysen_roh(real_maerkte_v2: pd.DataFrame,synth_maerkte_v2: pd.DataFrame, resampling_maerkte_v2: pd.DataFrame ,real_bestellungen_v2: pd.DataFrame,synth_bestellungen_v2: pd.DataFrame,real_bestellpositionen_v2: pd.DataFrame,synth_bestellpositionen_v2: pd.DataFrame, resampling_bestellpositionen_v2: pd.DataFrame) -> pd.DataFrame:

    df_maerkte = vergleich_real_synth_maerkte(real_maerkte_v2, synth_maerkte_v2, resampling_maerkte_v2).copy()
    df_maerkte["ebene"] = "maerkte"

    df_bestellungen = vergleich_real_synth_bestellungen(real_bestellungen_v2, synth_bestellungen_v2).copy()
    df_bestellungen["ebene"] = "bestellungen"

    df_bestellpos = vergleich_real_synth_bestellpositionen(real_bestellpositionen_v2, synth_bestellpositionen_v2, resampling_bestellpositionen_v2).copy()
    df_bestellpos["ebene"] = "bestellpositionen"

    # concat nimmt automatisch die Union aller Spalten; fehlende werden NaN
    return pd.concat([df_maerkte, df_bestellungen, df_bestellpos], ignore_index=True, sort=False)

def avg_kolli_visual(real_maerkte:pd.DataFrame, synth_maerkte:pd.DataFrame, sampling_maerkte:pd.DataFrame):

    jetzt = datetime.now()
    real_maerkte['Kategorie'] = "real"
    synth_maerkte['Kategorie'] = "synth"
    sampling_maerkte['Kategorie'] = "sampling"

    maerkte = pd.concat([real_maerkte, synth_maerkte, sampling_maerkte])

    plt.figure(figsize=(9, 5))
    ax =  sns.boxplot(data=maerkte, x='cluster', y='avg_kolli', hue='Kategorie')
    ax.set_ylim(0,10)
    plt.title("Verteilung von avg_kolli nach Marktcluster")
    plt.xlabel("Cluster")
    plt.ylabel("avg_kolli")
    plt.tight_layout()
    plt.savefig(f'reports/figures/auswertung_avg_kolli_{jetzt:%Y_%m_%d_%H_%M}.png', dpi=300, bbox_inches='tight')
    plt.close()

def diff_article_visual(real_maerkte:pd.DataFrame, synth_maerkte:pd.DataFrame, sampling_maerkte:pd.DataFrame):

    jetzt = datetime.now()
    real_maerkte['Kategorie'] = "real"
    synth_maerkte['Kategorie'] = "synth"
    sampling_maerkte['Kategorie'] = "sampling"

    maerkte = pd.concat([real_maerkte, synth_maerkte, sampling_maerkte])

    plt.figure(figsize=(9, 5))
    sns.violinplot(data=maerkte, x='cluster', y='diff_article', hue='Kategorie', split=False, inner='quartile')
    plt.title("Verteilung von diff_article nach Marktcluster")
    plt.xlabel("Cluster")
    plt.ylabel("diff_article")
    plt.tight_layout()
    plt.savefig(f'reports/figures/auswertung_diff_article_{jetzt:%Y_%m_%d_%H_%M}.png', dpi=300, bbox_inches='tight')
    plt.close()

def orderlines_visual(real_bestellungen:pd.DataFrame, synth_bestellungen:pd.DataFrame, sampling_bestellungen:pd.DataFrame):

    jetzt = datetime.now()
    real_bestellungen['Kategorie'] = "real"
    synth_bestellungen['Kategorie'] = "synth"
    sampling_bestellungen['Kategorie'] = "sampling"

    bestellungen = pd.concat([real_bestellungen, synth_bestellungen, sampling_bestellungen])

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=bestellungen, x='Kategorie', y='orderlines', inner='quartile')
    plt.title("Verteilung der Orderlines")
    plt.xlabel("Verfahren")
    plt.ylabel("orderlines")
    plt.tight_layout()
    plt.savefig(f'reports/figures/auswertung_orderlines_{jetzt:%Y_%m_%d_%H_%M}.png', dpi=300, bbox_inches='tight')
    plt.close()

def tagesvolumen_visual(real_bestellungen:pd.DataFrame, synth_bestellungen:pd.DataFrame, sampling_bestellungen:pd.DataFrame):

    jetzt = datetime.now()
    real_bestellungen['Kategorie'] = "real"
    synth_bestellungen['Kategorie'] = "synth"
    sampling_bestellungen['Kategorie'] = "sampling"

    bestellungen = pd.concat([real_bestellungen, synth_bestellungen, sampling_bestellungen])

    bestellungen = bestellungen.groupby((['Datum','Kategorie']))['orderlines'].sum().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=bestellungen, x='Datum', y='orderlines', hue='Kategorie')
    plt.title("Verteilung der Orderlines über Tage")
    plt.xlabel("Datum")
    plt.ylabel("orderlines")
    plt.tight_layout()
    plt.savefig(f'reports/figures/auswertung_orderlines_tagesvolumen{jetzt:%Y_%m_%d_%H_%M}.png', dpi=300, bbox_inches='tight')
    plt.close()

def artikelvolumen_visual(real_bestellpositionen:pd.DataFrame, synth_bestellpositionen:pd.DataFrame, sampling_bestellpositionen:pd.DataFrame):

    jetzt = datetime.now()
    real_bestellpositionen['Kategorie'] = "real"
    synth_bestellpositionen['Kategorie'] = "synth"
    sampling_bestellpositionen['Kategorie'] = "sampling"

    bestellungen = pd.concat([real_bestellpositionen, synth_bestellpositionen, sampling_bestellpositionen])
    bestellungen = bestellungen.groupby((['Artikelnummer', 'Kategorie']))['MengeInKolli'].sum().reset_index()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=bestellungen, x='Artikelnummer', y='MengeInKolli', hue='Kategorie')
    plt.title("Verteilung der Artikelnummern")
    plt.xlabel("Artikelnummer")
    plt.ylabel("MengeInKolli")
    plt.tight_layout()
    plt.savefig(f'reports/figures/auswertung_artikelvolumen{jetzt:%Y_%m_%d_%H_%M}.png', dpi=300, bbox_inches='tight')
    plt.close()

def korrelation_maerkte(df_maerkte:pd.DataFrame, synth_maerkte:pd.DataFrame, resamp_maerkte:pd.DataFrame, method="pearson") -> pd.DataFrame:
   
    df_real = df_maerkte[['diff_article', 'orders', 'avg_kolli']]
    df_synth = synth_maerkte[['diff_article', 'orders', 'avg_kolli']]
    df_resamp = resamp_maerkte[['diff_article', 'orders', 'avg_kolli']]

    corr_real = df_real.corr(method=method)
    corr_syn1 = df_synth.corr(method=method)
    corr_syn2 = df_resamp.corr(method=method)

    delta_syn1 = corr_syn1 - corr_real
    delta_syn2 = corr_syn2 - corr_real

    abs_delta_syn1 = np.abs(delta_syn1)
    abs_delta_syn2 = np.abs(delta_syn2)

    summary_df = pd.DataFrame({
        "dataset": ["synth", "resamp"],
        "mean_abs_delta": [
            abs_delta_syn1.values.mean(),
            abs_delta_syn2.values.mean()
        ],
        "max_abs_delta": [
            abs_delta_syn1.values.max(),
            abs_delta_syn2.values.max()
        ]
    })

    return summary_df

def korrelation_bestellungen(df_bestellungen:pd.DataFrame, synth_bestellungen:pd.DataFrame, resamp_bestellungen:pd.DataFrame, method="pearson") -> pd.DataFrame:
   
    df_real = df_bestellungen[['Wochentag', 'orderlines', 'diff_sortimente']]
    df_synth = synth_bestellungen[['Wochentag', 'orderlines', 'diff_sortimente']]
    df_resamp = resamp_bestellungen[['Wochentag', 'orderlines', 'diff_sortimente']]

    corr_real = df_real.corr(method=method)
    corr_syn1 = df_synth.corr(method=method)
    corr_syn2 = df_resamp.corr(method=method)

    delta_syn1 = corr_syn1 - corr_real
    delta_syn2 = corr_syn2 - corr_real

    abs_delta_syn1 = np.abs(delta_syn1)
    abs_delta_syn2 = np.abs(delta_syn2)

    summary_df = pd.DataFrame({
        "dataset": ["synth", "resamp"],
        "mean_abs_delta": [
            abs_delta_syn1.values.mean(),
            abs_delta_syn2.values.mean()
        ],
        "max_abs_delta": [
            abs_delta_syn1.values.max(),
            abs_delta_syn2.values.max()
        ]
    })

    return summary_df