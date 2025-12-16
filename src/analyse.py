import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.stats import chi2_contingency
from datetime import datetime

def vergleich_real_synth_maerkte(real_maerkte_v2: pd.DataFrame, synth_maerkte_v2: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for i in range(0,3):

        real_maerkte = real_maerkte_v2.copy()
        real_maerkte = real_maerkte[real_maerkte['cluster'] == i]

        synth_maerkte = synth_maerkte_v2.copy()
        synth_maerkte = synth_maerkte[synth_maerkte['cluster'] == i]

        rows.append({
            "cluster": i,
            "variable": "diff_article",
            "metrik": "median",
            "real": real_maerkte['diff_article'].median(),
            "synth": synth_maerkte['diff_article'].median()
        })

        rows.append({
            "cluster": i,
            "variable": "diff_article",
            "metrik": "stdv",
            "real": real_maerkte['diff_article'].std(),
            "synth": synth_maerkte['diff_article'].std()
        })

        rows.append({
            "cluster": i,
            "variable": "diff_article",
            "metrik": "kolmogorov-smirnov Test",
            "real & synth": ks_2samp(real_maerkte["diff_article"], synth_maerkte["diff_article"])
        })

        rows.append({
            "cluster": i,
            "variable": "diff_article",
            "metrik": "Wasserstein Distanz Test",
            "real & synth": wasserstein_distance(real_maerkte["diff_article"], synth_maerkte["diff_article"])
        })

        rows.append({
            "cluster": i,
            "variable": "avg_kolli",
            "metrik": "median",
            "real": real_maerkte['avg_kolli'].median(),
            "synth": synth_maerkte['avg_kolli'].median()
        })

        rows.append({
            "cluster": i,
            "variable": "avg_kolli",
            "metrik": "stdv",
            "real": real_maerkte['avg_kolli'].std(),
            "synth": synth_maerkte['avg_kolli'].std()
        })

        rows.append({
            "cluster": i,
            "variable": "avg_kolli",
            "metrik": "kolmogorov-smirnov Test",
            "real & synth": ks_2samp(real_maerkte["avg_kolli"], synth_maerkte["avg_kolli"])
        })

        rows.append({
            "cluster": i,
            "variable": "avg_kolli",
            "metrik": "Wasserstein Distanz Test",
            "real & synth": wasserstein_distance(real_maerkte["avg_kolli"], synth_maerkte["avg_kolli"])
        })

        rows.append({
            "cluster": i,
            "variable": "orders",
            "metrik": "mean",
            "real": real_maerkte['orders'].median(),
            "synth": synth_maerkte['orders'].median()
        })

        rows.append({
            "cluster": i,
            "variable": "orders",
            "metrik": "stdv",
            "real": real_maerkte['orders'].std(),
            "synth": synth_maerkte['orders'].std()
        })

        rows.append({
            "cluster": i,
            "variable": "orders",
            "metrik": "kolmogorov-smirnov Test",
            "real & synth": ks_2samp(real_maerkte["orders"], synth_maerkte["orders"])
        })

        rows.append({
            "cluster": i,
            "variable": "orders",
            "metrik": "Wasserstein Distanz Test",
            "real & synth": wasserstein_distance(real_maerkte["orders"], synth_maerkte["orders"])
        })


    df_auswertung = pd.DataFrame(rows)

    return df_auswertung

def vergleich_real_synth_maerkte_visual(real_maerkte: pd.DataFrame, synth_maerkte: pd.DataFrame):
    
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

    pass