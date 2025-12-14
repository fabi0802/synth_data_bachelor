import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.stats import chi2_contingency

def vergleich_real_synth_maerkte(real_maerkte: pd.DataFrame, synth_maerkte: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "variable": "Marktnummer",
        "metrik": "Anzahl",
        "real": real_maerkte['Marktnummer'].count(),
        "synth": synth_maerkte['Marktnummer'].count()
    })

    rows.append({
        "variable": "diff_article",
        "metrik": "median",
        "real": real_maerkte['diff_article'].median(),
        "synth": synth_maerkte['diff_article'].median()
    })

    rows.append({
        "variable": "diff_article",
        "metrik": "stdv",
        "real": real_maerkte['diff_article'].std(),
        "synth": synth_maerkte['diff_article'].std()
    })

    rows.append({
        "variable": "avg_kolli",
        "metrik": "median",
        "real": real_maerkte['avg_kolli'].median(),
        "synth": synth_maerkte['avg_kolli'].median()
    })

    rows.append({
        "variable": "avg_kolli",
        "metrik": "stdv",
        "real": real_maerkte['avg_kolli'].std(),
        "synth": synth_maerkte['avg_kolli'].std()
    })

    rows.append({
        "variable": "orders",
        "metrik": "mean",
        "real": real_maerkte['orders'].mean(),
        "synth": synth_maerkte['orders'].mean()
    })

    rows.append({
        "variable": "orders",
        "metrik": "stdv",
        "real": real_maerkte['orders'].std(),
        "synth": synth_maerkte['orders'].std()
    })

    df_auswertung = pd.DataFrame(rows)

    return df_auswertung
