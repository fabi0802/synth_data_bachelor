import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.stats import chi2_contingency

def orderkopf_auswertung(orig_df, synth_df):
    # 1. orig_df auf orderkopf aggregieren
    orig_df = orig_df.groupby('order_id').agg({
        'Marktnummer': 'first',
        'Datum': 'first',
        'Artikelnummer': 'count', # Anzahl Orderlines zusammenfassen
        'cluster': 'first'  
    }).rename(columns={'Artikelnummer': 'orderlines'}).reset_index()

    # Allgemeine Kennzahlen Auswertung
    print(f'''Orderkopfbezogene Auswertung
        Anzahl Orders (real/synth) {len(orig_df)}, {len(synth_df)}
        Anzahl unterschiedlicher Tage (real/synth) {orig_df['Datum'].nunique()}, {synth_df['Datum'].nunique()}
        Summe orderlines (real/synth) {orig_df['orderlines'].sum()}, {synth_df['orderlines'].sum()}
        Anzahl unterschiedlicher Marktnummern (real/synth) {orig_df['Marktnummer'].nunique()}, {synth_df['Marktnummer'].nunique()}
        Anzahl orderlines pro Marktnummer (real / synth) {orig_df.groupby('Marktnummer')['orderlines'].sum().mean()},{synth_df.groupby('Marktnummer')['orderlines'].sum().mean()}
        Anzahl orderlines pro Tag (real/synth) {orig_df.groupby('Datum')['orderlines'].sum().mean()}, {synth_df.groupby('Datum')['orderlines'].sum().mean()}\n''')

    # Verteilungen Auswertung
    print(f'''Verteilungen Auswertung
        Kolmogorov Test orderlines: {ks_2samp(orig_df['orderlines'], synth_df['orderlines'])}
        Wasserstein-Distanz orderlines: {wasserstein_distance(orig_df['orderlines'], synth_df['orderlines'])}''')

    # Visueller Vergleich Cluster
    orig_df_cluster = orig_df.groupby('cluster')['orderlines'].sum().reset_index()
    orig_df_cluster['category'] = 'real'
    synth_df_cluster = synth_df.groupby('cluster')['orderlines'].sum().reset_index()
    synth_df_cluster['category'] = 'synth'
    cluster_compare = pd.concat([orig_df_cluster, synth_df_cluster], ignore_index=True)

    # Visueller Vergleich Datum
    orig_df_date = orig_df.groupby('Datum')['orderlines'].sum().reset_index().sort_values(by='Datum')
    synth_df_date = synth_df.groupby('Datum')['orderlines'].sum().reset_index().sort_values(by='Datum')

    # Subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes = axes.flatten()

    # 1. Balkendiagramm: orderlines pro cluster
    sns.barplot(data=cluster_compare, x='cluster', y='orderlines', hue='category', ax=axes[0])
    axes[0].set_title("Orderlines pro Cluster")
    axes[0].set_ylabel("Summe Orderlines")

    # 2. ECDF
    sns.ecdfplot(data=orig_df, x='orderlines', label='Original', ax=axes[1])
    sns.ecdfplot(data=synth_df, x='orderlines', label='Synthetisch', ax=axes[1])
    axes[1].set_title("Verteilung der Orderlines")
    axes[1].set_xlabel("Orderlines pro Bestellung")
    axes[1].legend()

    # 3. Zeitverlauf
    sns.lineplot(data=orig_df_date, x='Datum', y='orderlines', label='Original', ax=axes[2])
    sns.lineplot(data=synth_df_date, x='Datum', y='orderlines', label='Synthetisch', ax=axes[2])
    axes[2].set_title("Orderlines über Zeit")
    axes[2].set_ylabel("Orderlines")
    axes[2].legend()

    # Optional: 4. Plot freilassen oder für weitere Metriken nutzen
    axes[3].axis('off')  # leer lassen, falls nicht verwendet

    plt.tight_layout()
    plt.show()


def orderline_auswertung(original_orderlines:pd.DataFrame, synth_orderlines:pd.DataFrame):

    # Vergleich auf Orderline Ebene
    orig_sum_menge = original_orderlines.groupby('cluster')['MengeInKolli'].sum()
    synth_sum_menge = synth_orderlines.groupby('cluster')['MengeInKolli'].sum()

    orig_count_artikel = original_orderlines.groupby('cluster')['Artikelnummer'].count()
    synth_count_artikel = synth_orderlines.groupby('cluster')['Artikelnummer'].count()

    orig_sum_menge_tag = original_orderlines.groupby('Datum')['MengeInKolli'].sum()
    synth_sum_menge_tag = synth_orderlines.groupby('Datum')['MengeInKolli'].sum()

    orig_count_artikel_day = original_orderlines.groupby('Datum')['Artikelnummer'].count()
    synth_count_artikel_day = synth_orderlines.groupby('Datum')['Artikelnummer'].count()

    print(f'''Orderline bezogen Auswertung
    Summe MengeInKolli (real/synth): {original_orderlines['MengeInKolli'].sum()} , {synth_orderlines['MengeInKolli'].sum()}
    Durchschnitt MengeInKolli (real/synth): {original_orderlines['MengeInKolli'].mean()}, {synth_orderlines['MengeInKolli'].mean()}
    Durschnitt MengeInKolli pro Cluster (real/synth): {orig_sum_menge.mean()}, {synth_sum_menge.mean()}
    Durschnitt Anzahl Artikelnummern pro Cluster (real/synth): {orig_count_artikel.mean()}, {synth_count_artikel.mean()}\n''')

    print(f'''Tagesbezogene Auswertung
    Summe MengeInKolli pro Tag (real/synth): {orig_sum_menge_tag.mean()}, {synth_sum_menge_tag.mean()}
    Durchschnitt Artikelnummern pro Tag (real/synth): {orig_count_artikel_day.mean()}, {synth_count_artikel_day.mean()} \n''')
    
    # Auswertung der Verteilung
    print(f'''Verteilung Auswertung
    FALSCHE ANWENDUNG Kolmogorov Test (MengeInKolli) pro Cluster: {ks_2samp(orig_sum_menge, synth_sum_menge)}
    FALSCHE ANWENDUNG Wasserstein-Distanz (MengeInKolli) pro Cluster: {wasserstein_distance(orig_sum_menge, synth_sum_menge)}
    FALSCHE ANWENDUNG Kolmogorov Test (Artikelnummern) pro Cluster: {ks_2samp(orig_count_artikel, synth_count_artikel)}
    FALSCHE ANWENDUNG Wasserstein-Distanz (Artikelnummern) pro Cluster: {wasserstein_distance(orig_count_artikel, synth_count_artikel)}
    ''')
    
    #Kolmogorov-Sminov-Test
    print(f"Kolmogorov Test (MengeInKolli): {ks_2samp(original_orderlines['MengeInKolli'], synth_orderlines['MengeInKolli'])}")

    # Wasserstein Distanz Test
    print(f"Wasserstein-Distanz (MengeInKolli): {wasserstein_distance(original_orderlines['MengeInKolli'], synth_orderlines['MengeInKolli'])}\n")

    #Anwendung chi2 test auf Markierung
    real_counts = original_orderlines.value_counts().sort_index()
    synth_counts = synth_orderlines.value_counts().sort_index()

    contingency_table = pd.concat([real_counts, synth_counts], axis=1)
    contingency_table.columns = ['real', 'synth']
    contingency_table.fillna(0, inplace=True)

    chi2, p_value, dof, expected = chi2_contingency(contingency_table.T)

    print(f"Chi²-Statistik (Markierung): {chi2}")
    print(f"p-Wert (Markierung): {p_value}")

    # Vorbereitung plots
    original_orderlines_cluster = original_orderlines.groupby('cluster')['MengeInKolli'].sum().reset_index()
    original_orderlines_cluster['category'] = 'real'
    synth_orderlines_cluster = synth_orderlines.groupby('cluster')['MengeInKolli'].sum().reset_index()
    synth_orderlines_cluster['category'] = 'synth'
    cluster_compare = pd.concat([original_orderlines_cluster, synth_orderlines_cluster])

    orig_df_date = original_orderlines.groupby('Datum')['MengeInKolli'].sum().reset_index()
    synth_df_date = synth_orderlines.groupby('Datum')['MengeInKolli'].sum().reset_index()

    orig_hist = original_orderlines.groupby('order_id')['MengeInKolli'].sum().reset_index()
    orig_hist['category'] = 'real'
    synth_hist = synth_orderlines.groupby('order_id')['MengeInKolli'].sum().reset_index()
    synth_hist['category'] = 'synth'
    concat_hist = pd.concat([synth_hist, orig_hist])
    concat_hist['order_id'] = concat_hist['order_id'].astype(str)


    # Subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes = axes.flatten()

    # 1. Balkendiagramm: menge pro cluster
    sns.barplot(data=cluster_compare, x='cluster', y='MengeInKolli', hue='category', ax=axes[0])
    axes[0].set_title("MengeInKolli pro Cluster")
    axes[0].set_ylabel("Summe Kolli")

    # 2. ECDF
    sns.ecdfplot(data=original_orderlines, x='MengeInKolli', label='Original', ax=axes[1])
    sns.ecdfplot(data=synth_orderlines, x='MengeInKolli', label='Synthetisch', ax=axes[1])
    axes[1].set_title("Verteilung der MengeInKolli auf Orderline Ebene")
    axes[1].set_xlabel("MengeInKolli pro Orderline")
    axes[1].legend()

   # 3. Zeitverlauf
    sns.lineplot(data=orig_df_date, x='Datum', y='MengeInKolli', label='Original', ax=axes[2])
    sns.lineplot(data=synth_df_date, x='Datum', y='MengeInKolli', label='Synthetisch', ax=axes[2])
    axes[2].set_title("MengeInKolli über Zeit")
    axes[2].set_ylabel("MengeInKolli")
    axes[2].legend()


    # Histogramm Kolii pro Orderlines
    sns.histplot(
    data=concat_hist, x='MengeInKolli', hue='category', ax=axes[3], bins=30, multiple='dodge')
    axes[3].set_title('Histogramm MengeInKolli pro Order')
    #axes[3].axis('off')  # leer lassen, falls nicht verwendet

    plt.show()

    