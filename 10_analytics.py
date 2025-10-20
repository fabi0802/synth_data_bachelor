import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance

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