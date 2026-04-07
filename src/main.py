import pandas as pd
from datetime import datetime
from maerkte import kmeans_cluster_maerkte, synth_maerkte, synth_maerkte_custom
from bestellungen import kmeans_cluster_bestellungen, synth_bestellungen, synth_bestellungen_custom
from bestellpositionen import synth_bestellpositionen
from analyse import  zusammenfassung_analysen_roh, avg_kolli_visual, orderlines_visual, diff_article_visual, tagesvolumen_visual, artikelvolumen_visual, korrelation_bestellungen, korrelation_maerkte
from resampling import resampling_hochrechnung, resampling_maerkte, resampling_bestellungen

jetzt = datetime.now()
anzahl_neue_maerkte = 409

df = pd.read_parquet('data/raw/orderlines_dezember.parquet')


# Bestehende Märkte zu Clustern zusammenfassen
df_maerkte = df.copy()
df_maerkte = kmeans_cluster_maerkte(df_maerkte)


# Neue Märkte generieren auf Basis der bisherigen Verteilung in den Originaldaten
df_neue_maerkte = synth_maerkte(
    df_maerkte, 
    anzahl_neue_maerkte
    )


# Bestehende Bestellungen zu Clustern zusammenfassen
df_bestellungen = df.copy()
df_bestellungen = kmeans_cluster_bestellungen(df_bestellungen)


# Neue Bestellungen generieren auf Basis der bisherigen Verteilung in den Originaldaten
df_neue_bestellungen = synth_bestellungen(
    df_bestellungen, 
    df_maerkte, 
    df_neue_maerkte)


# Neue Bestellpositionen generieren & abspeichern
df_neue_bestellpositionen = synth_bestellpositionen(
    df, 
    df_bestellungen, 
    df_neue_bestellungen
    )

df_neue_bestellpositionen.to_parquet(f'data/processed/synth_data_{jetzt:%Y_%m_%d_%H_%M}.parquet')


# Resampling der existierenden Bestellungen und Aggregierung auf Markt oder Bestellebene
df_resample = resampling_hochrechnung(df,anzahl_neue_maerkte)
df_resample_maerkte = resampling_maerkte(df_resample)
df_resample_bestellungen = resampling_bestellungen(df_resample)

df_resample.to_parquet(f'data/processed/resample_data_{jetzt:%Y_%m_%d_%H_%M}.parquet')


# Visuelle Darstellugnen und Speicherungen

avg_kolli_visual(
    df_maerkte, 
    df_neue_maerkte, 
    df_resample_maerkte
    )

orderlines_visual(
    df_bestellungen,
    df_neue_bestellungen,
    df_resample_bestellungen
    )

diff_article_visual(
    df_maerkte, 
    df_neue_maerkte, 
    df_resample_maerkte
)

tagesvolumen_visual(
    df_bestellungen,
    df_neue_bestellungen,
    df_resample_bestellungen
)

artikelvolumen_visual(
    df,
    df_neue_bestellpositionen,
    df_resample
)

# Korrelation auf Marktebene vergleichen
df_corr_maerkte = korrelation_maerkte(
    df_maerkte,
    df_neue_maerkte,
    df_resample_maerkte
)

df_corr_maerkte.to_excel(f'reports/summary/corr_maerkte{jetzt:%Y_%m_%d_%H_%M}.xlsx')

# Korrelation auf Bestellebene vergleichen
df_corr_bestellungen = korrelation_bestellungen(
    df_bestellungen,
    df_neue_bestellungen,
    df_resample_bestellungen
    )

df_corr_bestellungen.to_excel(f'reports/summary/corr_bestellungen{jetzt:%Y_%m_%d_%H_%M}.xlsx')

# Kennzahlen zu Märkten, Bestellungen und Bestellpositionen
roh_analyse = zusammenfassung_analysen_roh(
    df_maerkte,
    df_neue_maerkte,
    df_resample_maerkte,
    df_bestellungen,df_neue_bestellungen,
    df,
    df_neue_bestellpositionen, 
    df_resample
    )

roh_analyse.to_excel(f'reports/summary/roh_analyse_{jetzt:%Y_%m_%d_%H_%M}.xlsx')


print('Durchlauf ist fertig')