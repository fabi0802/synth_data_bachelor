import pandas as pd
from datetime import datetime
from maerkte import kmeans_cluster_maerkte, synth_maerkte
from bestellungen import kmeans_cluster_bestellungen, synth_bestellungen
from bestellpositionen import synth_bestellpositionen
from analyse import vergleich_real_synth_maerkte, vergleich_real_synth_maerkte_visual, vergleich_real_synth_bestellungen, vergleich_real_synth_bestellungen_visual

jetzt = datetime.now()
generiere_kunden = 41

df = pd.read_parquet('data/raw/orderlines_dezember.parquet')

# Bestehende Märkte zu Clustern zusammenfassen
df_maerkte = df.copy()
df_maerkte = kmeans_cluster_maerkte(df_maerkte)

# Neue Märkte generieren (bspw. 50 oder 100)
df_neue_maerkte = synth_maerkte(df_maerkte, generiere_kunden)

# Bestehende Bestellungen zu Clustern zusammenfassen
df_bestellungen = df.copy()
df_bestellungen = kmeans_cluster_bestellungen(df_bestellungen)

# Neue Bestellungen generieren (Anzahl orientiert sich anhand der generierten Märkten)
df_neue_bestellungen = synth_bestellungen(df_bestellungen, df_maerkte, df_neue_maerkte)

# Neue Bestellpositionen generieren & abspeichern
df_neue_bestellpositionen = synth_bestellpositionen(df, df_bestellungen, df_neue_bestellungen)

df_neue_bestellpositionen.to_parquet(f'data/processed/synth_data_{jetzt:%Y_%m_%d_%H_%M}.parquet')

# Kennzahlen Vergleich zwischen realen und synthetischen Märkten
Abgleich_maerkte = vergleich_real_synth_maerkte(df_maerkte, df_neue_maerkte)
Abgleich_maerkte.to_excel(f'reports/summary/markt_kennzahlen_{jetzt:%Y_%m_%d_%H_%M}.xlsx')

# Visuelle Verteilungen zwischen realen & synthetischen Märkten
vergleich_real_synth_maerkte_visual(df_maerkte, df_neue_maerkte)

# Kennzhalen Vergleich zwischen realen und synthetischen Bestellungen
Abgleich_bestellungen = vergleich_real_synth_bestellungen(df_bestellungen, df_neue_bestellungen)
Abgleich_bestellungen.to_excel(f'reports/summary/bestellkopf_kennzahlen_{jetzt:%Y_%m_%d_%H_%M}.xlsx')

# Visuelle Verteilungen zwischen realen & synthetischen Bestellungen
vergleich_real_synth_bestellungen_visual(df_bestellungen, df_neue_bestellungen)
