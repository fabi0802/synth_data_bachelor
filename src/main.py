import pandas as pd
from datetime import datetime
from maerkte import kmeans_cluster_maerkte, synth_maerkte, synth_maerkte_custom
from bestellungen import kmeans_cluster_bestellungen, synth_bestellungen, synth_bestellungen_custom
from bestellpositionen import synth_bestellpositionen, synth_bestellpositionen_sortiment
from analyse import vergleich_real_synth_maerkte, vergleich_real_synth_maerkte_visual, vergleich_real_synth_bestellungen, vergleich_real_synth_bestellungen_visual, vergleich_real_synth_bestellpositionen
from abc_klasse import abc_analyse, abc_analyse_plot, abc_zusammenfassung_neu

jetzt = datetime.now()
anzahl_neue_maerkte = 400

df = pd.read_parquet('data/raw/orderlines_dezember.parquet')


# Bestehende Märkte zu Clustern zusammenfassen
df_maerkte = df.copy()
df_maerkte = kmeans_cluster_maerkte(df_maerkte)

# Neue Märkte generieren auf Basis der bisherigen Verteilung in den Originaldaten
df_neue_maerkte = synth_maerkte(df_maerkte, anzahl_neue_maerkte)

'''Alternative zur standard Markt Generierung (synth_maerkte):

Aktiver Eingriff in die Markt Synthetisierung durch die Veränderung der relativen Markcluster Anteile.
Relativer Anteil bestimmt, wie viele Märkte vom jeweiligen Cluster erstellt werden. 

Kategorisierung für den Marktmix:
cluster_0 = Drittkunden (geringe Artikelvielfalt, hohe Kolli Menge)
cluster_1 = REWE Center (extrem hohe Artikelvielfalt, kleine Menge je Artikel)
cluster_2 = Standard REWE (hohe Artikelvielfalt, viele Bestellungen, kleine Kolli Mengen)'''

# marktmix = {
#     0: 0.005,
#     1: 0.99,
#     2: 0.005
# }
# df_neue_maerkte = synth_maerkte_custom(df_maerkte, generiere_kunden, marktmix)


# Bestehende Bestellungen zu Clustern zusammenfassen
df_bestellungen = df.copy()
df_bestellungen = kmeans_cluster_bestellungen(df_bestellungen)

# Neue Bestellungen generieren auf Basis der bisherigen Verteilung in den Originaldaten
df_neue_bestellungen = synth_bestellungen(df_bestellungen, df_maerkte, df_neue_maerkte)

'''Alternative zur standard Bestellungen Generierung (synth_bestellungen):

Aktiver Eingriff in die Bestell-Synthetisierung durch die Veränderung der relativen Bestellcluster Anteile.
Relativer Anteil bestimmt, wie viele Bestellungen vom jeweiligen Cluster erstellt werden. 

Kategorisierung für den Bestellmix:
cluster_0 = Großbestellungen (weniger Bestellungen, breite Sortimente pro Bestellung)
cluster_1 = Regelbestellungen (viele Bestellungen, geringe Menge pr Bestellung, breite Sortimente)'''

# bestellmix = {
#   0: {0: 0.1, 1: 0.9},
#   1: {0: 0.1, 1: 0.9},
#   2: {0: 0.1, 1: 0.9}
# }
# df_neue_bestellungen = synth_bestellungen_custom(df_bestellungen, df_maerkte, df_neue_maerkte, bestellmix)


# Neue Bestellpositionen generieren & abspeichern
df_neue_bestellpositionen = synth_bestellpositionen_sortiment(df, df_bestellungen, df_neue_bestellungen)
df_neue_bestellpositionen.to_parquet(f'data/processed/synth_data_{jetzt:%Y_%m_%d_%H_%M}.parquet')


# Markt Kennzahlen Überprüfung zwischen den realen und synthetischen Märkten & visueller Darstellung der Verteilungen
Abgleich_maerkte = vergleich_real_synth_maerkte(df_maerkte, df_neue_maerkte)
Abgleich_maerkte.to_excel(f'reports/summary/markt_kennzahlen_{jetzt:%Y_%m_%d_%H_%M}.xlsx')
vergleich_real_synth_maerkte_visual(df_maerkte, df_neue_maerkte)


# Bestellungen Kennzahlen Überprüfung zwischen den realen und synthetischen Bestellungen & visualler Darstellung der Verteilungen
Abgleich_bestellungen = vergleich_real_synth_bestellungen(df_bestellungen, df_neue_bestellungen)
Abgleich_bestellungen.to_excel(f'reports/summary/bestellkopf_kennzahlen_{jetzt:%Y_%m_%d_%H_%M}.xlsx')
vergleich_real_synth_bestellungen_visual(df_bestellungen, df_neue_bestellungen)


# Bestellpositionen Kennzahlen Überprüfung zwischen den realen und synthetischen Bestellpositionen & visualler Darstellung der Verteilungen
Abgleich_bestellpositinen = vergleich_real_synth_bestellpositionen(df, df_neue_bestellpositionen)
Abgleich_bestellpositinen.to_excel(f'reports/summary/bestellpositionen_kennzahlen_{jetzt:%Y_%m_%d_%H_%M}.xlsx')


# ABC-Analyse Vergleich zwischen real und real + synth orderlines & Analyse-Plot
Abgleich_abc_klasse = abc_analyse(df, df_neue_bestellpositionen)
abc_analyse_plot(Abgleich_abc_klasse)


# Zusammenfassung der ABC-Klasse der orig&synth Daten (Analysezwecke)
abc = abc_zusammenfassung_neu(df, df_neue_bestellpositionen)
abc.to_excel('data/processed/abc.xlsx')

print('Durchlauf ist fertig')