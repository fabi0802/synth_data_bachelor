import pandas as pd
import numpy as np

def synth_bestellpositionen(df: pd.DataFrame, df_bestellungen: pd.DataFrame, synth_bestellungen: pd.DataFrame) -> pd.DataFrame:
    """
        Generiert synthetische Bestellpositionen auf Basis realer, geclusterter Bestelldaten.

        Für jedes Bestellcluster werden empirische Pools aus Artikelnummern und
        Bestellmengen gebildet. Die synthetischen Bestellpositionen werden anschließend
        durch zufälliges Ziehen (mit Zurücklegen) aus diesen Pools pro synthetischer
        Bestellung erzeugt.

        Args:
            df (DataFrame): Originale Bestellpositionsdaten 
            df_bestellungen (DataFrame):Cluster Bestelldatensatz aus kmeans_cluster_bestellungen
            synth_bestellungen (DataFrame): Synthetische Bestellungen

        Returns:
            synth_orderlines (DataFrame): Synthetische Bestellpositionen 
    """

    # Erstellung der order_id für cluster merge
    orig_df = df.copy()
    orig_df['order_id'] = orig_df.groupby(['Marktnummer', 'Datum']).ngroup()
    orig_df['Sortiment'] = orig_df['Artikelnummer'].astype(str).str[2:4]

    # Merge der original bestellcluster an die original orderlines
    orig_df = pd.merge(
        orig_df,
        df_bestellungen[['order_id', 'cluster']],
        on = 'order_id',
        how='left'
    ).rename(columns={
        'cluster': 'bestell_cluster'
    }).reset_index(drop=True)

    # Pools pro Bestell-Cluster bauen (mit Häufigkeiten durch Wiederholungen)
    artikel_pool = orig_df.groupby('bestell_cluster')['Artikelnummer'].apply(lambda s: s.to_numpy()).to_dict()
    menge_pool   = orig_df.groupby('bestell_cluster')['MengeInKolli'].apply(lambda s: s.dropna().to_numpy()).to_dict()

    # Synthetische Orderlines generieren
    synth_bestellungen = synth_bestellungen.copy()
    synth_bestellungen['Datum'] = pd.to_datetime(synth_bestellungen['Datum'])

    out_parts = []

    # Iteration über synthetische Bestellungen
    for order in synth_bestellungen.itertuples(index=False):
        oid = order.order_id
        markt = order.Marktnummer
        datum = order.Datum
        bc = int(order.cluster_bestellungen)
        n = int(order.orderlines)

        # Reproduzierbarkeit gewährleisten
        rng = np.random.default_rng(seed=42)

        # Orderline Generierung
        # Zufällige Ziehung (Durch Häufigkeit, werden "beliebte Artikel" automatisch häufiger vorkommen)
        artikel = rng.choice(artikel_pool[bc], size=n, replace=True)
        mengen  = rng.random.choice(menge_pool[bc],   size=n, replace=True)

        # Plausibilisierung
        mengen = np.maximum(1, np.rint(mengen).astype(int))

        part = pd.DataFrame({
            'order_id': oid,
            'Marktnummer': markt,
            'Datum': datum,                 
            'cluster_bestellungen': bc,
            'Artikelnummer': artikel,
            'MengeInKolli': mengen
        })

        out_parts.append(part)

    if not out_parts:
        return pd.DataFrame(columns=['order_id', 'Marktnummer', 'Datum', 'cluster_bestellungen', 'Artikelnummer', 'MengeInKolli'])

    synth_orderlines = pd.concat(out_parts, ignore_index=True)
    return synth_orderlines

