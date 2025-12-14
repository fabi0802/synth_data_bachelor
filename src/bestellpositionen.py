import pandas as pd
import numpy as np

def synth_bestellpositionen(df: pd.DataFrame, df_bestellungen: pd.DataFrame, synth_bestellungen: pd.DataFrame) -> pd.DataFrame:

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

    # --- 3) Pools pro Bestell-Cluster bauen (mit Häufigkeiten durch Wiederholungen) ---
    artikel_pool = orig_df.groupby('bestell_cluster')['Artikelnummer'].apply(lambda s: s.to_numpy()).to_dict()
    menge_pool   = orig_df.groupby('bestell_cluster')['MengeInKolli'].apply(lambda s: s.dropna().to_numpy()).to_dict()

    # --- 4) Synthetische Orderlines generieren ---
    synth_bestellungen = synth_bestellungen.copy()
    synth_bestellungen['Datum'] = pd.to_datetime(synth_bestellungen['Datum'])

    out_parts = []

    for order in synth_bestellungen.itertuples(index=False):
        oid = order.order_id
        markt = order.Marktnummer
        datum = order.Datum
        bc = int(order.cluster_bestellungen)
        n = int(order.orderlines)

        if n <= 0:
            continue

        # Falls ein Cluster fehlt (sollte selten sein)
        if bc not in artikel_pool or bc not in menge_pool:
            continue

        artikel = np.random.choice(artikel_pool[bc], size=n, replace=True)
        mengen  = np.random.choice(menge_pool[bc],   size=n, replace=True)

        # Plausibilisierung
        mengen = np.maximum(1, np.rint(mengen).astype(int))

        part = pd.DataFrame({
            'order_id': oid,
            'Marktnummer': markt,
            'Datum': datum,                 # datetime64 lassen (spart RAM)
            'cluster_bestellungen': bc,
            'Artikelnummer': artikel,
            'MengeInKolli': mengen
        })

        out_parts.append(part)

    if not out_parts:
        return pd.DataFrame(columns=['order_id', 'Marktnummer', 'Datum', 'cluster_bestellungen', 'Artikelnummer', 'MengeInKolli'])

    synth_orderlines = pd.concat(out_parts, ignore_index=True)
    return synth_orderlines

