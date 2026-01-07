import pandas as pd
import numpy as np

def synth_bestellpositionen(df: pd.DataFrame, df_bestellungen: pd.DataFrame, synth_bestellungen: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
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

    orig_df = df.copy()
    orig_df['order_id'] = orig_df.groupby(['Marktnummer', 'Datum']).ngroup()
    orig_df['Sortiment'] = orig_df['Artikelnummer'].astype(str).str[2:4]

    orig_df = pd.merge(
        orig_df,
        df_bestellungen[['order_id', 'cluster']],
        on='order_id',
        how='left'
    ).rename(columns={'cluster': 'bestell_cluster'}).reset_index(drop=True)

    prob_cl_sort = (
        orig_df.groupby('bestell_cluster')['Sortiment']
        .value_counts(normalize=True)
        .rename('prob')
        .reset_index()
    )

    prob_cl_sort_artk = (
        orig_df.groupby(['bestell_cluster', 'Sortiment'])['Artikelnummer']
        .value_counts(normalize=True)
        .rename('prob')
        .reset_index()
    )

    prob_cl_sort_n = (
        orig_df.groupby(['bestell_cluster', 'Sortiment'])['MengeInKolli']
        .value_counts(normalize=True)
        .rename('prob')
        .reset_index()
    )

    # ---- Lookups bauen (einmalig) ----
    sort_dist = {}
    for bc, g in prob_cl_sort.groupby('bestell_cluster', sort=False):
        sort_dist[int(bc)] = (g['Sortiment'].to_numpy(), g['prob'].to_numpy(dtype=float))

    art_dist = {}
    for (bc, s), g in prob_cl_sort_artk.groupby(['bestell_cluster', 'Sortiment'], sort=False):
        art_dist[(int(bc), s)] = (g['Artikelnummer'].to_numpy(), g['prob'].to_numpy(dtype=float))

    qty_dist = {}
    for (bc, s), g in prob_cl_sort_n.groupby(['bestell_cluster', 'Sortiment'], sort=False):
        qty_dist[(int(bc), s)] = (g['MengeInKolli'].to_numpy(), g['prob'].to_numpy(dtype=float))

    synth_bestellungen = synth_bestellungen.copy()
    synth_bestellungen['Datum'] = pd.to_datetime(synth_bestellungen['Datum'])

    rng = np.random.default_rng(seed)

    out_order_id = []
    out_markt = []
    out_datum = []
    out_bc = []
    out_sort = []
    out_art = []
    out_qty = []

    for order in synth_bestellungen.itertuples(index=False):
        oid = order.order_id
        markt = order.Marktnummer
        datum = order.Datum
        bc = int(order.cluster_bestellungen)
        n = int(order.orderlines)

        sortimente, p_sort = sort_dist[bc]
        counts = rng.multinomial(n, p_sort)

        for i in np.nonzero(counts)[0]:
            s = sortimente[i]
            k_s = int(counts[i])

            art_vals, art_p = art_dist[(bc, s)]
            qty_vals, qty_p = qty_dist[(bc, s)]

            artikel = rng.choice(art_vals, size=k_s, replace=True, p=art_p)
            mengen  = rng.choice(qty_vals, size=k_s, replace=True, p=qty_p)

            out_order_id.extend([oid] * k_s)
            out_markt.extend([markt] * k_s)
            out_datum.extend([datum] * k_s)
            out_bc.extend([bc] * k_s)
            out_sort.extend([s] * k_s)
            out_art.extend(artikel.tolist())
            out_qty.extend(mengen.tolist())

    return pd.DataFrame({
        'order_id': out_order_id,
        'Marktnummer': out_markt,
        'Datum': out_datum,
        'cluster_bestellungen': out_bc,
        'Sortiment': out_sort,
        'Artikelnummer': out_art,
        'MengeInKolli': out_qty
    })


