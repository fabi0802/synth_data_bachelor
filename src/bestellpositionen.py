import pandas as pd
import numpy as np

def synth_bestellpositionen_sortiment(df: pd.DataFrame, df_bestellungen: pd.DataFrame, synth_bestellungen: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
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
        mengen  = rng.choice(menge_pool[bc],   size=n, replace=True)

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

