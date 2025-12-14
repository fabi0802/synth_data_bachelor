import pandas as pd

def combine_orderlines_order(synthetic_orders: pd.DataFrame, synth_orderlines: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt den generierten Orderlines die passende Order-ID der synthetischen Orders hinzu,
    basierend auf der Cluster-Zugehörigkeit und Anzahl gewünschter Orderlines pro Order.
    """

    # Liste zum Speichern Orderlines
    assigned_orderlines = []

    # Index für fortlaufende Orderline-Zuweisung 
    current_idx = {cluster: 0 for cluster in synthetic_orders['cluster'].unique()}

    # Wir sortieren die Orderlines nach Cluster für effizienteren Zugriff
    synth_orderlines = synth_orderlines.sort_values(by='cluster').reset_index(drop=True)

    for i, row in synthetic_orders.iterrows():
        cluster = row['cluster']
        n_lines = row['orderlines']
        order_id = f'synth_{i}'  # Neue eindeutige Order-ID für synthetische Orders

        # Alle Orderlines aus diesem Cluster
        cluster_lines = synth_orderlines[synth_orderlines['cluster'] == cluster]

        start_idx = current_idx[cluster]
        end_idx = start_idx + n_lines

        # Sicherstellen, dass wir nicht über das Ende hinausgehen
        if end_idx > len(cluster_lines):
            end_idx = len(cluster_lines)

        selected_lines = cluster_lines.iloc[start_idx:end_idx].copy()
        selected_lines['order_id'] = order_id
        selected_lines['Datum'] = row['Datum']
        selected_lines['Marktnummer'] = row['Marktnummer']

        # Speichern und Index aktualisieren
        assigned_orderlines.append(selected_lines)
        current_idx[cluster] = end_idx

    # Zusammenführen
    if assigned_orderlines:
        final_df = pd.concat(assigned_orderlines, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return final_df

