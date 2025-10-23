import pandas as pd
from clustering_01 import sampling, kmean_cluster, n_orders
from orderhead_02 import generate_orders
from orderline_03 import generate_orderlines
from combine_04 import combine_orderlines_order
from analytics_10 import orderkopf_auswertung, orderline_auswertung

SAMPLING_NR = 5000
RANDOM_SEED = 42

df = pd.read_parquet('data/raw/orderlines_dezember.parquet')

# Cluster mechanism
df_sample = sampling(df, SAMPLING_NR)
df_sample = kmean_cluster(df_sample)
order_count = n_orders(df_sample)

# Order Generation
synth_orders = generate_orders(df_sample, order_count)

# Order Auswertung
orderkopf_auswertung(df_sample, synth_orders)

# Orderline Generation
synth_orderlines = generate_orderlines(df_sample, synth_orders)

# Order / Orderline Verknüpfung
synth_df = combine_orderlines_order(synth_orders, synth_orderlines)

# Orderline Auswertung
orderline_auswertung(df_sample, synth_orderlines)