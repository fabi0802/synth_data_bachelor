import pandas as pd
from src.clustering import sampling, kmean_cluster, n_orders
from src.orderhead import generate_orders
from src.orderline import generate_orderlines
from src.combine import combine_orderlines_order
from src.analytics import orderkopf_auswertung, orderline_auswertung

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