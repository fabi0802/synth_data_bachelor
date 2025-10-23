import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE
from clustering_01 import sampling, kmean_cluster, n_orders
from orderhead_02 import generate_orders
from orderline_03 import generate_orderlines
from combine_04 import combine_orderlines_order

df = pd.read_parquet('data/raw/orderlines_dezember.parquet')