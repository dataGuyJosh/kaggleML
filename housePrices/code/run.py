import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from IPython.display import display, HTML

# load data
trn_df = pd.read_csv('data/train.csv')
tst_df = pd.read_csv('data/test.csv')

# summary stats for numerical features
num_stats = trn_df.select_dtypes(include=[np.number]).describe().T
# summary stats for categorical features
cat_stats = trn_df.select_dtypes(include=[object]).describe().T
print(num_stats, cat_stats, sep='\n')

# null values in dataset
nullsPerFeature = trn_df.isnull().sum()
print(nullsPerFeature / len(trn_df))

# explore rows with null values
nullRows = trn_df[trn_df.isnull().any(axis=1)]
print(nullRows)

# explore the dependent variable (SalePrice)