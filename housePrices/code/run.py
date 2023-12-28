import pandas as pd
import numpy as np

# load data
trn_df = pd.read_csv('data/train.csv')
tst_df = pd.read_csv('data/test.csv')

# summary stats for numerical features
num_stats = trn_df.select_dtypes(include=[np.number]).describe().T
# summary stats for categorical features
cat_stats = trn_df.select_dtypes(include=[object]).describe().T
# print(num_stats, cat_stats, sep='\n')

# null values in dataset
nullsPerFeature = trn_df.isnull().sum()
print(nullsPerFeature / len(trn_df))

# explore rows with null (missing) values
nullRows = trn_df[trn_df.isnull().any(axis=1)]
print(nullRows)

# ~~~ Explore the dependent varaible ~~~
'''
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

# Fit a normal distribution to the SalePrice data
mu, sigma = stats.norm.fit(trn_df['SalePrice'])

# Create a histogram of the SalePrice column
hist_data = go.Histogram(x=trn_df['SalePrice'], nbinsx=50, name="Histogram", opacity=0.75, histnorm='probability density')

# Calculate the normal distribution based on the fitted parameters
x_norm = np.linspace(trn_df['SalePrice'].min(), trn_df['SalePrice'].max(), 100)
y_norm = stats.norm.pdf(x_norm, mu, sigma)

# Create the normal distribution overlay
norm_data = go.Scatter(x=x_norm, y=y_norm, mode="lines", name=f"Normal dist. (μ={mu:.2f}, σ={sigma:.2f})", line=dict(color="green"))

# Combine the histogram and the overlay
fig = go.Figure(data=[hist_data, norm_data])

# Set the layout for the plot
fig.update_layout(
    title="SalePrice Distribution",
    xaxis_title="SalePrice",
    yaxis_title="Density",
    legend_title_text="Fitted Normal Distribution"
)

# Create a Q-Q plot
qq_data = stats.probplot(trn_df['SalePrice'], dist="norm")
qq_fig = px.scatter(x=qq_data[0][0], y=qq_data[0][1], labels={'x': 'Theoretical Quantiles', 'y': 'Ordered Values'})
qq_fig.update_layout(
    title="Q-Q plot"
)

# Calculate the line of best fit
slope, intercept, r_value, p_value, std_err = stats.linregress(qq_data[0][0], qq_data[0][1])
line_x = np.array(qq_data[0][0])
line_y = intercept + slope * line_x

# Add the line of best fit to the Q-Q plot
line_data = go.Scatter(x=line_x, y=line_y, mode="lines", name="Normal Line", line=dict(color="green"))

# Update the Q-Q plot with the normal line
qq_fig.add_trace(line_data)

# Show the plots
fig.show()
qq_fig.show()
'''


# ~~~ Creating a Data (cleaning) Pipeline ~~~
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define transformers for numerical and categorical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))
])

# Update categorical and numerical columns
cat_df = trn_df.select_dtypes(include=['object', 'category']).columns
num_df = trn_df.select_dtypes(include=['int64', 'float64']).columns

# Remove target variable from numerical columns
num_df = num_df.drop('SalePrice')

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_df),
        ('cat', categorical_transformer, cat_df)
    ],remainder = 'passthrough')

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)])

# Apply the pipeline to your dataset
X = trn_df.drop('SalePrice', axis=1)
y = np.log(trn_df['SalePrice']) #normalize dependent variable
X_preprocessed = pipeline.fit_transform(X)



# ~~~ Fit & Tune Models ~~~
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Define the hyperparameter grids for each model
param_grids = {
    'LinearRegression': {},
    'RandomForest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 30],
        'min_samples_split': [2, 5, 10],
    },
    'XGBoost': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 6, 10],
    }
}

# 3-fold cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Train and tune the models
grids = {}
for model_name, model in models.items():
    #print(f'Training and tuning {model_name}...')
    grids[model_name] = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grids[model_name].fit(X_train, y_train)
    best_params = grids[model_name].best_params_
    best_score = np.sqrt(-1 * grids[model_name].best_score_)
    
    print(f'Best parameters for {model_name}: {best_params}')
    print(f'Best RMSE for {model_name}: {best_score}\n')