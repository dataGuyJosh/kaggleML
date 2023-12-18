import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from utility import cross_validate_model

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(train_df.head())

train_df = train_df.fillna(train_df.mean())

X_train = train_df[['MSSubClass', 'LotFrontage']].to_numpy()
y_train = train_df[['SalePrice']].to_numpy()


model = LinearRegression().fit(X_train,y_train)

train_df['LinearPrediction'] = model.predict(X_train)

# print(train_df)
print(mean_absolute_error(train_df['SalePrice'], train_df['LinearPrediction']))

print(cross_validate_model(model, 10, X_train, y_train))