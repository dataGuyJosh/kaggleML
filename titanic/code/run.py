from preprocess import check_cardinality, check_nulls, read_data, preprocess
from model import check_feature_importance, fit_generic_models, cross_validate_models

raw_train_df = read_data('data/train.csv')
train_df = preprocess(raw_train_df.copy(), 'Survived')

# print(raw_train_df,train_df)

X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']

# check nulls, cardinality & feature importance
print(
    check_nulls(raw_train_df),
    check_cardinality(raw_train_df),
    check_feature_importance(X, y),
    sep = '\n~~~\n'
)

# fit models
models = fit_generic_models(['DecisionTreeRegressor', 'ExtraTreesRegressor', 'LinearRegression'], X, y)

# check scores
scores = cross_validate_models(models, 10, X, y)
print(scores)