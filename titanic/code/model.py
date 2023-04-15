import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error


def check_feature_importance(X, y):
    etr_model = ExtraTreesRegressor()
    etr_model.fit(X, y)
    importance = pd.Series(etr_model.feature_importances_, index=X.columns)
    return importance


def fit_generic_models(models, X, y):
    fit_models = []
    for model in models:
        model = eval(model+'()').fit(X, y)
        fit_models.append(model)
    return fit_models

def cross_validate_models(models, splits, X, y):
    k_fold = KFold(n_splits=splits, shuffle=False)
    model_scores = []
    for model in models:
        scores = cross_val_score(model, X, y, cv=k_fold)
        model_scores.append({model: scores.mean()})
    return model_scores

# def predict_date(date, df, model):
#     # date expected in format YYYY-MM-DD
#     date = [int(i) for i in date.split('-')]
#     input_variables = df[
#         (df.year == date[0]) &
#         (df.month == date[1]) &
#         (df.day == date[2])
#     ]
#     return model.predict(input_variables)