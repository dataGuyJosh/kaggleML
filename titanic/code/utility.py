from sklearn.model_selection import cross_val_score, KFold

def cross_validate_model(model, splits, X, y):
    k_fold = KFold(n_splits=splits, shuffle=False)
    model_scores = []
    scores = cross_val_score(model, X, y, cv=k_fold)
    model_scores.append({model: scores.mean()})
    return model_scores