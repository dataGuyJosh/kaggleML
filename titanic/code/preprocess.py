import pandas as pd
from sklearn import preprocessing

def check_cardinality(df):
    categorical_features = [col for col in df.columns if df[col].dtype == 'O']
    unique_categories = {}
    for feature in categorical_features:
        unique_categories[feature] = len(df[feature].unique())
        # print("Cardinality of {}: {}".format(
        #     feature, unique_categories[feature]))

    return unique_categories

def check_nulls(df):
    return df.isnull().sum()

def read_data(path):
    return pd.read_csv(path)

def categorical_feature_handler(df):
    # find categorical features
    categorical_features = df.select_dtypes(include=['O'])
    # fill null values as mode
    categorical_features.fillna(categorical_features.mode())
    # label encode categorical features (One Hot probably better here)
    le = preprocessing.LabelEncoder()
    for feature in categorical_features:
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

def preprocess(df, target):
    # drop high cardinality, low importance features
    df.drop(['Name','Ticket'], axis=1, inplace=True)
    
    # nulls
    df.fillna(df.mean(), inplace=True)

    # categorical features
    categorical_feature_handler(df)

    # numeric features
    
    return df