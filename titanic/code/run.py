import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utility import cross_validate_model

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(train_df.head(),test_df.head(),sep='\n~~~\n')

features = ['Pclass', 'Sex', 'SibSp', 'Parch']

X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])
y = train_df.Survived

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,y)
predictions = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': model.predict(X_test)})

print(cross_validate_model(model, 10, X, y))

# predictions.to_csv('data/submission.csv', index = False)


# X = train_df.drop(['Survived'], axis=1)
# y = train_df['Survived']