import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

diabetes=pd.read_csv("./data/diabetes.csv")

X=diabetes.drop(['Outcome'],axis=1).values
y=diabetes['Outcome'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

params_grid={'n_estimators':[10,20,50,100,200],
             'min_samples_leaf':[2,5,10,20],
             'max_depth':[5,10,20]}
clf=RandomForestClassifier()

grid_search=GridSearchCV(clf,params_grid,cv=5,scoring='accuracy',verbose=True)

grid_search.fit(X_train,y_train)
best_params=grid_search.best_params_

print(f'Best params: {best_params}')

best_model=grid_search.best_estimator_

y_pred=best_model.predict(X_test)

print(f"Accuracy TEST: {accuracy_score(y_test, y_pred)}")
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))