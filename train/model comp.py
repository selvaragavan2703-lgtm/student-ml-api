from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib
import pickle
df = pd.read_csv("student.csv")
X = df.drop(["Result","Student_ID"],axis=1)
y = df["Result"]
y = y.map({"Fail" : 0 , "Pass" : 1})

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42,stratify=y)
print("Train :",X_train.shape)
print("Test :",X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN" : KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
}
print(models)

for name,model in models.items():
    model.fit(X_train , y_train)
    print("Trained",name)

for name,model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:",accuracy)

for name,model in models.items():
    score = cross_val_score(model, X_train, y_train, cv=5)
    print("Cross Validation:",score)
    print("Accuracy:",score.mean())

rf = RandomForestClassifier(random_state=42)
parm_grid = {

    "n_estimators":[50,100,200],
    "max_depth":[None,3,5,10],
}

grid = GridSearchCV(estimator=rf, param_grid=parm_grid, cv=5, scoring="accuracy")

grid.fit(X_train,y_train)
print("Best params:",grid.best_params_)
print("Best score:",grid.best_score_)

final_model = grid.best_estimator_
final_model.fit(X_train,y_train)
print("Final Model Trained")

y_pred = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:",final_accuracy)

joblib.dump(final_model,"Student_model.pkl")
print("Successful")

with open("Student_model.pkl", "wb") as file:
    pickle.dump(model, file)