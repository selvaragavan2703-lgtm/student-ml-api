import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
df = pd.read_csv("student.csv")
print(df.head())
print(df.shape)
print(df.info())
X = df.drop(["Result","Student_ID"],axis=1)
y = df["Result"]
y = y.map({"Fail" : 0 , "Pass" : 1})

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42,stratify=y)
print("Train :",X_train.shape)
print("Test :",X_test.shape)

model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("y_pred :",y_pred)
print("y_test :",y_test.values)
accuracy = accuracy_score(y_test,y_pred)
print("accuracy :",accuracy)
cm = confusion_matrix(y_test,y_pred)
print("conf matrix : ",cm)
pre = precision_score(y_test,y_pred)
print("precision :",pre)
re_cll = recall_score(y_test,y_pred)
print("Recall score",re_cll)
f_score = f1_score(y_test,y_pred)
print("f1 score :",f_score)

score = cross_val_score(model,X ,y ,cv=5,scoring='accuracy')
print("CV score :",score)
print("mean score :",score.mean())