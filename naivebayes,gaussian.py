from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
x = [
    [21,25000,165],
    [25,30000,170],
    [30,40000,175],
    [35,50000,180],
    [40,60000,172],
    [45,70000,168]
]
y = [0,0,1,1,1,1]
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.5 , random_state = 42 , stratify=y)
model = GaussianNB()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy :", accuracy)
print("Actual :",y_test)
print("Predicted :",y_pred)
