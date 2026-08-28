from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
x = [
    [25,30000],
    [28,35000],
    [30,40000],
    [35,50000],
    [40,60000],
    [45,70000],
    [50,80000],
    [55,90000],
    [60,100000],
    [65,110000]
]
y = [0,0,0,0,1,1,1,1,1,1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42,stratify=y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = SVC(kernel='linear',C=1)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy :",accuracy)
print("Actual :",y_test)
print("Predicted :",y_pred)