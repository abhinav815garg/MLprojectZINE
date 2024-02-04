import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Dataset.csv')
# df = df.iloc[:500000]

df['PKT_TYPE'] = df['PKT_TYPE'].astype('category').cat.codes
df['FLAGS'] = df['FLAGS'].astype('category').cat.codes
df['NODE_NAME_FROM'] = df['NODE_NAME_FROM'].astype('category').cat.codes
df['NODE_NAME_TO'] = df['NODE_NAME_TO'].astype('category').cat.codes

for i in range(len(df)):
  if df.loc[i, "PKT_CLASS"] == "b'Normal'":
    df.loc[i, "PKT_CLASS"] = 0 #Not DDoS
  else:
    df.loc[i, "PKT_CLASS"] = 1 #DDoS

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = X[:len(df)]
y = y[:len(df)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_tesst)

y_train = y_train.astype(str)

y_test = y_test.astype(str)

model1 = SVC(kernel='sigmoid', gamma='auto')
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1) * 100
print("SVC Accuracy:", accuracy1, "%")
conf_mat1 = confusion_matrix(y_test, y_pred1)
print("\nSVC Confusion Matrix")
print(conf_mat1)
print("\nSVC Classification Report")
print(classification_report(y_test, y_pred1, zero_division=0))
# cm1 = pd.crosstab(y_test, y_pred1, rownames=['Actual'], colnames=['Predicted'], margins = True)
# sn.heatmap(cm1, annot=True)
# plt.show()

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2) * 100
print("KNN Accuracy:", accuracy2, "%")
conf_mat2 = confusion_matrix(y_test, y_pred2)
print("\nKNN Confusion Matrix")
print(conf_mat2)
print("\nKNN Classification Report")
print(classification_report(y_test, y_pred2, zero_division=0))
# cm2 = pd.crosstab(y_test, y_pred2, rownames=['Actual'], colnames=['Predicted'], margins = True)
# sn.heatmap(cm2, annot=True)
# plt.show()

model3 = GaussianNB()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred3) * 100
print("GaussianNB Accuracy:", accuracy3, "%")
conf_mat3= confusion_matrix(y_test, y_pred3)
print("\nGaussianNB Confusion Matrix")
print(conf_mat3)
print("\nGaussianNB Classification Report")
print(classification_report(y_test, y_pred3, zero_division=0))
# cm3 = pd.crosstab(y_test, y_pred3, rownames=['Actual'], colnames=['Predicted'], margins = True)
# sn.heatmap(cm3, annot=True)
# plt.show()

model4 = RandomForestClassifier(n_estimators=200)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred4) * 100
print("RandomForestClassifier Accuracy:", accuracy_rf, "%")
conf_mat4= confusion_matrix(y_test, y_pred4)
print("\nRandomForestClassifier Confusion Matrix")
print(conf_mat4)
print("\nGaussianNB Classification Report")
print(classification_report(y_test, y_pred4, zero_division=0))
# cm4 = pd.crosstab(y_test, y_pred4, rownames=['Actual'], colnames=['Predicted'], margins = True)
# sn.heatmap(cm4, annot=True)
# plt.show()

model5 = DecisionTreeClassifier()
model5.fit(X_train, y_train)
y_pred5 = model5.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred5) * 100
print("DecisionTreeClassifier Accuracy:", accuracy_rf, "%")
conf_mat5= confusion_matrix(y_test, y_pred5)
print("\nDecisionTreeClassifier Confusion Matrix")
print(conf_mat5)
print("\nDecisionTreeClassifier Classification Report")
print(classification_report(y_test, y_pred5, zero_division=0))
# cm5 = pd.crosstab(y_test, y_pred5, rownames=['Actual'], colnames=['Predicted'], margins = True)
# sn.heatmap(cm5, annot=True)
# plt.show()

model6 = LogisticRegression()
model6.fit(X_train, y_train)
y_pred6 = model6.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred5) * 100
print("LogisticRegression Accuracy:", accuracy_rf, "%")
conf_mat6 = confusion_matrix(y_test, y_pred6)
print("\nLogisticRegression Confusion Matrix")
print(conf_mat6)
print("\nLogisticRegression Classification Report")
print(classification_report(y_test, y_pred6, zero_division=0))
# cm6 = pd.crosstab(y_test, y_pred6, rownames=['Actual'], colnames=['Predicted'], margins = True)
# sn.heatmap(cm6, annot=True)
# plt.show()