import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('Demographics_data.csv')
#print(df.head)

X = df[["Race Quant.", "Gender Quant.", "Age", "Preparedness Quant."]]
y = df["Goal_Ach_Quant."]
y2 = df["Rec_Quant."]
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)
#print(X_train)

regr = LogisticRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print(regr.coef_) 
print(metrics.classification_report(y_test, y_pred))
cm = metrics.confusion_matrix
print(cm(y_test, y_pred))

X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.2, random_state = 0)
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

regr = LogisticRegression()
regr.fit(X_train, y2_train)
y2_pred = regr.predict(X_test)
print(regr.coef_) 
print(metrics.classification_report(y2_test, y2_pred))
cm = metrics.confusion_matrix
print(cm(y2_test, y2_pred))

df2 = pd.read_csv("Feedback_Data.csv")

X2 = df2[["Challenge_Quant", "Current_Quant" ,"Relevant_Quant" , "Valuable_Quant" ,"Nav_Quant", "Mat_Quant", "Office_Quant", "Quick_Quant","Effective_Quant","Helpful_Quant","Healthy_Quant","Respect_quant","Knowledge_Quant"]]
y3 = df2["Goals_Quant"]
y4 = df2["Recc_Quant"]
#print(X)

X2_train, X2_test, y3_train, y3_test = train_test_split(X2, y3, test_size = 0.2, random_state = 0)

nb = MultinomialNB()
nb.fit(X2_train, y3_train)
y3_pred = nb.predict(X2_test)
print(nb.coef_) 
print(metrics.classification_report(y3_test, y3_pred))
cm = metrics.confusion_matrix
print(cm(y3_test, y3_pred))

sc_x = StandardScaler()
X2_train = sc_x.fit_transform(X2_train)
X2_test = sc_x.fit_transform(X2_test)
#print(X_train)

regr = LogisticRegression()
regr.fit(X2_train, y3_train)
y3_pred = regr.predict(X2_test)
print(regr.coef_) 
print(metrics.classification_report(y3_test, y3_pred))
cm = metrics.confusion_matrix
print(cm(y3_test, y3_pred))


X2_train, X2_test, y4_train, y4_test = train_test_split(X2, y4, test_size = 0.2, random_state = 0)

nb.fit(X2_train, y4_train)
y4_pred = nb.predict(X2_test)
print(nb.coef_) 
print(metrics.classification_report(y4_test, y4_pred))
cm = metrics.confusion_matrix
print(cm(y4_test, y4_pred))

X2_train = sc_x.fit_transform(X2_train)
X_2test = sc_x.fit_transform(X2_test)

regr = LogisticRegression()
regr.fit(X2_train, y4_train)
y4_pred = regr.predict(X2_test)
print(regr.coef_) 
print(metrics.classification_report(y4_test, y4_pred))
cm = metrics.confusion_matrix
print(cm(y4_test, y4_pred))