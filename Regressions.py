"""
Linear Regression & Polynomial Regression ..

🧮 Supervised Regression Practices 
(Linear + Polynomial)
(Classification)
(Decisions Trees)

"""
import numpy as np
from numpy import round
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
  
#1
houses = {
    "size":[50, 60, 70, 90, 120, 150, 180, 220],
    "price":[1500, 1800, 2000, 2600, 3100, 3800, 4400, 5200]
}
df = pd.DataFrame(houses)
x=df[['size']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#plt.scatter(x,y)
#plt.show() # almost linear regression 
model= LinearRegression()
model.fit(x_train,y_train)
y_pred= model.predict(x_test)
print(round(model.predict([[100]])))
print('MSE',mean_squared_error(y_test,y_pred))
print("'Pearson's R",r2_score(y_test,y_pred))

print('-----------Seperate------------')
#========================================
#2
students = {
    "hours":[1, 2, 3, 4, 5, 6, 7, 8],
    "score":[25, 35, 45, 60, 72, 85, 92, 96]
}
df = pd.DataFrame(students)
x = df[['hours']].values
y = df['score'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)

model = LinearRegression()
model.fit(x_poly_train, y_train)

new = np.array([[5.5]])
new_poly = poly.transform(new)
y_pred_new = model.predict(new_poly)
print("Predicted Score for 5.5h =", round(y_pred_new[0], 2))

y_pred = model.predict(x_poly_test)
print("R²:", r2_score(y_test, y_pred))

print('-----------Seperate------------')
#=========================================
#3
cars = {
    "age":[1, 2, 3, 4, 5, 6, 8, 10],
    "price":[28000, 26000, 24000, 21000, 18000, 15000, 12000, 9000]
}
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)

model_p = LinearRegression()
model_p.fit(x_poly_train, y_train)

y_pred_poly = model_p.predict(x_poly_test)
print("R² Poly:", r2_score(y_test, y_pred_poly))

new = np.array([[7]])
new_poly = poly.transform(new)
print("Predicted price (7 years):", round(model_p.predict(new_poly)[0], 2))

print('-----------Seperate------------')
#1

houses = {
    "size":[30, 40, 50, 70, 100, 130, 160, 200],
    "price":[900, 1200, 1500, 2100, 3200, 4600, 6000, 8200]
}
df = pd.DataFrame(houses)
print(df.head())
#First 'linear'
plt.scatter(x,y,color='blue')
#plt.show() 
x=df[['size']].values
y=df['price'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Predicted price for (350 size)",round(model.predict([[350]])[0],2))
#plt.plot(x,model.predict(x))
#plt.show()
#Seconed Polynomial
poly=PolynomialFeatures(degree=2,include_bias=False)
x_TR=poly.fit_transform(x_train)
x_TS=poly.transform(x_test)
model_p=LinearRegression()
model_p.fit(x_TR,y_train)
y_pred_p=model_p.predict(x_TS)
new=np.array([[350]])
new_p=poly.transform(new)
print(round(model_p.predict(new_p)[0],2))

#MSE
print('-- MSE score for Linear =>',round(mean_squared_error(y_test,y_pred)))
print('-- MSE score for Poly =>',round(mean_squared_error(y_test,y_pred_p)))

#R2
print('-- R2 score for Linear =>',round(r2_score(y_test,y_pred)))
print('-- R2 score for Poly =>',round(r2_score(y_test,y_pred_p)))

#Plt
x_curve = np.linspace(min(x), max(x), 200).reshape(-1,1)
x_curve_poly = poly.transform(x_curve)
y_curve = model_p.predict(x_curve_poly)

#plt.scatter(x, y, color='black')
#plt.plot(x_curve, y_curve, color='red')
#plt.show()

print('-----------Seperate------------')
#2

students = {
    "hours":[1, 2, 3, 4, 5, 6, 7, 8, 9],
    "score":[25, 30, 40, 55, 70, 82, 90, 94, 96]
}
df = pd.DataFrame(students)
print(df.head())
x=df[['hours']]
y=df['score']
plt.scatter(x,y)
#plt.show()
#First linear
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("-- For Linear Predicted Score",round(model.predict([[12]])[0],2))
#seconed Polynomial 
poly=PolynomialFeatures(degree=3,include_bias=False)
x_TR=poly.fit_transform(x_train)
x_TS=poly.transform(x_test)
model_p=LinearRegression()
model_p.fit(x_TR,y_train)
y_pred_p=model_p.predict(x_TS)
new=np.array([[6.5]])
new_p=poly.transform(new)
print('-- Predicted Score for 6,5 hours =>',round(model_p.predict(new_p)[0],2))


print('-----------Seperate------------')
#3
cars = {
    "age":[1, 2, 3, 4, 5, 6, 8, 10, 12],
    "price":[29000, 27000, 24000, 21000, 17000, 14000, 11000, 8000, 6000]
}
df = pd.DataFrame(cars)
print(df.head())
x=df[['age']].values
y=df['price'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
poly=PolynomialFeatures(degree=2,include_bias=False)
x_TR=poly.fit_transform(x_train)
x_TS=poly.transform(x_test)
p_model=LinearRegression()
p_model.fit(x_TR,y_train)
y_pred_p=p_model.predict(x_TS)
new=np.array([[7]])
new_p=poly.transform(new)
print('-- Predicte price for age = 7 => ',round(p_model.predict(new_p)[0],2))

print('-----------Seperate------------')
#4

ads = {
    "budget":[5, 10, 15, 20, 25, 30, 35, 40],
    "sales":[50, 80, 110, 170, 250, 330, 420, 520]
}
df = pd.DataFrame(ads)
print(df.head())
x=df[['budget']]
y=df['sales']
#plt.scatter(x,y)
#plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
poly=PolynomialFeatures(degree=2,include_bias=False)
x1=poly.fit_transform(x_train)
x2=poly.transform(x_test)
model=LinearRegression()
model.fit(x1,y_train)
yHat=model.predict(x2)
print('-- R2 =>',r2_score(y_test,yHat))
print('-----------Seperate------------')
#5

icecream = {
    "temperature":[10, 15, 20, 25, 30, 35, 40],
    "sales":[100, 180, 300, 420, 520, 580, 610]
}
df = pd.DataFrame(icecream)
print(df.head())

x=df[['temperature']]
y=df['sales']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
poly=PolynomialFeatures(degree=2,include_bias=False)
x1=poly.fit_transform(x_train)
x2=poly.transform(x_test)
model=LinearRegression()
model.fit(x1,y_train)
yHat=model.predict(x2)
new=np.array([[28]])
new_p=poly.transform(new)
print("-- Predicted sales for 28 ْdegrees =>",round(model.predict(new_p)[0],2))
x_curve = np.linspace(min(x.values), max(x.values), 40).reshape(-1,1)
x_curve_poly = poly.transform(x_curve)
y_curve = model.predict(x_curve_poly)

#plt.scatter(x, y, color='black')
#plt.plot(x_curve, y_curve, color='red')
#plt.show()

print('-----------Seperate------------')
#===================================================
#Exam (Regression + Classification )
#===================================================
#1
houses = {
    "size":[40, 55, 70, 90, 120, 150, 180, 220],
    "price":[1200, 1600, 2000, 2600, 3400, 4200, 5000, 6200]
}
df = pd.DataFrame(houses)
x=df[['size']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('-- Predicted price for 100 size =>',round(model.predict([[100]])[0],2))
print("-- MSE=> ",mean_squared_error(y_test,y_pred))
print("-- R2=> ",r2_score(y_test,y_pred))
#plt.scatter(x,y)
#plt.show()
#plt.plot(x,model.predict(x))
#plt.show()
print('-----------Seperate------------')
#2
rent = {
    "area":[40, 60, 80, 100, 120, 150, 180, 200],
    "rooms":[1, 1, 2, 3, 3, 4, 4, 5],
    "rent":[400, 550, 700, 950, 1200, 1500, 1800, 2200]
}
df = pd.DataFrame(rent)
x=df[['area','rooms']].values
y=df['rent'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("-- Predicted rent for 3 rooms and 110 area => ",round(model.predict([[110,3]])[0],2))
print("-- MSE => ",mean_squared_error(y_test,y_pred))
print("-- R2 => ",r2_score(y_test,y_pred))
print('-----------Seperate------------')
#3
salary = {
    "experience":[1, 2, 3, 4, 5, 6, 8, 10],
    "salary":[1800, 2200, 2600, 3200, 4000, 4800, 6000, 7200]
}
df = pd.DataFrame(salary)
x=df[['experience']].values
y=df['salary'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
poly= PolynomialFeatures(degree=2,include_bias=False)
x_TR=poly.fit_transform(x_train)
x_TS=poly.transform(x_test)
model_p=LinearRegression()
model_p.fit(x_TR,y_train)
y_predP=model_p.predict(x_TS)
new=np.array([[7]])
new_p=poly.transform(new)
print("-- Poly Predicted salary for 7 yrs exp.. => ",round(model_p.predict(new_p)[0],2))
print("-- Poly R2 => ",round(r2_score(y_test,y_predP)))
plt.scatter(x,y)
x_curve=np.linspace(min(x),max(x),10).reshape(-1,1)
x_curve_poly=poly.transform(x_curve)
y_curve=model_p.predict(x_curve_poly)
#plt.plot(x_curve,y_curve)
#plt.show()
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("-- Linear Predicted salary for 7 yrs exp.. => ",round(model.predict([[7]])[0],2))
print("-- Linear R2 => ",round(r2_score(y_test,y_pred)))
print('-----------Seperate------------')
#4
from sklearn.linear_model import LogisticRegression
students = {
    "hours":[1,2,3,4,5,6,7,8,9,10],
    "passed":[0,0,0,0,1,1,1,1,1,1]
}
df = pd.DataFrame(students)
x=df[['hours']]
y=df['passed']
#plt.scatter(x,y)
#plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=model.score(x_test,y_test)
print('-- Predicted passed by 4 hours => ',round(model.predict([[4]])[0],2))
print("-- Accuracy => ",accuracy)

print('-----------Seperate------------')
#5
students = {
    "GPA":[2.1,2.4,2.7,3.0,3.2,3.5,3.7,3.9,4.0],
    "activities":[0,0,1,0,1,1,0,1,1],
    "accepted":[0,0,0,1,1,1,1,1,1]
}
df = pd.DataFrame(students)
x=df[['GPA','activities']]
y=df['accepted']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc=model.score(x_test,y_test)
print("-- Accuracy => ",acc)
print('-- Predicted 3.3 GPA and active => ',model.predict([[3.3,1]]))
print('-----------Seperate------------')
#====================================================
#Decision Trees ..
#====================================================

# Decision Tree Classifier Example (Titanic)

data = {
    'Sex': ['female','male','female','male','female','male','female','male'],
    'Age': [25, 30, 19, 45, 33, 22, 40, 36],
    'Pclass': [1, 3, 1, 2, 3, 1, 2, 3],  
    'Survived': [1, 0, 1, 0, 1, 1, 0, 0] 
}
df = pd.DataFrame(data)

df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

X = df[['Sex', 'Age', 'Pclass']]  # features
y = df['Survived']                # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", round(acc, 2))

new_passenger = pd.DataFrame({'Sex':[0], 'Age':[28], 'Pclass':[3]})
print("Prediction for male age 28 class 3:", tree.predict(new_passenger))

#plt.figure(figsize=(15,6))
#plot_tree(tree, feature_names=['Sex', 'Age', 'Pclass'], class_names=['Dead','Survived'], filled=True)
#plt.show()
print('-----------Seperate------------')
#1
students = {
    "StudyHours": [1,2,3,4,5,6,7,8,9,10],
    "SleepHours": [8,8,7,6,6,5,5,4,3,3],
    "Passed": [0,0,0,0,1,1,1,1,1,1]
}
df = pd.DataFrame(students)
print(df)
x=df[['StudyHours','SleepHours']]
y=df['Passed']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(max_depth=2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(model.predict([[3,7]]))
#plt.figure(figsize=(10,8))
#plot_tree(model,feature_names=['Study Hours','Sleep Hours'],class_names=['Successed','Failed'],filled=True)
#plt.show()
print('-----------Seperate------------')
#2
loan = {
    "Income": [25,40,35,50,60,80,20,70,90,100],
    "CreditScore": [400,600,550,650,700,800,300,750,820,900],
    "Approved": [0,0,0,1,1,1,0,1,1,1]
}
df = pd.DataFrame(loan)
print(df)
x=df[['Income','CreditScore']]
y=df['Approved']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
m=DecisionTreeClassifier(max_depth=2,random_state=42)
m.fit(x_train,y_train)
y_pred=m.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print("-- Accuracy => ",acc)
print("-- Predicted Approved for 150 inc ,1500 cred => ",m.predict([[150,1500]]))
print('-----------Seperate------------')
#3
weather = {
    "Outlook": [0,0,1,2,2,2,1,0,0,2,0,1,1,2],
    "Humidity": [85,80,78,90,70,95,85,75,72,80,65,70,90,75],
    "Windy": [0,1,0,0,0,1,1,0,1,0,1,0,1,0],
    "Play": [0,0,1,1,1,0,1,1,1,1,1,1,0,1]
}
df = pd.DataFrame(weather)
x=df[['Outlook','Humidity','Windy']].values
y=df['Play'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
m=DecisionTreeClassifier(max_depth=2,random_state=42)
m.fit(x_train,y_train)
y_pred=m.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print("-- Accuracy => ",acc)
print("-- Predicted Play for 2,100,1 => ",m.predict([[2,100,1]]))
#plt.figure(figsize=(9,6))
#plot_tree(m,feature_names=['Outlook','Humidity','Windy'],class_names=['In Game','Out Game'],filled=True)
#plt.show()
#حيرانة لية واشمعنا بنستخدم فيجر وبلوت ترى لية مش سكاتر !!
#زائد انا بحتار فحكاية المستويات دى يعنى المستويات اعملها على اد الفيتشرز !!

print('-----------Seperate------------')
#4
data = {
    "Hours_Study": [1,2,3,4,5,6,7,8,9,10],
    "Sleep_Hours": [8,8,7,6,6,5,5,4,3,3],
    "Passed": [0,0,0,0,1,1,1,1,1,1]
}
df = pd.DataFrame(data)

x = df[['Hours_Study','Sleep_Hours']]
y = df['Passed']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=2,random_state=42)   # ✳️ إكملي هنا الباراميتر المناسب لتفادي الـ overfitting
tree.fit(x_train, y_train)

print("Predicted result for 3h study + 6h sleep =>", tree.predict([[3,6]]))
