import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df=pd.read_csv("Advert.csv")
print(df)
df1=df.drop(columns='Unnamed: 0',axis=1)
print(df1)
print(df1.head)
print(df1.shape)
print(df1.describe())
X=df1.iloc[:,:-1]
print(X)
Y=df1.iloc[:,-1]
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=100)
reg=LinearRegression()
reg.fit(X_train,Y_train)
y_pred = reg.predict(X_test)
print(reg.intercept_)
print(reg.coef_)
coeff_df = pd.DataFrame(reg.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)
predictions =reg.predict(X_test)
print(predictions)
print(r2_score(Y_test,predictions)*100)


# Visualization of the given data:
sns.heatmap(df1.corr(),cmap="pink",annot=True)
plt.show()
sns.pairplot(data=df1,kind='scatter')
plt.show()