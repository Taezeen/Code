import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,GridSearchCV
from imblearn.over_sampling import SMOTE
import lux
from sklearn.ensemble import StackingClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import imblearn
from sklearn import metrics
parkinsons_data=pd.read_csv('C:/Users/user/Downloads/parkinsons.data')
parkinsons_data.head(n=10)
parkinsons_data.shape
parkinsons_data.info
parkinsons_data.describe
parkinsons_data.isnull().sum()
for i in parkinsons_data.columns:
    print("***********************************",i,"**********************************")
    print()
    print(set(parkinsons_data[i].tolist()))
    print()
import matplotlib.pyplot as plt
import seaborn as sns
p=parkinsons_data["status"].value_counts()
p_parkinsons_data=pd.DataFrame({'status':p.index,'values':p.values})
print(sns.barplot(x='status', y='values', data=p_parkinsons_data))
sns.pairplot(parkinsons_data)
def distplots(col):
    sns.distplot(parkinsons_data[col])
    plt.show()
for i in list(parkinsons_data.columns)[1:]:
    distplots(i)
def boxplots(col):
    sns.boxplot(parkinsons_data[col])
    plt.show()
for i in list(parkinsons_data.select_dtypes(exclude=["object"]).columns)[1:]:
    boxplots(i)    
plt.figure(figsize=(10,10))
corr=parkinsons_data.corr()
sns.heatmap(corr,annot=True)
X= parkinsons_data.drop(columns=[ 'name','status'], axis=1)
y=parkinsons_data['status']
#SAMPLING 
from collections import Counter
print(Counter(y))
sm=SMOTE(random_state=2)
x,y=sm.fit_resample(X, y)
print(Counter(y)) 
#SCALING
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(x) 
x     
import sklearn_relief as relief
r = relief.Relief(
    n_features=6) 
x = r.fit_transform(x,y)
print(x)     
print("--------------")
print("(No. of tuples, No. of Columns before ReliefF) : "+str(X.shape)+
      "\n(No. of tuples , No. of Columns after ReliefF) : "+str(x.shape))   
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
param_grid=[{
    'n_estimators': [100,200,300,400],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9],
    'criterion' :['gini', 'entropy','log_loss']
    }]
grid_search=GridSearchCV(clf,param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=10)
grid_search.fit(x_train,y_train)
final_clf=grid_search.best_estimator_
final_clf
final_clf.score(x_test, y_test)
from sklearn.linear_model import LogisticRegression
clf2=LogisticRegression()
param_grid=[{
    'C':[0.2,0.3,0.4,0.5,0.6],
    'max_iter':[30,40,50,60,70],
    'solver':['saga','sag','lbfgs','newton-cg'],
    'penalty':['l1','l2','elasticnet','none'],
    'l1_ratio':[0.2,0.3,0.4]
    }]
grid_search=GridSearchCV(clf2,param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=10)
grid_search.fit(x_train,y_train)
final_clf2=grid_search.best_estimator_
final_clf2
final_clf2.score(x_test, y_test)
from xgboost import XGBClassifier
clf3=XGBClassifier()
param_grid=[{
    'booster':['gbtree','gblinear','dart'],
    'eta':[0.1,0.2,0.3,0.4,0.5],
    'max_depth':[4,5,6,7],
    'subsample':[0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'min_child_weight':[0.6,0.7,0.8,0.9,1]
    }]
grid_search=GridSearchCV(clf3,param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=10)
grid_search.fit(x_train,y_train)
final_clf3=grid_search.best_estimator_
final_clf3
final_clf3.score(x_test, y_test)
from sklearn.neighbors import KNeighborsClassifier
clf4=KNeighborsClassifier()
param_grid=[{
    'n_neighbors':[2,3,4,5,6,7],
    'p':[1,2,3],
    'weights':['uniform','distance'],
    'leaf_size':[4,5,6,7,8],  
    }]
grid_search=GridSearchCV(clf4,param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=10)
grid_search.fit(x_train,y_train)
final_clf4=grid_search.best_estimator_
final_clf4
final_clf4.score(x_train, y_train)
from sklearn.svm import SVC
clf5=SVC()
param_grid=[{
    'cache_size':[100,200,300,400],
    'C':[1,2,3,4,5],
    'kernel':['rbf','linear','sigmoid']
    }]
grid_search=GridSearchCV(clf5,param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=10)
grid_search.fit(x_train,y_train)
final_clf5=grid_search.best_estimator_
final_clf5
final_clf5.score(x_train, y_train)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
estimators=[
    ('rfi',RandomForestClassifier(criterion='gini', random_state=22,max_depth=8, n_estimators=200)),
    ('xgb',XGBClassifier(random_state=22,base_score=0.4,booster='gbtree', eta=0.3, max_depth=7, subsample=0.7, grow_policy='depthwise',max_bin=256,min_child_weight=0.6,n_estimators=200)),
    ('knn',KNeighborsClassifier(n_neighbors=6,p=2,leaf_size=8,weights='distance'))]
sc=StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(C=0.9, l1_ratio=0.7,max_iter=300, penalty='elasticnet',solver='saga', random_state=12))
stackclf=model_selection.cross_val_score(sc,x,y,cv=5, scoring='accuracy')
np.mean(stackclf)*100

