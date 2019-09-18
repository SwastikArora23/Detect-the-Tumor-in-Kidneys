### Importing modules
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import warnings
warnings.filterwarnings('always')
import sklearn

df = pd.read_csv('kidney_disease.csv')
df.head()
print(df.head())

df.columns
print(df.columns)

df.info()
print(df.info())

df.describe
print(df.describe)

df.head()
print(df.head())

df.info()

df.describe

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.grid()
plt.title("Number of Missing Values")
plt.savefig('missing.png')
plt.show()


df.describe(include='all')
print(df.head())

df.rc.head()
print(df.rc.head())

df.wc.head()
print(df.wc.head())

df.pcv.head()
print(df.pcv.head())


for i in ['rc','wc','pcv']:
    df[i] = df[i].str.extract('(\d+)').astype(float)

df.pcv.head()
print(df.pcv.head())

for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:
    df[i].fillna(df[i].mean(),inplace=True) # mean value calculation

df.info()
print(df.info())

df['rbc'].unique()
print(df['rbc'].unique())
sns.countplot(data=df,x='rbc')
plt.show()
df['rbc'].fillna('normal',inplace=True)

df['pc'].unique()
print(df['pc'].unique())
sns.countplot(data=df,x='pc')
plt.show()
df['pc'].fillna('normal',inplace=True)

df['pcc'].unique()
print(df['pcc'].unique())
sns.countplot(data=df,x='pcc')
plt.show()
df['pcc'].fillna('notpresent',inplace=True)

df['ba'].unique()
print(df['ba'].unique())
sns.countplot(data=df,x='ba')
plt.show()
df['ba'].fillna('notpresent',inplace=True)

df['htn'].unique()
print(df['htn'].unique())
sns.countplot(data=df,x='htn')
plt.show()
df['htn'].fillna('no',inplace=True)

df['dm'].unique()
print(df['dm'].unique())
df['dm'] = df['dm'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'}) # changing \tyes and \tno to yes and no
sns.countplot(data=df,x='dm')
plt.show()
df['dm'].fillna('no',inplace=True)

df['cad'].unique()
print(df['cad'].unique())
df['cad'] = df['cad'].replace(to_replace='\tno',value='no')
sns.countplot(data=df,x='cad')
plt.show()
df['cad'].fillna('no',inplace=True)

df['appet'].unique()
print(df['appet'].unique())
sns.countplot(data=df,x='appet')
plt.show()
df['appet'].fillna('good',inplace=True)

df['pe'].unique()
print(df['pe'].unique())
sns.countplot(data=df,x='pe')
plt.show()
df['pe'].fillna('no',inplace=True)

df['ane'].unique()
print(df['ane'].unique())
sns.countplot(data=df,x='ane')
plt.show()
df['ane'].fillna('no',inplace=True)

sns.countplot(data=df,x='cad')
plt.show()
df['cad'] = df['cad'].replace(to_replace='ckd\t',value='ckd')
df.info()

print(df.isnull().sum())

df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
df.rename(columns={'classification':'class'},inplace=True)

df.drop('id',inplace=True,axis=1)
df['pe'] = df['pe'].replace(to_replace='good',value=0)

df.head()
print(df.head())

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.grid()
plt.title("Number of Missing Values")
plt.savefig('missing.png')
plt.show()

df['dm'].replace('yes',1,inplace=True)
df['dm'].replace('no',0,inplace=True)

df.info()

d=df[['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']]
print(d.describe())
print(df.head())

X=df.drop('class',axis=1).values
y=df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,random_state=42) # splitting dataset
knn = KNeighborsClassifier(n_neighbors=4)          #randomly taking value of k_neighbors
knn.fit(X_train, y_train)                           
knn.score(X_test,y_test)
print(knn.score(X_test,y_test))

k_range = list(range(2, 25))              #taking values of n_neighbors from 2-24 in order to test the best
knn = KNeighborsClassifier()                
param_grid = dict(n_neighbors=k_range)
clf = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy',iid=True)
clf.fit(X_train, y_train)
KNN=clf.score(X_test,y_test)
print('Best value of k_neighbors is',clf.best_params_)  #displaying the best value of the n_neighbors possible
print('Best accuracy is ',KNN)


clf = GaussianNB(priors=None,var_smoothing=1e-4) ## We kept on changing the values parameters manually and this was the best possible choice
clf.fit(X_train, y_train)
NB=clf.score(X_test,y_test)
print('Best Accuracy,provided by Naive Bayes Classification is ',NB)


X=df.drop('class',axis=1).values
y=df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,random_state=42)
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)
clf_tree.score(X_test,y_test)
print('Decision Trees',clf_tree.score(X_test,y_test))

clf.get_params
tuned_parameters = [{'max_leaf_nodes':[4,5,6,7,8,9,10,11],'max_depth':[2,3,4,5,6,None],
                     'class_weight':[None,{0: 0.33,1:0.67},'balanced'],'random_state':[24,42]}]
clf_tree = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5,scoring='f1',iid=True)
clf_tree.fit(X_train, y_train)
print (clf_tree.best_params_)  # displaying best possible parameters

X=df.drop('class',axis=1).values
y=df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,random_state=42)
clf_tree = tree.DecisionTreeClassifier(class_weight='balanced', max_depth= 3, max_leaf_nodes=7,random_state= 24)
clf_tree.fit(X_train, y_train)
DT=clf_tree.score(X_test,y_test)
print('Best Accuracy by Decision Tree ',DT)


X=df.drop('class',axis=1).values
y=df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,random_state=42) # splitting dataset into 2 
cl = RandomForestClassifier(n_estimators=2,class_weight='balanced',max_depth=4, random_state=42)
cl.fit(X_train,y_train)
preds = cl.predict(X_test)
cl.score(X_test, y_test)
print("Random Forest Classification",cl.score(X_test, y_test))

tuned_parameters = [{'n_estimators':[7,8,9,10,11,12,13,14,15,16],'max_depth':[2,3,4,5,6,None],
                     'class_weight':[None,{0: 0.33,1:0.67},'balanced'],'random_state':[24,42]}]
clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,scoring='f1',iid=True)
clf.fit(X_train, y_train)
print (clf.best_params_)  # displaying best possible parameters
clf_best = clf.best_estimator_
print(clf_best)


feature_names=df.columns.values.tolist()
feature_names=feature_names[:-1]

plt.figure(figsize=(12,3))
importance = clf_best.feature_importances_.tolist()
feature_series = pd.Series(data=importance,index=feature_names)
feature_series.plot.bar()
plt.title('Feature Importance')
plt.show()

list_to_fill = feature_series[feature_series>0]
print(list_to_fill)


Y=df['class'].values
X=df.drop('class',axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33,random_state=24)
cl = RandomForestClassifier(n_estimators=13,class_weight='balanced',max_depth=6,random_state=24)
cl.fit(X_train,y_train)
preds = cl.predict(X_test)
cl.score(X_test, y_test)
print("Tuned Random Forest Classifier ",cl.score(X_test, y_test))

Y=df['class'].values
X=df[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'bgr', 'bu', 'sc', 'sod',
       'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'appet', 'pe'] ].values
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=24)
cl = RandomForestClassifier(n_estimators=13,class_weight='balanced',max_depth=6,random_state=24)
cl.fit(X_train,y_train)
preds = cl.predict(X_test)
RF=cl.score(X_test, y_test)
print(RF)


X=df.drop('class',axis=1).values
y=df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,random_state=42) # splitting dataset
clf_l=LogisticRegression(solver='liblinear')
clf_l.fit(X_train,y_train)
print("Logistic Regression Classification ",clf_l.score(X_test,y_test))
print(clf_l.get_params)

penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(LogisticRegression(), hyperparameters, cv=5,scoring='f1',iid=True)
clf.fit(X_train, y_train)
print (clf.best_params_)


X=df.drop('class',axis=1).values
y=df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,random_state=42) # splitting dataset
clf_l=LogisticRegression(C=3593.8136638046258,penalty='l2',solver='liblinear')
clf_l.fit(X_train,y_train)
LR=clf_l.score(X_test,y_test)
print(LR)


X=df.drop('class',axis=1).values
Y=df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=24)
clf = svm.SVC(gamma='auto')
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
clf.score(X_test, y_test)
print("SVM Classification ",clf.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=24)

clf = svm.SVC(kernel="sigmoid", gamma=0.01 , C=1000 , random_state=24)

clf.fit(X_train,y_train)
preds = clf.predict(X_test)
SVM=clf.score(X_test, y_test)
print(SVM)

label=['KNN','Naive\n Bayes','Decision\n Tree','Random\n forest','SVM','LR']
data=[KNN,NB,DT,RF,SVM,LR]
print (data)

axes = plt.gca()
#axes.set_ylim([0,1])
index = np.arange(len(label))
plt.bar(index, data)
#plt.yticks(np.arange(0,1,0.1))
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.xticks(index, label)
plt.title('')
plt.show()
