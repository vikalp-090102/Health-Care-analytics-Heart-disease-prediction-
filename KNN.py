import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings
df=pd.read_csv("heartDataSet.csv")
df.head()
def transform_label(value):
    if value>=1.0:
        return 1
    else:
        return 0

df["target"]=df.target.apply(transform_label)
df.head(10)
df.info()
df.describe()
corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X=df.drop('target',axis=1)
y=df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
X_train = pd.DataFrame(data=X_train)
y_train = pd.DataFrame(data=y_train)
X_train = np.random.permutation(X_train)
y_train = np.random.permutation(y_train)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
X_test
def eucledian(p1,p2):
    dist = np.sqrt(np.sum(p1-p2)**2)
    return dist
def KNN(X_train, y_train , X_test, k):

    op_labels = []
     
    for item in X_test: 
         
        #Array ko declare kara distances between training point and testing point
        point_dist = []
         

        for j in range(len(X_train)):
            
            distances = eucledian(np.array(X_train[j,:]) , item) 

            point_dist.append(distances) 
        point_dist = np.array(point_dist)
        
        dist = np.argsort(point_dist)[:k]
         
        labels = y_train[dist]
         
        #Voting karaayi hai
        lab = mode(labels)
        lab = lab.mode[0]
        op_labels.append(lab)
 
    return op_labels   

y_pred = KNN(X_train, y_train, X_test,4)
score=accuracy_score(y_test, y_pred) * 100
warnings.simplefilter(action='ignore',category=FutureWarning)
X_test.shape
print(score)
new_input=[[2.12309797, -1.41131072,2.12309797,-1.41131072,2.12309797,-1.41131072,2.12309797,-1.41131072,2.12309797, -1.41131072,2.12309797, -1.41131072,2.12309797,2.12309797, -1.41131072,2.12309797,-1.41131072,2.12309797,-1.41131072,2.12309797,-1.41131072,2.12309797, -1.41131072,2.12309797, -1.41131072,2.12309797,-1.41131072,2.12309797]]
new_output = KNN(X_train,y_train,new_input,5)
new_output=np.array(new_output)
flat_array=new_output.flatten()
print(flat_array)
 