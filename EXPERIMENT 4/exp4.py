import pandas as pd
import autosklearn.classification
from sklearn.model_selection import train_test_split
import csv
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

cls = autosklearn.classification.AutoSklearnClassifier()

# formatting the dataset 
# 1 - AWS 
q=[]
with open('aws_score.csv') as f:
     read = csv.reader(f)
     for rows in read:
         q.append(rows)

w=[]
for i in range(0,98):
    e=q[1:100][i][1]
    w.append(e)

w = pd.DataFrame(w,columns=['aws'])

# 2 - azure 
a=[]
with open('azure_score.csv') as f:
     read = csv.reader(f)
     for rows in read:
         a.append(rows)

s=[]
for i in range(0,98):
    e=a[1:100][i][1]
    s.append(e)

s = pd.DataFrame(s,columns=['azure'])

# 3 - Google 
k=[]
with open('google_score.csv') as f:
     read = csv.reader(f)
     for rows in read:
         k.append(rows)

g=[]
for i in range(0,98):
    e=k[1:100][i][1]
    g.append(e)

g = pd.DataFrame(g,columns=['google'])

# 4 - watson
h=[]
with open('watson_score.csv') as f:
     read = csv.reader(f)
     for rows in read:
         h.append(rows)

j=[]
for i in range(0,98):
    e=h[1:100][i][1]
    j.append(e)

j = pd.DataFrame(j,columns=['watson'])

# 5 - manual
r=[]
with open('manual_sentiment.csv') as f:
     read = csv.reader(f)
     for rows in read:
         r.append(rows)

t=[]
for i in range(0,98):
    e=r[1:100][i][1]
    t.append(e)

t = pd.DataFrame(t,columns=['manual'])

df = w.join(s)
df = df.join(g)
df = df.join(j)

#wsgj
lbl= LabelEncoder()
#df = lbl.fit_transform(df)
t = lbl.fit_transform(t)

X_train,X_test,y_train,y_test = train_test_split(df,t,test_size=0.3)
#X_train = numpy.asarray(X_train)
#X_train = X_train.reshape(-1,1)
#y_train = numpy.ravel(y_train)
#y_train = y_train.asfactor()
cls.fit(X_train, y_train)
#X_test = numpy.asarray(X_test)
#X_test = X_test.reshape(-1,1)
predictions = cls.predict(X_test)
cm = confusion_matrix(y_test,predictions)
# cm = 
'''
0    0	3	2
1    1	2	1
2    1	4	16
'''