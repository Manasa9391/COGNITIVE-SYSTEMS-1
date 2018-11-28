import pandas as pd
import autosklearn.classification
from sklearn.model_selection import train_test_split
import csv
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sn 
import matplotlib.pyplot as plt 

cls = autosklearn.classification.AutoSklearnClassifier()

# formatting the dataset 
# google 
google = []
with open('google.csv') as f :
    reader = csv.reader(f)
    for rows in reader:
        google.append(rows)

google = google[1:]
google = pd.DataFrame(google,columns=['id','google score'])

google_ml = google[['google score']]

# aws 
aws = []
with open('aws.csv') as f :
    reader = csv.reader(f)
    for rows in reader:
        aws.append(rows)

aws = aws[1:]
aws = pd.DataFrame(aws,columns=['id','aws_score',])

aws_ml = aws[['aws_score']]

#azure
azure = []
with open('azure.csv') as f :
    reader = csv.reader(f)
    for rows in reader:
        azure.append(rows)

azure = azure[1:]
azure = pd.DataFrame(azure,columns=['id','azure score'])

azure_ml = azure[['azure score']]

#watson
watson = []
with open('watson.csv') as f :
    reader = csv.reader(f)
    for rows in reader:
        watson.append(rows)

watson = watson[1:]
watson = pd.DataFrame(watson,columns=['id','watson_score'])

watson_ml = watson[['watson_score']]

#manual
manual = []
with open('manual.csv') as f :
    reader = csv.reader(f)
    for rows in reader:
        manual.append(rows)

manual = manual[1:]
manual = pd.DataFrame(manual,columns=['id','manual_sentiment'])

manual_ml = manual[['manual_sentiment']]

df = azure_ml.join(aws_ml)
df = df.join(google_ml)
df = df.join(watson_ml)

#wsgj
lbl= LabelEncoder()
#df = lbl.fit_transform(df)
t = lbl.fit_transform(manual_ml)

X_train,X_test,y_train,y_test = train_test_split(df,t,test_size=0.2)
#X_train = numpy.asarray(X_train)
#X_train = X_train.reshape(-1,1)
#y_train = numpy.ravel(y_train)
#y_train = y_train.asfactor()
cls.fit(X_train, y_train)
#X_test = numpy.asarray(X_test)
#X_test = X_test.reshape(-1,1)
predictions = cls.predict(X_test)
cm = confusion_matrix(y_test,predictions)
ax = plt.subplot()
ax.set_xlabel('Predicted Sentiments')
ax.set_ylabel('Manual Sentiments')
ax.set_title('Confusion Matrix')
sn.heatmap(cm, annot=True,annot_kws={"size": 16})# font size
ax.xaxis.set_ticklabels(['Negative','Neutral','Positive'])
ax.yaxis.set_ticklabels(['Negative','Neutral','Positive'])
plt.show()
'''
0 0	 5	9
1 0	19	29
2 0	14	49
'''