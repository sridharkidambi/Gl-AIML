import pandas as pd;
import numpy as np;
from sklearn import svm;
import matplotlib.pyplot as plt;
import string;
from sklearn import metrics;
import seaborn as sns;

def getAccuracy(testResult,PreictResult):
    correct=0;
    for i in range(len(testResult)):
        if(testResult[i]==PreictResult[i]):
            correct=correct+1;
    return (100* (correct/float(len(testResult))));


ldata=pd.read_csv("letterdata.csv");
print(ldata.head(5));

# splitting data
X,y=np.array(ldata)[:,1:16],np.array(ldata.letter)[:];

X_train=X[:16000,:];
X_test=X[16000:,:];
y_train=y[:16000];
y_test=y[16000:];

clf=svm.SVC(gamma=0.025,C=3);

clf.fit(X_train,y_train);
y_pred = clf.predict(X_test);

print(getAccuracy(y_test,y_pred));

y_grid=np.column_stack([y_test,y_pred]);
print(y_grid);
np.savetxt("sk.csv",y_grid,fmt='%s');

lab=list(string.ascii_uppercase[0:26]);
plab=["Pr"+ s for s in lab];

cm =metrics.confusion_matrix(y_test,y_pred,labels=lab);
df_cm=pd.DataFrame(cm,index=[i for i in lab],columns=[i for i in plab]);
plt.figure(figsize=(20,16));
sns.heatmap(df_cm,annot=True,fmt='g');
plt.show();