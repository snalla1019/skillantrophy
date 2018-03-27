import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error
import matplotlib.pylab as plt
from sklearn.cross_validation import cross_val_predict
import scikitplot.plotters as skplt
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, precision_recall_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import itemfreq

#read data
scored_data = pd.read_csv('filename.csv', header=None, sep= )
scored_data.shape
scored_data.columns=['id','prob','label','none']
y_true=scored_data['label']
y_score=scored_data['prob']
np.place(y_true,y_true==-1,0)
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# Plot ROC Curve
roc_auc = auc(fpr, tpr) 
roc_auc
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#Plot PR Curve
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
pr_auc = auc(recall, precision) 
plt.title('Precision-Recall')
plt.plot(recall, precision, 'b')
plt.legend(loc='lower right')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

pr_auc

#checks
np.place(y_score,y_score=0.07012,1)
np.place(y_score,y_score0.07012,0)
recall=recall_score(y_true, y_score) # 0.115
precision=precision_score(y_true, y_score) #0.099
itemfreq(y_score)
array([[ 0.00000000e+00, 1.38226700e+06],
[ 1.00000000e+00, 5.60000000e+01]])

#F1 score
f1_score(y_true, y_score) 

#accuracy
accuracy_score(y_true, y_score) 

sum(y_score)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(1)
fpr[i], tpr[i], _ = roc_curve(y_test[, i], y_score[, i])
roc_auc[i] = auc(fpr[i], tpr[i])
metrics.roc_auc_score(y_true,y_score) 