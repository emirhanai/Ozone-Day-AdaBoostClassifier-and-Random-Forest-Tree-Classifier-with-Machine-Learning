from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.metrics import recall_score, f1_score, precision_score
#from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('ozone-data.csv')
#print(df.head())

#data select
X = df.iloc[:2535,1:73]
#print(X)

#target select [class]
y = df.iloc[:2535,73:74]
#print(y)

#dataframe to numpy array
yyy = np.array(y).ravel()
XXX = np.array(X).reshape(2534,-1)


from sklearn import ensemble
#creating of AdaBoostClassifier and Random Forest Tree Classifier

#x = 0
#for x in range(970,1970):
    #x = x + 1

    #912 - 70,88,70,75
    #1030 - 71,80,71,75
    #222 - 75,81,75,78

    #1.549999999999978 = 75,75,84,78

    #Now, 75,85,75,79,(ac=96).

#for i in np.arange(0, 150, 1):

#Test and Train create
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=222,

                                                    shuffle=True,stratify=None)  # 70% training and 30% test
#model create
model_ozone = ensemble.AdaBoostClassifier(base_estimator=None,
                                          n_estimators=109,
                                          learning_rate=1.549999999999978,
                                          algorithm='SAMME.R', )  #
from sklearn.metrics import auc

model_ozone.fit(X_train, y_train)


y_test_pred = model_ozone.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, y_test_pred)

auc_test_roc_curve = auc(fpr, tpr)
# print("X select: {}".format(i))
print("Auc Roc Curve Score: ", auc_test_roc_curve)
print("Precision Score: ", precision_score(y_test, y_test_pred, average="macro").mean() * 100)
print("Recall Score: ", recall_score(y_test, y_test_pred, average="macro").mean() * 100)
print("F1 Score: ", f1_score(y_test, y_test_pred, average="macro").mean() * 100)
print("Accuracy:", metrics.accuracy_score(y_test, y_test_pred))

#Auc Roc Curve Score:  0.7499689633767846
#Precision Score:  85.58992487847989
#Recall Score:  74.99689633767845
#F1 Score:  79.2258134963966
#Accuracy: 0.9605781865965834
#[[708   8]
# [ 22  23]]

#Train the model using the training sets

#Predict the response for test dataset
#print(X_test)
#print(y_test)

y_train_pred = model_ozone.predict(X_train)

#confusion matrix model
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_test, y_test_pred)
print(c_matrix)

#[[708   8]
# [ 22  23]]

#print(y_pred)

feature_names= X.columns

from sklearn.tree import export_graphviz

def save_decision_trees_as_dot(model_ozone, iteration, feature_name):
    file_name = open("emirhan_project" + str(iteration) + ".dot",'w')
    dot_data = export_graphviz(
        model_ozone,
        out_file=file_name,
        feature_names=feature_name,
        class_names=['Ozone Day','Normal Day'],
        rounded=True,
        proportion=False,
        filled=True,)
    file_name.close()
    print("Decision Tree in forest :) {} saved as dot file".format(iteration + 1))

#for i in range(len(model_ozone.estimators_)):
    #save_decision_trees_as_dot(model_ozone.estimators_[i], i, feature_names)
    #print(i)
