# Import train_test_split function
from sklearn.ensemble import RandomForestClassifier
from random import shuffle as array
from sklearn.model_selection import train_test_split
import numpy as np

def classify(x,y,tr,acc,TPR,TNR):
    tpr = tr/100
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=tpr)
    target = np.concatenate((y_train, y_test), axis=0)
    unique_clas = np.unique(target)
    #Import Random Forest Model
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    y_pred = np.concatenate((y_train, y_pred), axis=0)
    array(y_pred)
    pred_val = np.unique(y_pred)
    tp, tn, fn, fp = 0, 0, 0, 0
    for i1 in range(len(unique_clas)):
        # c = unique_clas[i1]
        c = unique_clas[i1]
        for i in range(len(target)):
            if (target[i] == c and y_pred[i] == c):
                tp = tp + 1
            if (target[i] != c and y_pred[i] != c):
                tn = tn + 1
            if (target[i] == c and y_pred[i] != c):
                fn = fn + 1
            if (target[i] != c and y_pred[i] == c):
                fp = fp + 1

    tn = tn / len(pred_val)
    tp = tp / len(pred_val)
    fn = fn / pred_val[len(pred_val) - 1]
    fp = fp / pred_val[len(pred_val) - 1]
    tn = tn / len(unique_clas)
    TPR.append(tp / (tp + fn))
    TPR.sort()
    TNR.append(tn / (tn + fp))
    TNR.sort()
    acc.append((tp + tn) / (tp + tn + fp + fn))
    acc.sort()
