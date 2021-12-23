import math, numpy as np
from CNN import cnn


def train_test_split(data, clas, tr_per):
    train_x, train_y = [], []  # training data, training class
    test_x, test_y, label = [], [], []  # testing data, testing class, label
    uni = np.unique(clas)  # unique class

    for i in range(len(uni)):  # n_unique class
        tem = []
        for j in range(len(clas)):
            if (uni[i] == clas[j]):  # if class of data = unique class
                tem.append(data[j])  # get unique class as tem

        tp = int((len(tem) * tr_per) / 100)  # training data size

        for k in range(len(tem)):
            if (k < tp):  # adding training data & its class
                train_x.append(tem[k])
                train_y.append(float(uni[i]))
                label.append(float(uni[i]))
            else:  # adding testing data & its class
                test_x.append(tem[k])
                test_y.append(float(uni[i]))
                label.append(float(uni[i]))
    return train_x, train_y, test_x, test_y, label


def bound(f_data):
    fe = []
    sq = int(math.sqrt(len(f_data[0])))
    n = int(sq * sq)
    for i in range(len(f_data)):
        tem = []
        for j in range(n):  # attributes in each row
            tem.append(f_data[i][j])  # add value to tem array
        fe.append(tem)  # add 1 row of array value to fe
    return fe




def callmain(data, label,trp,acc,TPR,TNR):
    train_x, train_y, test_x, test_y, target = train_test_split(data, label, trp)  # splitting training & testing data
    feature = np.asarray(bound(data))
    feature = feature.astype('float')
    y_pred = cnn.classify(np.array(feature), np.array(target), np.array(train_y), np.array(test_y), trp)
    target = test_y
    unique_clas = np.unique(test_y)
    tp, tn, fn, fp = 0, 0, 0, 0
    pred_val = np.unique(target)
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
    TNR.append(tn / (tn + fp))
    acc.append((tp + tn) / (tp + tn + fp + fn))
