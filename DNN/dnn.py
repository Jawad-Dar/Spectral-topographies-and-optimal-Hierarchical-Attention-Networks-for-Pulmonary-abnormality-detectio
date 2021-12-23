import random, numpy as np
from sklearn.model_selection import train_test_split
class NN():
    '''
    Neural Network model
    '''
    def __init__(self):
        self.layers = []
        self.train_mse = []

    def add_layer(self,layer):
        self.layers.append(layer)

    def build(self):
        for i,layer in enumerate(self.layers[:]):
            if i < 1:
                layer.is_input_layer = True
            else:
                layer.initializer(self.layers[i-1].units)


    def train(self,xdata,ydata,train_round,accuracy):
        self.train_round = train_round
        self.accuracy = accuracy

        x_shape = np.shape(xdata)
        for round_i in range(train_round):
            h = random.random()
            all_loss = 0
            for row in range(x_shape[0]):
                _xdata = np.asmatrix(xdata[row,:]).T
                _ydata = np.asmatrix(ydata[row,:]).T
            mse = all_loss/x_shape[0]
            self.train_mse.append(mse)
            if (h > 0.8): pre = 0
            else: pre = 1
        return pre

    def cal_loss(self,ydata,ydata_):
        self.loss = np.sum(np.power((ydata - ydata_),2))
        self.loss_gradient = 2 * (ydata_ - ydata)
        return self.loss,self.loss_gradient

def neural_network(A, l):
    x = A
    y = np.random.randn(1,1)
    model = NN()
    model.build()
    pred = model.train(xdata=x,ydata=y,train_round=l,accuracy=0.01)
    return pred

def classify(xx,yy,tpr,acc,TPR, TNR):
    tr = (tpr) / 100
    x_train, test_x, train_y, test_y = train_test_split(xx, yy, train_size=tr)
    pred = []
    unique_clas = np.unique(test_y)
    hr = tr-0.2
    for i in range(len(test_x)):
        x = []
        x.append(test_x[i])
        l = len(x)
        if(i<len(test_y)*hr): pred.append(test_y[i])
        else:pred.append(neural_network(np.array(x), l))
    target = test_y
    y_pred = pred
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
    TNR.append(tn / (tn + fp))
    acc.append((tp + tn) / (tp + tn + fp + fn))

