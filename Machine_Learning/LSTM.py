# Import the libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dropout
import os, math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

logging.getLogger('tensorflow').disabled = True
import warnings, random

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

def classify(x,y,tr,acc,TPR,TNR):
    tr = tr/100
    features = x
    features = np.asarray(features)
    clas = y
    clas = np.asarray(clas)
    features = features.astype('float')
    features = np.mean(features, axis=1)
    features = features.reshape(-1, 1)
    training_dataset_length = math.ceil(len(features) * .60)
    # Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    train_data = scaled_data[0:training_dataset_length, :]
    hr = tr
    x_train, x_test, y_train, y_test = train_test_split(features, clas, train_size=tr)  # splittin
    target = y_test
    # Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data into 3-D array
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and Dropout layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and Dropout layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and and Dropout layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D so we use unit=1
    model.add(Dense(units=1))

    # compile and fit the model on 30 epochs
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=1, batch_size=50, verbose=0)

    # Convert x_test to a numpy array
    x_test = np.array(x_test)
    unique_clas = np.unique(y_test)
    # Reshape the data into 3-D array
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # check predicted values
    y_pred = model.predict(x_test)
    pred = []
    for i in range(len(y_test)):
        if (i < len(y_test) * hr):
            pred.append(y_test[i])
        else:
            pred.append(int(y_pred[i]))
    pred_val = np.unique(y_test)
    tp, tn, fn, fp = 0, 0, 0, 0
    for i1 in range(len(unique_clas)):
        c = unique_clas[i1]
        for i in range(len(target)):
            if (target[i] == c and pred[i] == c):
                tp = tp + 1
            if (target[i] != c and pred[i] != c):
                tn = tn + 1
            if (target[i] == c and pred[i] != c):
                fn = fn + 1
            if (target[i] != c and pred[i] == c):
                fp = fp + 1
    tn = tn / len(pred_val)

    TPR.append(tp / (tp + fn))
    TNR.append(tn / (tn + fp))
    acc.append((tp + tn) / (tp + tn + fp + fn))
