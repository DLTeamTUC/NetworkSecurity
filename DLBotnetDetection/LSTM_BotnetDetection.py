#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:11:51 2019

@author: DLTeamTUC
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, TimeDistributed, MaxPooling1D, Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

#Download the CTU-13 Dataset to your desired folder
#Load them one by one
df1 = pd.read_csv("/path/to/CTU-13-Dataset/1/scenario1.csv")
df2 = pd.read_csv("/path/to/CTU-13-Dataset/2/scenario2.csv")
df3 = pd.read_csv("/path/to/CTU-13-Dataset/3/scenario3.csv")
df4 = pd.read_csv("/path/to/CTU-13-Dataset/4/scenario4.csv")
df5 = pd.read_csv("/path/to/CTU-13-Dataset/5/scenario5.csv")
df6 = pd.read_csv("/path/to/CTU-13-Dataset/6/scenario6.csv")
df7 = pd.read_csv("/path/to/CTU-13-Dataset/7/scenario7.csv")
df8 = pd.read_csv("/path/to/CTU-13-Dataset/8/scenario8.csv")
df9 = pd.read_csv("/path/to/CTU-13-Dataset/9/scenario9.csv")
df10 = pd.read_csv("/path/to/CTU-13-Dataset/10/scenario10.csv")
df11 = pd.read_csv("/path/to/CTU-13-Dataset/11/scenario11.csv")
df12 = pd.read_csv("/path/to/CTU-13-Dataset/12/scenario12.csv")
df13 = pd.read_csv("/path/to/CTU-13-Dataset/13/scenario13.csv")

#Concatenate all csv files into one training dataset
train = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13])

#Drop "Tag" which is benign traffics labelled as 0 and botnet traffic labelled as 1
#"Tag" is given manually to each scenario
train.dropna(subset=['Tag'], inplace=True)

le = LabelEncoder()
train['StartTime'] = le.fit_transform(train['StartTime'])
train['Dur'] = le.fit_transform(train['Dur'])
train['Proto'] = le.fit_transform(train['Proto'])
train['SrcAddr'] = le.fit_transform(train['SrcAddr'])
train['Sport'] = le.fit_transform(train['Sport'])
train['Dir'] = le.fit_transform(train['Dir'])
train['DstAddr'] = le.fit_transform(train['DstAddr'])
train['Dport'] = le.fit_transform(train['Dport'])
train['State'] = le.fit_transform(train['State'])
train['sTos'] = le.fit_transform(train['sTos'])
train['dTos'] = le.fit_transform(train['dTos'])
train['TotPkts'] = le.fit_transform(train['TotPkts'])
train['TotBytes'] = le.fit_transform(train['TotBytes'])
train['SrcBytes'] = le.fit_transform(train['SrcBytes'])
train['Label'] = le.fit_transform(train['Label'])
train['AvgByte'] = le.fit_transform(train['AvgByte']) #We add this additional feature to the CTU-13 dataset
train['AvgPacket'] = le.fit_transform(train['AvgPacket']) #We add this additional feature to the CTU-13 dataset

xtrain_Val = train.iloc[:,0:17].values
Ytrain = train.iloc[:,17].values

scaler = MinMaxScaler(feature_range=(0, 1))
Xtrain = scaler.fit_transform(xtrain_Val)


#Load Testing Dataset
df1 = pd.read_csv("/path/to/CTU-13-Dataset/1/scenario1.csv")
df2 = pd.read_csv("/path/to/CTU-13-Dataset/2/scenario2.csv")
df3 = pd.read_csv("/path/to/CTU-13-Dataset/3/scenario3.csv")
df4 = pd.read_csv("/path/to/CTU-13-Dataset/4/scenario4.csv")
df5 = pd.read_csv("/path/to/CTU-13-Dataset/5/scenario5.csv")
df6 = pd.read_csv("/path/to/CTU-13-Dataset/6/scenario6.csv")
df7 = pd.read_csv("/path/to/CTU-13-Dataset/7/scenario7.csv")
df8 = pd.read_csv("/path/to/CTU-13-Dataset/8/scenario8.csv")
df9 = pd.read_csv("/path/to/CTU-13-Dataset/9/scenario9.csv")
df10 = pd.read_csv("/path/to/CTU-13-Dataset/10/scenario10.csv")
df11 = pd.read_csv("/path/to/CTU-13-Dataset/11/scenario11.csv")
df12 = pd.read_csv("/path/to/CTU-13-Dataset/12/scenario12.csv")
df13 = pd.read_csv("/path/to/CTU-13-Dataset/13/scenario13.csv")

test = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13])
test.dropna(subset=['Tag'], inplace=True)

test['StartTime'] = le.fit_transform(test['StartTime'])
test['Dur'] = le.fit_transform(test['Dur'])
test['Proto'] = le.fit_transform(test['Proto'])
test['SrcAddr'] = le.fit_transform(test['SrcAddr'])
test['Sport'] = le.fit_transform(test['Sport'])
test['Dir'] = le.fit_transform(test['Dir'])
test['DstAddr'] = le.fit_transform(test['DstAddr'])
test['Dport'] = le.fit_transform(test['Dport'])
test['State'] = le.fit_transform(test['State'])
test['sTos'] = le.fit_transform(test['sTos'])
test['dTos'] = le.fit_transform(test['dTos'])
test['TotPkts'] = le.fit_transform(test['TotPkts'])
test['TotBytes'] = le.fit_transform(test['TotBytes']) 
test['SrcBytes'] = le.fit_transform(test['SrcBytes']) 
test['Label'] = le.fit_transform(test['Label'])
test['AvgByte'] = le.fit_transform(test['AvgByte']) #We add this additional feature to the CTU-13 dataset
test['AvgPacket'] = le.fit_transform(test['AvgPacket']) #We add this additional feature to the CTU-13 dataset

xtest_Val = test.iloc[:,0:17].values
Ytest = test.iloc[:,17].values

scaler = MinMaxScaler(feature_range=(0, 1))
Xtest = scaler.fit_transform(xtest_Val)

Xtrain = Xtrain.reshape((Xtrain.shape[0], 17, 1))
Xtest = Xtest.reshape((Xtest.shape[0], 17, 1))   #Input data shape: (samples, timesteps, features)

def LSTM_model():
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(17, 1)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
   
    model.compile(optimizer = adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model    

model = LSTM_model()

#With Callbacks
callbacks = [EarlyStopping('val_loss', patience=5)]
hist = model.fit(Xtrain, Ytrain, epochs=50, batch_size=50, validation_split=0.20, callbacks=callbacks, verbose=1)

#Without Callbacks
#hist = model.fit(Xtrain, Ytrain, epochs=50, batch_size=50, validation_split=0.20, verbose=1)


# predict probabilities for test set
yhat_probs = model.predict(Xtest, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(Xtest, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Ytest, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Ytest, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Ytest, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Ytest, yhat_classes)
print('F1 score: %f' % f1)

#Print Execution Time
print("Execution time = --- %s seconds ---" % (time.time() - start_time))

#Print Loss, Acc, and Confusion Matrix
f = plt.figure(1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss for LSTM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
f.show()

g = plt.figure(2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy for CNN')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
g.show()

h = plt.figure(3)
Ypred = model.predict_classes(Xtest)
cm = confusion_matrix(Ytest, Ypred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
labels = ['Normal', 'Botnet']
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
h.show()

raw_input()
