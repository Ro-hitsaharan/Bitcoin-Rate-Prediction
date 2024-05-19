
#for calculations and importing dataframes
import numpy as np
import pandas as pd
import math

#for scaling, metrics and trnsorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

#for building LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

#for visualizing the prediction
from bokeh.plotting import figure
from bokeh.io import output_notebook,show,push_notebook
from bokeh.models import Legend

from google.colab import files
uploaded = files.upload()

bi = pd.read_csv('Bitcoin.csv')
bi.head()

#counting the total number of cryptocurrencies
bi['rpt_key'].value_counts()
#only usd for our prediction
bi1= bi.loc[(bi['rpt_key'] == 'btc_usd')]
bi1.head()

#setting a certain thershold date above which we start predicting
bi1 = bi1.reset_index(drop=True)
bi1['datetime'] = pd.to_datetime(bi['datetime_id'])
bi1 = bi1.loc[bi1['datetime']>pd.to_datetime('2019-06-28 00:00:00')]

#keeping only the required columns for prediction
bi1 = bi1[['datetime','last','low','high','volume']]
bi1.head()

bi2 = bi1['last']
bi2

#Scaling the features ranging from one to two
scaler = MinMaxScaler(feature_range=(1,2))
bi2 = scaler.fit_transform(np.array(bi2).reshape(-1,1))

bi2

out = [] #store those in the dummy variable
for i in bi2:
  out.append(i[0])
bi2 = out

bi2

#splitting the dataset into 60-40 for testing and traning
train_size = int(len(bi2)*0.6)
test_size = len(bi2) - train_size
train, test = bi2[0:train_size], bi2[train_size:len(bi2)]
print(len(train), len(test))

#convert an array of values into a dataset matrix
def create_dataset(dataset,look_back):
  dataX,dataY=[],[]
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back)]
    dataX.append(a)
    dataY.append(dataset[i+look_back])
    return np.array(dataX),np.array(dataY)

look_back = 10
trainX,trainY = create_dataset(train,look_back)
testX,testY = create_dataset(test,look_back)

trainX

trainY

testX

testY

#reshape input to be[sample,time steps,features]
trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

trainX.shape

"""**Building LSTM Model**"""

model = Sequential()
model.add(LSTM(4,input_shape=(1,look_back)))#fourth hidden layer
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=100,batch_size=256,verbose=1)

"""Predicting the model

"""

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

"""Finding the RSME value"""

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' %(trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], trainPredict[:,0]))
print('Test Score: %.2f RMSE' %(testScore))

#shift train predictions for plotting
trainPredictPlot = np.zeros(len(bi2))
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict[:,0]

#shift test predictions for plotting
testPredictPlot = np.zeros(len(bi2),dtype = np.float64)
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(bi2)-1]= testPredict[:,0]

output_notebook()

#perdiction plot
p=figure(width=900,height=300)
p.line(np.arange(len(bi1['last'])),bi1['last'],legend_label = 'Actual',color = 'Black')
tpp = pd.DataFrame(trainPredictPlot,columns=["close"],index = bi1.index).close
qpp = pd.DataFrame(testPredictPlot,columns=["close"],index = bi1.index).close
p.line(np.arange(len(tpp)),tpp,legend_label = 'Training',color = 'Blue')
p.line(np.arange(len(qpp)),qpp,legend_label = 'Testing',color = 'Red')
p.legend.location="top_right"
show(p)

