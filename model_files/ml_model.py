import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

data = pd.read_csv("SRE_data.csv")
#data.head()
data.drop("Name",axis=1,inplace=True)
data_col=["open","high","low","close"]

df=data.drop("date",axis=1)
fig,axes=plt.subplots(4,1,dpi=100,figsize=(10,6))
for i,ax in enumerate(axes.flatten()):
    dataa=df[df.columns[i]]
    ax.plot(dataa,color="b",linewidth=1)
    ax.set_title(df.columns[i])
    
plt.tight_layout()
df=data.reset_index()["close"]

from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler(feature_range=(0,1))
df1=scale.fit_transform(np.array(df).reshape(-1,1))

train_data=int(len(df1)*0.70)
test_data=len(df1)-train_data
train_data=df1[:-test_data]
test_data=df1[-test_data:]


def create_dataset(dataset,time_step=1):
    datax,datay=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        datax.append(a)
        datay.append(dataset[i+time_step,0])
    return np.array(datax),np.array(datay)

time_step=100
x_train,y_train=create_dataset(train_data,time_step)
x_test,y_test=create_dataset(test_data,time_step)

(x_train.shape,y_train.shape)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])

epochs=100
batch_size=64
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size,verbose=1)

train_predict=model.predict(x_train)
test_predict=model.predict(x_test)

train_predict=scale.inverse_transform(train_predict)
test_predict=scale.inverse_transform(test_predict)

import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

math.sqrt(mean_squared_error(y_test,test_predict))
look_back=100
trainPredictPlot=np.empty_like(df1)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
testPredictPlot=np.empty_like(df1)
testPredictPlot[:,:]=np.empty_like(df1)
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1]=test_predict
plt.plot(scale.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()   

x_input=test_data[278:].reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

import array
lst_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape(1,n_steps,1)
        yhat=model.predict(x_input,verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
print(lst_output)

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

df2=df1.tolist()
df2.extend(lst_output)

plt.plot(day_new,scale.inverse_transform(df1[1159:]))
plt.plot(day_pred,scale.inverse_transform(lst_output))

df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])