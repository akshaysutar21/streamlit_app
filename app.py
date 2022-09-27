import pandas as pd
import numpy as np
# import matplotlib as plt
import pandas_datareader as data
from datetime import datetime, timedelta
from keras.models import load_model
import streamlit as st
#x=st.number_input("Enter No. of days to be predicted")
start =  '1996-01-01'
now=datetime.now()
#now=datetime.now()+timedelta(days=x)
end=now.strftime("%Y-%m-%d")

st.title('Stock Prediction')

user_input=st.text_input('Enter Stock Ticker','GOOG')
df=data.DataReader(user_input,'yahoo',start,end)


#Descirbing data
st.subheader('Data till today')
st.write(df.tail())


#Visualization
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100  Moving Average')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df['Close'],'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 & 200  Moving Average')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma200,'g')
plt.plot(ma100,'r')
plt.plot(df['Close'],'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 10 Exponential moving average')
ema10=df.Close.ewm(span=10,adjust=False).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ema10,'r')
plt.plot(df['Close'],'b')
st.pyplot(fig)

#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Loading Model

model=load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df= past_100_days.append(data_testing,ignore_index=True)

input_data= scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test , y_test = np.array(x_test) , np.array(y_test)

#Making Predictions

y_predicted = model.predict(x_test)

scaler1=scaler.scale_
scale_factor= 1/scaler1[0]
y_predicted = y_predicted*scale_factor
y_test = y_test * scale_factor



#final Graph
st.subheader('Prediction vs  Original')
fig2 = plt.figure(figsize=(12,8))
#plt.plot(ff,'b',label='CLosing')
plt.plot(y_test , 'g', label='OG price')
plt.plot(y_predicted , 'r', label='Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


#Predicted Price for tomorrow close
last_100_days=df.Close[-100:]
last_100_days=pd.DataFrame(last_100_days)

last_100_days_scaled=scaler.transform(last_100_days)

x_test=[]

x_test.append(last_100_days_scaled)
x_test=np.array(x_test)


x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


pred_price=model.predict(x_test)
model.reset_states()

pred_price=scaler.inverse_transform(pred_price)

print(pred_price)
tomorrow=now+timedelta(days=1)
tom=datetime.date(tomorrow)
st.write('Predicted price for',tom,' is',pred_price)
