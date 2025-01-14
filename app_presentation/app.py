import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, r2_score


start = '2010-01-01'
end = '2024-12-09'

st.title('stock trend prediction')

user_input = st.text_input("Enter stock ticker","AAPL")
df = yf.download(user_input, start=start, end=end)
df2 =yf.download(user_input, start=start, end=end)

st.subheader('Data from 2010 - 2024')
st.write(df.describe())

#visualize
st.subheader('closing price vs time chart ')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing price vs time chart with 100MA ')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing price vs time chart with 100MA & 200MA ')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close)
st.pyplot(fig)

#splitting data for low and high
data_training_high = pd.DataFrame(df2['High'][0:int(len(df)*0.70)])
data_testing_high = pd.DataFrame(df2['High'][int(len(df)*0.70):])

data_training_low = pd.DataFrame(df2['Low'][0:int(len(df)*0.70)])
data_testing_low = pd.DataFrame(df2['Low'][int(len(df)*0.70):])

#spliting data into training and testing , we are making predictions based on closing price
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) 
scaler_high = MinMaxScaler(feature_range=(0, 1))
scaler_low = MinMaxScaler(feature_range=(0, 1))

data_training_high_array = scaler_high.fit_transform(data_training_high)
data_training_low_array = scaler_low.fit_transform(data_training_low)
data_training_array = scaler.fit_transform(data_training)

#example is say we train on 10 days and we want to predict for 11th day

#loading project
model = load_model('ml_project.h5')       #closing price model
model_high = load_model('model_high.h5')  # Model for high price prediction
model_low = load_model('model_low.h5')    # Model for low price prediction
#to predict the next value we need the value of the previous 100 days
past_100_days = data_training.tail(100)
past_100_days_high = data_training_high.tail(100)
past_100_days_low = data_training_low.tail(100)  

final_df_high = pd.concat([past_100_days_high, data_testing_high], ignore_index=True)
final_df_low = pd.concat([past_100_days_low, data_testing_low], ignore_index=True)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data_high = scaler_high.fit_transform(final_df_high)
input_data_low = scaler_low.fit_transform(final_df_low)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
x_test_high = []
y_test_high = [] 
x_test_low = []
y_test_low = []  

for i in range(100, input_data_low.shape[0]):
    x_test_low.append(input_data_low[i-100:i])
    y_test_low.append(input_data_low[i, 0]) 

for i in range(100, input_data_high.shape[0]):
    x_test_high.append(input_data_high[i-100:i])
    y_test_high.append(input_data_high[i, 0]) 

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])  

x_test,y_test = np.array(x_test),np.array(y_test)
x_test_high, y_test_high = np.array(x_test_high), np.array(y_test_high)
x_test_low, y_test_low = np.array(x_test_low), np.array(y_test_low)

#making predictions 
y_predicted = model.predict(x_test)
y_high_predictions = model_high.predict(x_test_high)
y_low_predictions = model_low.predict(x_test_low)

scaler1 = scaler.scale_
scaler2 = scaler_high.scale_
scaler3= scaler_low.scale_
scale_factor = 1/scaler1[0]
scale_factor_high = 1/scaler2[0]
scale_factor_low = 1/scaler3[0]

y_predicted_high = y_high_predictions * scale_factor_high
y_predicted_low = y_low_predictions * scale_factor_low  
y_test_high = y_test_high * scale_factor_high
y_test_low = y_test_low * scale_factor_low  
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor  

#final graph 1 
st.subheader('predictions vs origional using RNN-LSTM')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

mae = mean_absolute_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)
st.subheader('Model Performance Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
col2.metric("R-squared (R²)", f"{r2:.2f}")
col3.metric("Accuracy", f"{100 - mae:.2f}%")

#final graph 2 
st.subheader('predictions daily high vs origional daily high using RNN-LSTM')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test_high,'b',label='Original Price')
plt.plot(y_predicted_high,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3) 

mae_high = mean_absolute_error(y_test_high, y_predicted_high)
r2_high = r2_score(y_test_high, y_predicted_high)
accuracy_high = 100 - mae_high  # Accuracy as 100 - MAE
st.subheader('Model Performance Metrics for High Prices')
col1, col2, col3 = st.columns(3)
col1.metric("Mean Absolute Error (MAE)", f"{mae_high:.2f}")
col2.metric("R-squared (R²)", f"{r2_high:.2f}")
col3.metric("Accuracy", f"{accuracy_high:.2f}%")

#final graph 3 
st.subheader('predictions daily low vs origional daily low using RNN-LSTM')
fig4 = plt.figure(figsize=(12,6))
plt.plot(y_test_low,'b',label='Original Price')
plt.plot(y_predicted_low,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4) 

mae_low = mean_absolute_error(y_test_low, y_predicted_low)
r2_low = r2_score(y_test_low, y_predicted_low)
accuracy_low = 100 - mae_low  # Accuracy as 100 - MAE
st.subheader('Model Performance Metrics for Low Prices')
col1, col2, col3 = st.columns(3)
col1.metric("Mean Absolute Error (MAE)", f"{mae_low:.2f}")
col2.metric("R-squared (R²)", f"{r2_low:.2f}")
col3.metric("Accuracy", f"{accuracy_low:.2f}%")