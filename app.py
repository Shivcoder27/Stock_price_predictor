import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt 

# Correcting the file path
model = load_model('C:/Users/shivashish kaushik/Desktop/tensorflow/Price Prediction ML/Stock Prediction Model.keras')

st.header('Stock Market Predictor')
stock = st.text_input('Enter the Stock symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(ma_50_days, color='red', label='MA50')
ax1.plot(data.Close, color='green', label='Close Price')
ax1.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(ma_50_days, color='red', label='MA50')
ax2.plot(ma_100_days, color='blue', label='MA100')
ax2.plot(data.Close, color='green', label='Close Price')
ax2.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.plot(ma_100_days, color='red', label='MA100')
ax3.plot(ma_200_days, color='blue', label='MA200')
ax3.plot(data.Close, color='green', label='Close Price')
ax3.legend()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

# Converting the lists to numpy arrays
x = np.array(x)
y = np.array(y)

predict = model.predict(x)
scale = 1 / scaler.scale_
predict = predict * scale

y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.plot(predict, color='red', label='Predicted')
ax4.plot(y, color='green', label='Original')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)
