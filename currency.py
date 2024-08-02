import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import streamlit as st
from keras.models import  load_model

st.title('Cryptoprice Predictor')

use_input = st.text_input('Enter a currency to predict')

# Slider to select the number of years for data display
n_years = st.slider('Select the number of years for data display:', 1, 4)

if st.button('Predict'):
    df = yf.download(use_input, period=f'{n_years}y')

    # Describe data
    st.subheader(f'Data for the last {n_years} years')
    st.write(df.describe())

    # Plots
    st.subheader('Closing Price vs Time Chart ')
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df.Close, color='yellow')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100-day Moving Average')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(ma100, color='red')
    plt.plot(df.Close, color='yellow')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100-day and 200-day Moving Averages')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(ma100, color='red', label='100-day MA')
    plt.plot(ma200, color='green', label='200-day MA')
    plt.plot(df.Close, color='yellow', label='Closing Price')
    plt.legend()
    st.pyplot(fig)

    # Splitting data into train and test
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)
    model = load_model('model.h5')

    # Linear Regression Model
    model = LinearRegression()

    # Training data
    x_train = []
    y_train = []
    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i, 0])
        y_train.append(data_training_array[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    model.fit(x_train, y_train)

    # Testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i, 0])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    # Scaling back the predicted and actual values
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor


    # Final graph
    def plot_transparent_graph():
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.style.use('dark_background')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

    def main():
        st.title('Crypto Price Predicted Analysis')

        # Call the function to plot the transparent graph
        plot_transparent_graph()

    if __name__ == "__main__":
        main()
