import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import streamlit as st


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def catboost_forecast(series, lags, iterations, depth, learning_rate, loss_function):
    x, y = sliding_windows(series, lags)

    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        loss_function=loss_function,
        learning_rate=learning_rate,
        random_seed=0
    )

    model.fit(x, y)

    res = []
    data = series

    for i in range(0, 4):
        test = np.array(data[-lags:])
        predict = model.predict(np.expand_dims(test, axis=0))
        res.append(predict.item())
        data.append(predict.tolist()[0])

    return res


numbers = st.text_input("Enter sequence")
lags_select = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
iteration_select = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1550, 2000]
depth_select = [2, 4, 6, 8, 10]
learning_rate_select = [0.1, 0.3, 0.05, 0.01]
loss_function_select = ['RMSE', 'MAPE']

lags = st.select_slider(
    'Select lag',
    options=lags_select)

iteration = st.select_slider(
    'Select iteration',
    options=iteration_select)

depth = st.select_slider(
    'Select tree depth',
    options=depth_select)

learning_rate = st.select_slider(
    'Select learning rate',
    options=learning_rate_select)

loss_function = st.select_slider(
    'Select loss function',
    options=loss_function_select)

if numbers:
    series = [float(i) for i in numbers.split(",")]
    train_series = series[:-4]

    res = catboost_forecast(train_series, int(lags), int(iteration), int(depth), float(learning_rate),
                            str(loss_function))

    test_series = series[-4:]

    final_df = pd.DataFrame({'Test data': test_series, 'Prediction': res})

    final_df['Difference'] = final_df['Test data'] - final_df['Prediction']
    final_df['Error%'] = (final_df['Difference'] / final_df['Test data']) * 100
    st.dataframe(final_df)
