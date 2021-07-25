# Imports
import pandas as pd
import streamlit as st
import io
import datetime
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import math as math
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
# Local imports
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from SessionState import SessionState as session_state
import Util as util


KEY_FORECAST = "Forecast"
KEY_DATA_COL = "Data"
KEY_CONFIDENCE_UP = "Upper Confidence"
KEY_CONFIDENCE_DOWN = "Lower Confidence"


# Main Function building this page
def build_page_forecasting(session: session_state):
    # Load CSV
    in_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, type='csv')
    if in_file is not None:
        # Load and refactor data
        data = pd.read_csv(io.StringIO(in_file.read().decode('utf-8')), sep=',', index_col=0)
        data.index = pd.to_datetime(data.index)
        data.columns = [KEY_DATA_COL]
        # Derive freq from data (Daily, weekly, monthly etc.)
        freq_dictionary = {'D': "Daily", 'W': "Weekly", 'M': "Montly", 'Y': "Yearly"}
        best_freq = [999999, None]
        for date_freq in freq_dictionary.keys():
            date_difference = data.index[1] - data.index[0]
            rem = math.fabs(date_difference / np.timedelta64(1, date_freq[0]) - 1.0)
            if rem < best_freq[0]:
                best_freq = [rem, date_freq]
        date_freq = best_freq[1]
        # By reformating the CSV datetime we make sure the datetime indexes of the data and prediction line up later
        data.index = pd.date_range(start=data.index[0], periods=len(data.index), freq=date_freq)
        # Present Data
        graph = util.make_timeseries_graph(data, title="Complete Timeseries Data")
        st.markdown(f"Datapoints: **{len(data)}**, Date Range: **{data.index[0].date()}** - **{data.index[-1].date()}**, Frquency: **{freq_dictionary[date_freq]}**")
        st.pyplot(graph)

        # Select Model {model key: [build function, if has confidence interval]}
        opt_NONE, opt_ARIMA, opt_PROPHET, opt_RNNLSTM = ("None", "ARIMA", "Prophet", "RNN LSTM")
        in_model_type = st.selectbox("Model for Forecasting", (opt_NONE, opt_ARIMA, opt_PROPHET, opt_RNNLSTM))
        build_model_dict = {opt_ARIMA: (build_model_ARIMA, True),
                            opt_PROPHET: (build_model_Prophet, True),
                            opt_RNNLSTM: (build_model_RNNLSTM, False)}

        # One of the models has been picked and we call the the method to handle it
        if in_model_type != opt_NONE:
            # Inputs
            in_model_train_test_threshold = st.number_input("Training Percentage", value=50, step=1, min_value=1, max_value=100)
            in_model_train_test_threshold = in_model_train_test_threshold / 100.0
            in_model_horizon = st.number_input("Number of dates to predict", value=int(len(data) * 0.5), step=1, min_value=1)
            # Confidence Interval
            confidence_value = -1
            if build_model_dict[in_model_type][1]:
                in_confidence = st.number_input("Confidence Interval", value=0.85, step=0.05, min_value=0.01, max_value=0.99)
                confidence_value = in_confidence

            # Training Data & prediction Range
            training_data = data.iloc[:int(len(data) * in_model_train_test_threshold)]
            prediction_date_range = pd.date_range(start=training_data.index[-1], periods=in_model_horizon+1, freq=date_freq)

            # Start Training:
            in_button_model_build = st.button("Train & Run Model")
            result = None
            if in_button_model_build:
                result = build_model_dict[in_model_type][0](training_data, prediction_date_range, confidence_value)

            # Returned result is None if training not done at current point else [prediction, confidence (None if N/A)]
            if result is not None:
                prediction, confidence_interval = result
                # Plot
                fig = util_plot(data, prediction, confidence_interval)
                st.pyplot(fig)
                # Evaluate
                calcuate_evals(data, prediction)


# Constructs the UI necessary for training ARIMA and returns result data
def build_model_ARIMA(training_data: pd.DataFrame, prediction_date_range, confidence_value: float):
    # Train the model
    pdq, seasonal_pdq, _ = find_ARIMA_pdq(training_data, 1)
    model = ARIMA(training_data, order=pdq, seasonal_order=seasonal_pdq, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    # Mean Prediction and Confidence Intervals as DataFrames
    pred = model_fit.get_prediction(start=prediction_date_range[0], end=prediction_date_range[-1], dynamic=False)
    prediction = pd.DataFrame(data={KEY_FORECAST: pred.predicted_mean}, index=prediction_date_range)
    confidence_alpha = 1.0 - confidence_value
    confidence_interval = pred.conf_int(alpha=confidence_alpha)

    out = [prediction, confidence_interval]
    return out


# Constructs and builds Prophet Model
def build_model_Prophet(training_data: pd.DataFrame, prediction_date_range, confidence_value: float):
    training_data = pd.DataFrame(data={'ds': training_data.index, 'y': training_data[KEY_DATA_COL]})
    model = Prophet(interval_width=confidence_value)
    model.fit(training_data)

    forecast = model.predict(pd.DataFrame(data=prediction_date_range, columns=['ds']))
    prediction = pd.DataFrame(data=forecast['yhat'].to_numpy(), index=prediction_date_range, columns=[KEY_FORECAST])
    confidence_interval = pd.DataFrame(data={KEY_CONFIDENCE_DOWN: forecast['yhat_lower'].to_numpy(), KEY_CONFIDENCE_UP: forecast['yhat_upper'].to_numpy()}, index=prediction_date_range)

    out = [prediction, confidence_interval]
    return out


# Constructs and builds RNN LSTM model
def build_model_RNNLSTM(training_data: pd.DataFrame, prediction_date_range, confidence_value: float):
    #TODO: Completely rework this method
    def univariate_data(dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + target_size])
        return np.array(data), np.array(labels)

    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler()
    norm_data = min_max_scaler.fit_transform(training_data.values)

    TRAIN_SPLIT = int(len(norm_data) * 0.8)
    past_history = 5
    future_target = 0

    x_train, y_train = univariate_data(norm_data, 0, TRAIN_SPLIT, past_history, future_target)
    x_test, y_test = univariate_data(norm_data, TRAIN_SPLIT, None, past_history, future_target)

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
    from tensorflow.keras.optimizers import Adam

    num_units = 64
    learning_rate = 0.0001
    activation_function = 'sigmoid'
    adam = Adam(lr=learning_rate)
    loss_function = 'mse'
    batch_size = 5
    num_epochs = 50

    model = Sequential()

    # model.add(LSTM(units=num_units, activation=activation_function, input_shape=(None, 1)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.compile(optimizer=adam, loss=loss_function)

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=num_epochs,
        shuffle=False
    )

    model.summary()

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    predictions = pd.DataFrame(min_max_scaler.inverse_transform(model.predict(x_test)))

    util.make_timeseries_multi_graph([x_test, predictions])

    print("Done")


# Constructs a plot rendering data, prediction and confidence_interval of predicition
def util_plot(data: pd.DataFrame, prediction: pd.DataFrame, confidence_interval=None):
    fig, ax = plt.subplots()
    ax.plot(data, label="Observed")
    # prediction.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
    prediction.plot(ax=ax, label='Forecast', alpha=.7)
    if confidence_interval is not None:
        ax.fill_between(confidence_interval.index, confidence_interval.iloc[:, 0], confidence_interval.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    return fig


# Calculates the eval measures between the overlapping parts of the 2 provided DataFrames
def calcuate_evals(expected: pd.DataFrame, predicted: pd.DataFrame):
    # Cut down data & prediction to only shared entries
    drop_list_ex = np.invert(expected.index.isin(predicted.index))
    drop_list_pd = np.invert(predicted.index.isin(expected.index))
    expected = expected.drop(expected[drop_list_ex].index)
    date_range = expected.index
    expected = expected.to_numpy()
    predicted = predicted.drop(predicted[drop_list_pd].index).to_numpy()
    # Calculate Different Error Measures
    forecast_errors = [expected[i] - predicted[i] for i in range(len(expected))]
    mean_forecast_error = sum(forecast_errors)[0] * 1.0/len(forecast_errors)
    mae = metrics.mean_absolute_error(expected, predicted)
    mse = metrics.mean_squared_error(expected, predicted)
    rmse = math.sqrt(mse)
    # TODO: Return and integrate instead of printing (also use forecast errors?)
    st.markdown(f"Error calculation based on entries between: **{date_range[0].date()}** - **{date_range[-1].date()}**")
    st.markdown(f"Mean Error: **{mean_forecast_error}**, MAE: **{mae}**, MSE: **{mse}**, RMSE: **{rmse}**")
    # TODO: Move this somewhere more sensible
    abs_forecast_errors = [math.fabs(forecast_errors[i]) for i in range(len(forecast_errors))]
    fig = util.make_timeseries_graph(pd.DataFrame(data=abs_forecast_errors, index=date_range), title="Absolute Errors")
    st.pyplot(fig)


# Resamples given Dataframe into Monthly Frequency
def util_resample(data: pd.DataFrame):
    out = data.resample('M').sum()
    return out


# Returns pdq and seasonal pdqs for an ARIMA model based on the given Data Frame(Using Grid Search)
def find_ARIMA_pdq(training_data: pd.DataFrame, pdq_range: int):
    import itertools
    import warnings
    best_pdq = None
    best_seasonal = None
    best_aic = 9999999
    # Define the p, d and q parameters to take any value between 0 and given range
    p = d = q = range(0, pdq_range + 1)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore")  # specify to ignore warning messages
    # Find best p, d & q with default seasonality P D Q
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = ARIMA(training_data, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()
                aic = results.aic
                if aic < best_aic:
                    best_pdq = param
                    best_seasonal = param_seasonal
                    best_aic = aic
            except:
                continue
    out = [best_pdq, best_seasonal, best_aic]
    return out

