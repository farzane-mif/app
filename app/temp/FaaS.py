

import datetime
#import timesynth as ts
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import os
import matplotlib.pyplot as plt
from fbprophet import Prophet
from pandas import to_datetime
from sklearn.metrics import mean_squared_error

os.chdir('C:/Users/FarzanehAkhbar/Documents/FAAS/data')


time_sampler = ts.TimeSampler(stop_time=20)
# Sampling irregular time samples
irregular_time_samples = time_sampler.sample_irregular_time(num_points=500, keep_percentage=50)
# Initializing Sinusoidal signal
sinusoid = ts.signals.Sinusoidal(frequency=0.25)
# Initializing Gaussian noise
white_noise = ts.noise.GaussianNoise(std=0.3)
# Initializing TimeSeries class with the signal and noise objects
timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
# Sampling using the irregular time samples
samples, signals, errors = timeseries.sample(irregular_time_samples)


samples1 = samples.reshape((250,1))
errors1 = errors.reshape((250,1))
signals1 = signals.reshape((250,1))




df = pd.DataFrame(signals1)
df.columns=['signal']
#arr = sparse.coo_matrix((errors1,samples1), shape=(250,1))
df['sample'] = samples1
df['error'] = errors1
#make all the values above zero
df['sample'] = df['sample'].apply(abs)
df['error'] = df['error'].apply(abs)
df['signal'] = df['signal'].apply(abs)
df['val'] = df['signal'] + df['error']

#adding date
start_date = datetime.date(2019, 9, 30)
date_list = [start_date - datetime.timedelta(days=x) for x in range(250)]
datels = np.asarray(date_list)
df['date'] = datels

#save the synthetic data as csv file
df.to_csv("signaldata.csv")


# #plot generated time series
# def plot_time_series(time, values, label):
#     plt.figure(figsize=(10,6))
#     plt.plot(time, values)
#     plt.xlabel("Time", fontsize=20)
#     plt.ylabel("Value", fontsize=20)
#     plt.title(label, fontsize=20)
#     plt.grid(True)
# plot_time_series(df.index, df.signal, label="Seasonality + Upward Trend + Noise")
# plot_time_series(df.index, df.error, label="Seasonality + Upward Trend + Noise")
# plot_time_series(df.index, df.sample, label="Seasonality + Upward Trend + Noise")


#start the prediction
dfpre = df.drop(['signal', 'sample', 'error'],1)
dfpre.columns = ['y','ds']
column_names = ['ds','y']

dfpre = dfpre.reindex(columns=column_names)
dfpre['ds']= to_datetime(dfpre['ds'])
m = Prophet()
m.fit(dfpre)

y = pd.DataFrame(dfpre['y'][:15])
fut = pd.DataFrame(dfpre['ds'][:15])
# future = m.make_future_dataframe(periods=100)


forecast = m.predict(fut)
res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

rmse = (mean_squared_error(y, forecast.yhat))**(1/2)
rmse


