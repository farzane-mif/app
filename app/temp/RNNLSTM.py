#!/usr/bin/env python
# coding: utf-8

# # Predicting the price of Bitcoin, intro to LSTM
# 

# In[6]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
#import seaborn as sns
import os


# In[7]:


os.chdir('C:/Users/FarzanehAkhbar/Documents/FAAS/bitcoin/New folder/bitcoin-predict-master/data')


# ## Data Exploration

# In[8]:


data = pd.read_csv("bitcoin.csv")
data = data.sort_values('Date')
data.head()


# In[29]:


data['Date'].min()


# In[30]:


data['Date'].max()


# In[9]:


price = data[['Close']]

plt.figure(figsize = (15,9))
plt.plot(price)
plt.xticks(range(0, data.shape[0],50), data['Date'].loc[::50],rotation=45)
plt.title("Bitcoin Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (USD)',fontsize=18)
plt.show()


# In[10]:


price.info()


# ## Data Preparation

# ### Normalization

# In[11]:


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

norm_data = min_max_scaler.fit_transform(price.values)


# In[12]:


print(f'Real: {price.values[0]}, Normalized: {norm_data[0]}')
print(f'Real: {price.values[500]}, Normalized: {norm_data[500]}')
print(f'Real: {price.values[1200]}, Normalized: {norm_data[1200]}')


# ### Data split

# In[13]:


def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

past_history = 5
future_target = 0

TRAIN_SPLIT = int(len(norm_data) * 0.8)


x_train, y_train = univariate_data(norm_data,
                                   0,
                                   TRAIN_SPLIT,
                                   past_history,
                                   future_target)

x_test, y_test = univariate_data(norm_data,
                                 TRAIN_SPLIT,
                                 None,
                                 past_history,
                                 future_target)


# ## Build the model

# In[21]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
#from keras.layers import Dense, LSTM, LeakyReLU, Dropout


# In[15]:


#from tensorflow.keras.layers import Adam
from tensorflow.keras.optimizers import Adam


# In[16]:


num_units = 64
learning_rate = 0.0001
activation_function = 'sigmoid'
adam = Adam(lr=learning_rate)
loss_function = 'mse'
batch_size = 5
num_epochs = 50


# In[17]:


model = Sequential()


# In[18]:


model.add(LSTM(units = num_units, activation=activation_function, input_shape=(None, 1)))


# In[22]:


# Initialize the RNN
#model = Sequential()
#model.add(LSTM(units = num_units, activation=activation_function, input_shape=(None, 1)))
model.add(LeakyReLU(alpha=0.5))
#model.add(Dropout(0.1))
#model.add(Dense(units = 1))

# Compiling the RNN
#model.compile(optimizer=adam, loss=loss_function)


# In[23]:


model.add(Dropout(0.1))
model.add(Dense(units = 1))


# In[24]:


# Compiling the RNN
model.compile(optimizer=adam, loss=loss_function)


# In[25]:


model.summary()


# ## Train the model

# In[26]:


# Using the training set to train the model
history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    batch_size=batch_size,
    epochs=num_epochs,
    shuffle=False
)


# In[27]:


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()

plt.show()


# ## Prediction
# For each of the items we used for the validation, let's now predict them so we can compare how well we did.

# In[28]:


original = pd.DataFrame(min_max_scaler.inverse_transform(y_test))
predictions = pd.DataFrame(min_max_scaler.inverse_transform(model.predict(x_test)))

ax = sns.lineplot(x=original.index, y=original[0], label="Test Data", color='royalblue')
ax = sns.lineplot(x=predictions.index, y=predictions[0], label="Prediction", color='tomato')
ax.set_title('Bitcoin price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)


# In[ ]:




