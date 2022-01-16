#!/usr/bin/env python
# coding: utf-8

# # Predicting Mortgage Prepayment using Deep Neural Network 
# 
# 
# 
# 

# In[2]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[ ]:


#Read the csv file

data = pd.read_excel('/content/gdrive/My Drive/Colab Notebooks/ANLY535/ANLY535 Final Project/Main_Data.xlsx')
data.drop(['Date'],axis = 1 , inplace = True)
data.head()


# In[ ]:


X = data.drop('MBA Refi Index',axis =1).values
y = data['MBA Refi Index'].values
#splitting Train and Test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape )


# ## Exploratory Data Analysis

# In[ ]:


data.hist(figsize = (12,10))
plt.tight_layout()
plt.show()


# ## Scaling

# In[ ]:


#standardization scaler - fit&transform on train, fit only on test
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))


# ## Defining Callbacks

# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Define a checkpoint to save the data
checkpoint_name = 'Models/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
early_stop = EarlyStopping(monitor='val_mean_absolute_error', patience=10)
callbacks_list = [checkpoint, early_stop]


# # Hyper Parameter Tuning

# In[ ]:


get_ipython().system('pip install -q -U keras-tuner')
import keras_tuner as kt  


# In[ ]:


def MAPE(pred,actual):
  actual, pred = np.array(actual), np.array(pred)
  return np.mean(np.abs((actual-pred)/actual))*100

def model_builder(hp):
  model = Sequential()
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
# The Input Layer :
  model.add(Dense(units=hp_units, kernel_initializer='normal', activation='relu'))

# The Hidden Layers :
  model.add(Dense(256, kernel_initializer='normal',activation='relu'))
  model.add(Dense(256, kernel_initializer='normal',activation='relu'))
  model.add(Dense(256, kernel_initializer='normal',activation='relu'))
  model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
  model.add(Dense(1, kernel_initializer='normal',activation='linear'))
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(optimizer = Adam(learning_rate=hp_learning_rate), loss='MAPE', metrics=['mean_absolute_error','MAPE'])
  return model

tuner = kt.Hyperband(model_builder,
                     objective='MAPE',
                     max_epochs=10,
                     factor=3,
                     directory='/content/gdrive/My Drive/Colab Notebooks/ANLY535/ANLY535 Final Project/hp',
                     project_name='hp_tuning')
  
  


# In[ ]:


tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='mean_absolute_error', patience=10)])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")


# ## Train Model with Searched Hyperparameters

# In[ ]:


model = tuner.hypermodel.build(best_hps)


history = model.fit(X_train, y_train, epochs=500, validation_split=0.2, callbacks = callbacks_list)


# In[ ]:


plt.style.use('ggplot')

def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    x = range(1, len(loss) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(x, loss, label='Training MAPE')
    ax[0].plot(x, val_loss, label='Validation MAPE')
    ax[0].set_title('Training & Validation MAPE', fontsize=16)
    ax[0].set_xlabel('Epochs', fontsize=16)
    ax[0].set_ylabel('MAPE', fontsize=16)
    ax[0].legend()
    #
    # Plot the loss vs Epochs
    #
    ax[1].plot(x, mae, label='Training MAE')
    ax[1].plot(x, val_mae,label='Validation MAE')
    ax[1].set_title('Training & Validation MAE', fontsize=16)
    ax[1].set_xlabel('Epochs', fontsize=16)
    ax[1].set_ylabel('MAE', fontsize=16)
    ax[1].legend()

    plt.show()

plot_history(history)


# ## Results

# ### Mean Average Percentage Error (MAPE)

# In[ ]:


y_pred = model.predict(X_test)[:,0]
print("MAPE of trained model is ",MAPE(y_pred, y_test))


# In[ ]:


MAPE(y_pred, y_test)


# ### MAE, MSE and RMSE

# In[ ]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))


# ### Visualizing Predictions

# In[ ]:


# Visualizing Our predictions
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
# Perfect predictions
plt.plot(y_test,y_test,'b')


# In[ ]:


plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Actual')
plt.plot(y_pred, color = 'green', label = 'Predicted')
plt.title('MBA Refinance Index Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to html "/content/gdrive/My Drive/Colab Notebooks/ANLY535/ANLY535 Final Project/Team3 Final Project.ipynb"')

