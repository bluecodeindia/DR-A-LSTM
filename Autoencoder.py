#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
# warnings.filterwarnings('once')
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# In[2]:



from keras.models import Sequential, load_model, save_model
from keras.layers import Dense,Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Activation, Embedding, multiply
from keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
import glob
import keras
from datetime import datetime
from keras.callbacks import EarlyStopping
import time
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
# from sklearn.metrics import r2_score

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy import stats
from keras.utils import to_categorical
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.stattools import pacf
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
# import seaborn as sns
rcParams['figure.figsize']=15,5


# In[88]:


from time import time

import numpy as np
import pandas as pd


# For plotting
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import plotly.graph_objects as go

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,7)})

#For standardising the dat
from sklearn.preprocessing import StandardScaler

#PCA
from sklearn.manifold import TSNE

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[ ]:





# In[ ]:





# In[89]:


Stations = ['Batseri kinnaur','ddharmshalakangara','ghoda_farm3_mandi','ghoda_farm5_mandi',
            'griffon peak_2','griffon peak5 mandi','kuppa_data','nigulasridata','pagalnala_data',
            'purbani_kinnaur','sanarli_1_mandi','sanarli_3_mandi','sandhol kangra','Tattapani Mandi',
            'urni_dhank_kinnaur']


# In[90]:


Stations[0]


# In[91]:


Column = ['Date','Tem','Hum','Pressure','Rain','Light','Ax','Ay','Az','Wx','Wy','Wz','Moisture','Count']


# In[100]:


#Rearrange the Array
def makeArray(Array):
    New=np.array(Array[0])

    for i in range(1,len(Array)):
        New = np.append(New,Array[i],axis=0)
        
    return New


# In[101]:


def readData(Stations):
    
    Data, C = [], []
    
    for i in range(len(Stations)):
        file = Stations[i]+'.csv'
        newfile = 'clean_'+file
        df = pd.read_csv('Clean/'+newfile, header=0, index_col=None)
        print(newfile)
        df = df.reset_index(drop=True)
        data=df[['Tem','Hum','Pressure','Rain','Light','Ax','Ay','Az','Wx','Wy','Wz','Moisture']].values
        count=df['Count'].values

        #Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(data)
         
        Data.append(dataset)
        C.append(count)
       
    #Train Test split
    len_test1=int(len(Data)*0.2)
    train_data = Data[:-len_test1]
    test_data = Data[-len_test1:]
    train_count = C[:-len_test1]
    test_count = C[-len_test1:]
    
    
        
    return (makeArray(train_data), makeArray(test_data), makeArray(train_count), makeArray(test_count))


# In[ ]:





# In[102]:


(trainX,testX, trainCount, testCount) = readData(Stations)


# In[103]:


df = pd.read_csv('Clean/clean_ghoda_farm5_mandi.csv', header=0, index_col=None)

df = df.reset_index(drop=True)
data=df[['Ax','Ay','Az','Wx','Wy','Wz']].values
count=df['Count'].values


# In[104]:


df


# In[105]:


Ax=data[:,0]


# In[ ]:





# In[108]:


plt.plot(Ax)
plt.plot(count)


# In[107]:


Ax


# In[ ]:





# In[ ]:





# In[ ]:





# In[308]:


trainX.shape


# In[309]:


trainCount.shape


# In[ ]:


input_lyr = Input(shape=(12,))
encoded1 = Dense(10, activation='relu')(input_lyr)
encoded2 = Dense(5, activation='relu')(encoded1)
encoded3 = Dense(10, activation='relu')(encoded2)
decoded = Dense(12)(encoded3)
autoencoder = Model(input_lyr, decoded)

# This model maps an input to its encoded representation
encoder = Model(input_lyr, encoded2)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()


# In[310]:


callback=keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1.0e-4, 
                                          patience=10, verbose=0, mode="auto", baseline=None, restore_best_weights=True)
history = autoencoder.fit(trainX, trainX, epochs=500, batch_size=512, 
                          validation_data=(testX, testX), verbose=1, callbacks=[callback])


# In[111]:


CombinedData = np.append(trainX,testX,axis=0)
CountData = np.append(trainCount,testCount)


# In[112]:


CombinedData.shape


# In[113]:


encoded_data = encoder.predict(CombinedData)


# In[114]:


C = autoencoder.predict(CombinedData)


# In[315]:


C.shape


# In[316]:


# plt.plot(CombinedData[:100,0])
plt.plot(encoded_data[:,:])


# In[110]:


CombinedData[:100,5]


# In[116]:


encoded_data.shape


# In[42]:


CountData=CountData.reshape((CountData.shape[0],1))


# In[43]:


CountData.shape


# In[ ]:





# In[47]:


bestFeature = np.hstack((encoded_data,CountData))


# In[49]:


bestFeature.shape


# In[ ]:





# In[115]:


CombinedData.shape


# In[ ]:




