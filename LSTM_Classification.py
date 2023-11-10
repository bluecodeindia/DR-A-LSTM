#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
# warnings.filterwarnings('once')
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# In[52]:



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


# In[3]:


Stations = ['Batseri kinnaur','ddharmshalakangara','ghoda_farm3_mandi','ghoda_farm5_mandi',
            'griffon peak_2','griffon peak5 mandi','kuppa_data','nigulasridata','pagalnala_data',
            'purbani_kinnaur','sanarli_1_mandi','sanarli_3_mandi','sandhol kangra','Tattapani Mandi',
            'urni_dhank_kinnaur']


# In[4]:


Column = ['Date','Tem','Hum','Pressure','Rain','Light','Ax','Ay','Az','Wx','Wy','Wz','Moisture','Count']


# In[ ]:





# In[30]:


def getData(data, lag):
    DataX, DataY = [], []
    count=0
    for i in range(lag,len(data)):
        if data[i,-1]>0:
            count+=1
            DataX.append(data[i-lag+1:i+1,:-1])
            
    
    hipcount=0
    countnew=0
    idx=lag
    for i in range(lag,len(data)-lag):
        if idx>len(data)-1:
            continue
        if data[idx,-1]==0:
            hipcount+=1
            if hipcount==lag:
                countnew+=1
                hipcount=0
                DataY.append(data[idx-lag+1:idx+1,:-1])
                idx+=1
        else:
            hipcount=0
            
        if countnew==count:
            break
        
        idx+=1
    
    return np.array(DataX).astype(np.float32), np.array(DataY).astype(np.float32)
    


# In[31]:


#Rearrange the Array
def makeArray(Array):
    New=np.array(Array[0])

    for i in range(1,len(Array)):
        New = np.append(New,Array[i],axis=0)
        
    return New
    


# In[32]:


Column = ['Date','Tem','Hum','Pressure','Rain','Light','Ax','Ay','Az','Wx','Wy','Wz','Moisture','Count']


# In[33]:


def makeData(Stations,lag):
    
    Mov, Nonmov = [], []
    
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
        datasetnew = np.append(dataset,count.reshape((count.shape[0],1)),axis=1)

        #Make the packets by lag value
        DataX, DataY = getData(datasetnew, lag)

        print(len(DataX),len(DataY))
        
        if len(DataX)==0:
            continue
            

        Mov.append(DataX)
        Nonmov.append(DataY)
    

    trainX = makeArray(Mov)
    trainY = makeArray(Nonmov)
    print(trainX.shape)
    print(trainY.shape)
    
    #Shuffle Packets
    perx=np.random.permutation(len(trainX))
    pery=np.random.permutation(len(trainY))
    
    trainX = trainX[perx]
    trainY = trainY[pery]
    
    #Train Test split
    len_test1=int(len(trainX)*0.2)
    len_test2=int(len(trainY)*0.2)
    MovTrain = trainX[:-len_test1]
    MovTest = trainX[-len_test1:]
    NonMovTrain = trainY[:-len_test2]
    NonMovTest = trainY[-len_test2:]
    
    train_data=np.append(MovTrain,NonMovTrain,axis=0)
    test_data=np.append(MovTest,NonMovTest,axis=0)
    
    #Creating labels
    A=np.ones(len(MovTrain))
    B=np.zeros(len(NonMovTrain))
    trainLabel=np.append(A,B)
    A=np.ones(len(MovTest))
    B=np.zeros(len(NonMovTest))
    testLabel=np.append(A,B)

    #Creating One-Hot-Encoding
    train_label=to_categorical(trainLabel, dtype ="uint8")
    test_label =to_categorical(testLabel, dtype ="uint8")
    
    
    if(len(train_label)!=len(train_data) or len(test_label)!=len(test_data)):
        print('Wrong------------')
    
        
    return (train_data, train_label), (test_data, test_label)


# In[34]:


lag=10


# In[35]:



(train_data, train_label), (test_data, test_label) = makeData(Stations,lag)


# In[36]:


train_data.shape


# In[37]:


s=0


# In[38]:


#Plot data for movement
A=[]
for i in range(len(train_data)):
    if np.argmax(train_label[i])==1:
        A.append(train_data[i,:,s])
        
plt.plot(np.array(A))


# In[39]:


#Plot data for non movement
A=[]
for i in range(len(train_data)):
    if np.argmax(train_label[i])==0:
        A.append(train_data[i,:,s])
        
plt.plot(np.array(A))


# In[40]:


INIT_LR = 1e-4
NUM_EPOCHS = 500
BATCH_SIZE = 32


# In[ ]:





# In[41]:


train_data.shape


# In[42]:



model = Sequential()
model.add(Input(shape=(train_data.shape[1], train_data.shape[2])))
# model.add(LSTM(500,activation='tanh',return_sequences=True))
model.add(LSTM(100,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='relu'))
model.add(Dense(2,activation='softmax'))
opt = Adam(lr=INIT_LR)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()


# In[43]:


# H = model.fit(trainX, trainY, epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,verbose=1, shuffle=True)


# In[44]:


train_label.shape


# In[45]:



H = model.fit(train_data, train_label, epochs=NUM_EPOCHS,batch_size=BATCH_SIZE, validation_data=(test_data, test_label),callbacks=[EarlyStopping(monitor='val_loss', patience=100,restore_best_weights=True)],verbose=1, shuffle=True)


# In[46]:


model.evaluate(test_data,test_label)


# In[47]:


Predict=[]
True_cls=[]
test=test_data
y=test_label
for i in range (len(test)):
    P=model.predict(np.expand_dims(test[i], axis=0),verbose=0)
    Predict.append(np.argmax(P))
    True_cls.append(np.argmax(y[i]))


# In[48]:


print(len(train_data),len(test_data))


# In[49]:


1-np.sum(abs(np.array(Predict)-np.array(True_cls)))/len(Predict)


# In[50]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 
cm = confusion_matrix(True_cls, Predict, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No Mov','Mov'])
disp.plot()
plt.show()


# In[53]:


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


# In[56]:


trainX=train_data
testX=test_data


# In[57]:


callback=keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1.0e-4, 
                                          patience=10, verbose=0, mode="auto", baseline=None, restore_best_weights=True)
history = autoencoder.fit(trainX, trainX, epochs=500, batch_size=512, 
                          validation_data=(testX, testX), verbose=1, callbacks=[callback])


# In[ ]:





# In[ ]:




