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


# In[3]:


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


# In[4]:


Stations = ['Batseri kinnaur','ddharmshalakangara','ghoda_farm3_mandi','ghoda_farm5_mandi',
            'griffon peak_2','griffon peak5 mandi','kuppa_data','nigulasridata','pagalnala_data',
            'purbani_kinnaur','sanarli_1_mandi','sanarli_3_mandi','sandhol kangra','Tattapani Mandi',
            'urni_dhank_kinnaur']


# In[5]:


Column = ['Date','Tem','Hum','Pressure','Rain','Light','Ax','Ay','Az','Wx','Wy','Wz','Moisture','Count']


# In[6]:


#Rearrange the Array
def makeArray(Array):
    New=np.array(Array[0])

    for i in range(1,len(Array)):
        New = np.append(New,Array[i],axis=0)
        
    return New


# In[7]:


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
#         data = scaler.fit_transform(data)
         
        Data.append(data)
        C.append(count)
       
    #Train Test split
    len_test1=int(len(Data)*0.2)
    train_data = Data[:-len_test1]
    test_data = Data[-len_test1:]
    train_count = C[:-len_test1]
    test_count = C[-len_test1:]
    
    
        
    return (makeArray(train_data), makeArray(test_data), makeArray(train_count), makeArray(test_count))


# In[8]:


(trainX,testX, trainCount, testCount) = readData(Stations)


# In[9]:


trainX.shape


# In[10]:


trainCount.shape


# In[11]:


maxvalue=max(np.max(trainX),np.max(testX))
minvalue=min(np.min(testX),np.min(testX))
trainX = (trainX-minvalue)/(maxvalue-minvalue)
testX = (testX-minvalue)/(maxvalue-minvalue)


# In[12]:


DataX = np.vstack((trainX,testX))


# In[13]:


CountData = np.append(trainCount,testCount)


# In[14]:


trainX.shape


# In[15]:


DataX.shape


# In[16]:


from sklearn.decomposition import PCA
pca = PCA(n_components=6)
principalComponents = pca.fit_transform(DataX)
principalDf = pd.DataFrame(data = principalComponents)


# In[17]:


principalDf


# In[18]:


principalComponents.shape


# In[19]:


CountData=CountData.reshape((CountData.shape[0],1))


# In[20]:


CountData.shape


# In[ ]:





# In[21]:


bestFeature = np.hstack((principalComponents,CountData))


# In[22]:


bestFeature.shape


# In[23]:


def getData(data, lag):
    DataX, DataY = [], []
    count=0
    for i in range(lag,len(data)):
        if data[i,-1]>0:
            count+=1
            DataX.append(data[i-lag:i,:-1])
            
    
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
                DataY.append(data[idx-lag:idx,:-1])
                idx+=1
        else:
            hipcount=0
            
        if countnew==count:
            break
        
        idx+=1
    
    return np.array(DataX).astype(np.float32), np.array(DataY).astype(np.float32)
    


# In[24]:


def getData1(data, lag):
    DataX, DataY = [], []
    count=0
    for i in range(lag,len(data)):
        if data[i,-1]>0:
            count+=1
            DataX.append(data[i-lag:i,:])
            
    
    countnew=0
    idx=lag
    per = np.random.permutation(len(data))
    for i in per:

        if data[i,-1]==0:
            
            countnew+=1
            hipcount=0
            DataY.append(data[i-lag:i,:])
                
        if countnew==count:
            break
        
    
    return np.array(DataX).astype(np.float32), np.array(DataY).astype(np.float32)
    
    


# In[25]:


def makeData(Data,lag):
    
    
    
    #Make the packets by lag value
    DataX, DataY = getData(Data, lag)

    print(len(DataX),len(DataY))
    
    
    #Train Test split
    per1 = np.random.permutation(len(DataX))
    per2 = np.random.permutation(len(DataY))
    DataX= DataX[per1]
    DataY= DataY[per2]
    len_test1=int(len(DataX)*0.2)
    len_test2=int(len(DataY)*0.2)
    MovTrain = DataX[:-len_test1]
    MovTest = DataX[-len_test1:]
    NonMovTrain = DataY[:-len_test2]
    NonMovTest = DataY[-len_test2:]
    
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


# In[26]:


lag=12


# In[27]:



(train_data, train_label), (test_data, test_label) = makeData(bestFeature,lag)


# In[ ]:





# In[28]:


x_subset = train_data[0:1000]
y_subset = [np.argmax(x) for x in train_label][0:1000]

print(np.unique(y_subset))


# In[29]:


x_subset=x_subset.reshape((x_subset.shape[0],x_subset.shape[1]*x_subset.shape[2]))


# In[30]:


get_ipython().run_line_magic('time', '')
tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300).fit_transform(x_subset)


# In[ ]:





# In[31]:


plt.scatter(tsne[:, 0], tsne[:, 1], s= 50, c=y_subset, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(2)-0.5).set_ticks(np.arange(2))
plt.title('Visualizing through t-SNE', fontsize=24);


# In[32]:


INIT_LR = 1e-4
NUM_EPOCHS = 500
BATCH_SIZE = 32


# In[33]:


train_data.shape


# In[34]:



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


# In[35]:


# H = model.fit(trainX, trainY, epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,verbose=1, shuffle=True)


# In[36]:


train_data.shape


# In[37]:



H = model.fit(train_data, train_label, epochs=NUM_EPOCHS,batch_size=BATCH_SIZE, validation_data=(test_data, test_label),callbacks=[EarlyStopping(monitor='val_loss', patience=100,restore_best_weights=True)],verbose=1, shuffle=True)


# In[38]:


model.evaluate(train_data,train_label)


# In[39]:


model.evaluate(test_data,test_label)


# In[40]:


# import tensorflow as tf
# from tensorflow import keras 
# model.save('weights2-outputlstm',save_format='.h5')
# model.save_weights("weights2-outputlstm/model.h5")


# In[ ]:





# In[41]:


Predict=[]
True_cls=[]
test=test_data
y=test_label
for i in range (len(test)):
    P=model.predict(np.expand_dims(test[i], axis=0),verbose=0)
    Predict.append(np.argmax(P))
    True_cls.append(np.argmax(y[i]))


# In[42]:


print(len(train_data),len(test_data))


# In[43]:


1-np.sum(abs(np.array(Predict)-np.array(True_cls)))/len(Predict)


# In[44]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 
cm = confusion_matrix(True_cls, Predict, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No Mov','Mov'])
disp.plot()
plt.show()


# In[45]:


5,0.7941176470588236


# In[46]:


6,0.7647,0.8823


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




