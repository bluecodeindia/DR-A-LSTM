#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
# warnings.filterwarnings('once')
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# In[2]:


import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense,Input,Reshape, Flatten
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
import sklearn
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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[3]:


# seed=1226 = 0.98
seed=1226


# In[4]:


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


# In[5]:


Stations = ['Batseri kinnaur','Gharpa','ghoda_farm3_mandi','griffon peak 2','griffon_peak3',
            'kuppa_data','nigulasridata','pagalnala_data','purbani_kinnaur','sanarli_1_mandi','sanarli_3_mandi',
            'sandhol kangra','urni_dhank_kinnaur','griffon peak5 mandi']


# In[6]:


Stations = ['Batseri kinnaur','Gharpa','ghoda_farm3_mandi','griffon peak 2','griffon_peak3',
            'kuppa_data','nigulasridata','pagalnala_data','purbani_kinnaur','sanarli_1_mandi','sanarli_3_mandi',
            'sandhol kangra','urni_dhank_kinnaur','griffon peak5 mandi']


# In[7]:


Column = ['Date','Tem','Hum','Pressure','Rain','Light','Ax','Ay','Az','Wx','Wy','Wz','Moisture','Count']


# In[8]:


#Rearrange the Array
def makeArray(Array):
    New=np.array(Array[0])

    for i in range(1,len(Array)):
        New = np.append(New,Array[i],axis=0)
        
    return New


# In[9]:


def readData1(Stations):
    
    Data, C = [], []
    
    print(Stations)
    file = Stations+'.csv'
    newfile = 'clean_'+file
    df = pd.read_csv('Clean-dataset-LMS/'+newfile, header=0, index_col=None)
    print(newfile)
    df = df.reset_index(drop=True)
    data=df[['Tem','Hum','Pressure','Rain','Light','Ax','Ay','Az','Wx','Wy','Wz','Moisture','Count']].values
    data=data.astype('float32')
    count=df['Count'].values
    count=count.astype('float32')

    #Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    Data.append(data)
    C.append(count)
           
    
        
    return makeArray(Data), makeArray(C)


# In[10]:


def readData(Stations):
    
    Data, C = [], []
    
    for i in range(len(Stations)):
        file = Stations[i]+'.csv'
        newfile = 'clean_'+file
        df = pd.read_csv('Clean-dataset-LMS/'+newfile, header=0, index_col=None)
        print(newfile)
        df = df.reset_index(drop=True)
        data=df[['Tem','Hum','Pressure','Rain','Light','Ax','Ay','Az','Wx','Wy','Wz','Moisture','Count']].values
        data=data.astype('float32')
        count=df['Count'].values
        count=count.astype('float32')

        #Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
         
        Data.append(data)
        C.append(count)
           
    
        
    return makeArray(Data), makeArray(C)


# In[11]:


DataOne, CountOne = readData(Stations)


# In[12]:


Data=[[] for x in range(len(Stations))]
Count=[[] for x in range(len(Stations))]
for i in range(len(Stations)):
    Data[i], Count[i] = readData1(Stations[i])


# In[13]:


Data[0].shape


# In[14]:


def getData(data, C, lag,z):
    DataX, DataY = [], []
    count=0
    for i in range(lag,len(data)):
        if C[i]>0:
            count+=1
            DataX.append(data[i-lag:i,:])
            
    
    countnew=0
    idx=lag
    np.random.seed(z)
    per = np.random.permutation(len(data))
    for i in per:

        if C[i]==0:
            if sum(C[lag-i:i])==0:
                countnew+=1
                DataY.append(data[i-lag:i,:])
                
        if countnew==count:
            break
    DataX=np.array(DataX).astype(np.float32)   
    DataY=np.array(DataY).astype(np.float32)
    print(DataX.shape,DataY.shape)
    return DataX, DataY
    


# In[ ]:





# In[15]:


def makeData(Data,Count,lag):
    
    #Make the packets by lag value
    f = True
    g=0
    h=0
    y=0
    l=len(Count)
    R=[1998,7161,8200,1461,7168,5312,2202,8668,9335,6188,6934,6314,4156,5377]
    while(f and g<l):
        try:
            DataX, DataY = getData(Data, Count, lag, R[h])
            f=False
        except:
            g+=1
            h+=1
            h%=14
            x=int(g/l*100)
            if x-y!=0:
                print(str(x)+', ',end='')
            y=x

#     print(len(DataX),len(DataY))
    
    np.random.seed(seed)
    per = np.random.permutation(len(DataX))
    Mov = DataX[:]
    NonMov = DataY[:]
    
    
    #Creating labels
    A=np.ones(len(Mov))
    B=np.zeros(len(NonMov))
    
       

    #Creating One-Hot-Encoding
    C=np.hstack((A,B))
    D=to_categorical(C, dtype ="uint8")
    E=D[:len(Mov)]
    F=D[len(Mov):]

    
    
        
    return (Mov, E), (NonMov, F)


# In[16]:


def comp(Train):
    T=[]
    for x in Train:
        for y in x:
            T.append(y)
    return np.array(T)


# In[71]:


def anotherLevel(encoded_Data,Count,lag):
    Mov=[]
    Mov_L=[]
    NoMov= []
    NoMov_L= []

    for i in range(len(Stations)):
        if sum(Count[i][lag:])>0:
            (A, B),(C, D) = makeData(encoded_Data[i], Count[i],lag)
            Mov.append(A)
            NoMov.append(C)
            Mov_L.append(B)
            NoMov_L.append(D)

    total = len(Mov)

    Train =[]
    Test =[]
    TrainL =[]
    TestL =[]
    Movement=[]
    NoMovement=[]
    MovementL=[]
    NoMovementL=[]

    for i in range(total):

        MM = Mov[i][:]
        NN = NoMov[i][:]
        OO = Mov_L[i][:]
        PP = NoMov_L[i][:]
        Movement.append(MM)
        NoMovement.append(NN)
        MovementL.append(OO)
        NoMovementL.append(PP)

    
    Movement=comp(Movement) 
    NoMovement=comp(NoMovement)
    MovementL=comp(MovementL)
    NoMovementL=comp(NoMovementL)
    
    t=int(len(Movement)*0.2)
    np.random.seed(10)
    per=np.random.permutation(len(Movement))
    per=per[:t]
    for i in range(len(Movement)):
        if i in per:
            Test.append(Movement[i])
            Test.append(NoMovement[i])
            TestL.append(MovementL[i])
            TestL.append(NoMovementL[i])
        else:
            Train.append(Movement[i])
            Train.append(NoMovement[i])
            TrainL.append(MovementL[i])
            TrainL.append(NoMovementL[i])
    
    Train=np.array(Train)
    TrainL=np.array(TrainL)
    Test=np.array(Test)
    TestL=np.array(TestL)
    return (Train,TrainL),(Test,TestL)

        


# In[72]:


D=DataOne.copy()
np.random.seed(seed)
per = np.random.permutation(len(D))
D=D[per]
t=int(len(D)*.2)
trainX=D[:-t]
testX=D[-t:]
trainX.shape,testX.shape


# In[ ]:





# In[73]:


maxvalue=max(np.max(trainX),np.max(testX))
minvalue=min(np.min(trainX),np.min(testX))
trainX = (trainX-minvalue)/(maxvalue-minvalue)
testX = (testX-minvalue)/(maxvalue-minvalue)
maxvalue,minvalue


# In[74]:


lag=20


# In[75]:


p=0.3
b=2000
act='relu'
tf.random.set_seed(seed)
np.random.seed(seed)
input_lyr = Input(shape=(13*lag,))
E=Dense(300, activation=act)(input_lyr)
E=Dense(500, activation=act)(E)
E=Dense(1000, activation=act)(E)


Z = Dense(b, activation=act)(E)



output_lyr = Dense(2,activation='softmax')(Z)

mlp = Model(input_lyr, output_lyr)

# This model maps an input to its encoded representation
encoder = Model(input_lyr, Z)

mlp.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
# autoencoder.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:


tf.random.set_seed(seed)
np.random.seed(seed)
model = Sequential()
model.add(Input(shape=(lag, 100)))
# model.add(LSTM(500,activation='tanh',return_sequences=True))
model.add(LSTM(2000,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

# model.summary()


# In[77]:


tf.random.set_seed(seed)
np.random.seed(seed)
model2 = Sequential()
model2.add(Input(shape=(lag, 13)))
# model.add(LSTM(500,activation='tanh',return_sequences=True))
model2.add(LSTM(100,activation='tanh'))
model2.add(Dropout(0.2))
model2.add(Dense(50,activation='relu'))
model2.add(Dense(2,activation='softmax'))
model2.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

# model2.summary()


# In[78]:


Data[i].shape


# In[79]:


(TrainOne,TrainOneL),(TestOne,TestOneL) = anotherLevel(Data,Count,lag)


# In[80]:


maxvalue=max(np.max(TrainOne),np.max(TestOne))
minvalue=min(np.min(TrainOne),np.min(TestOne))
TrainOne = (TrainOne-minvalue)/(maxvalue-minvalue)
TestOne = (TestOne-minvalue)/(maxvalue-minvalue)
maxvalue,minvalue


# In[81]:


TrainOne=TrainOne.reshape((TrainOne.shape[0],TrainOne.shape[1]*TrainOne.shape[2]))
TestOne=TestOne.reshape((TestOne.shape[0],TestOne.shape[1]*TestOne.shape[2]))


# In[82]:


TrainOne.shape


# In[83]:



callback=keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1.0e-4, 
                                          patience=500, verbose=0, mode="auto", baseline=None, restore_best_weights=True)
tf.random.set_seed(seed)
np.random.seed(seed)
history = mlp.fit(TrainOne, TrainOneL, epochs=500, batch_size=2048, 
                          validation_data=(TestOne, TestOneL), verbose=1, callbacks=[callback])


# In[ ]:





# In[84]:


Train=encoder.predict(TrainOne)
Test=encoder.predict(TestOne)


# In[85]:


Train.shape


# In[86]:


Train=Train.reshape((Train.shape[0],20,-1))
Test=Test.reshape((Test.shape[0],20,-1))


# In[87]:


Train.shape


# In[88]:


maxvalue=max(np.max(Train),np.max(Test))
minvalue=min(np.min(Train),np.min(Test))
Train = (Train-minvalue)/(maxvalue-minvalue)
Test = (Test-minvalue)/(maxvalue-minvalue)
maxvalue,minvalue


# In[ ]:





# In[89]:


tf.random.set_seed(seed)
np.random.seed(seed)
H = model.fit(Train, TrainOneL, epochs=500,batch_size=512, validation_data=(Test, TestOneL),callbacks=[EarlyStopping(monitor='val_accuracy', patience=50,restore_best_weights=True)],verbose=1, shuffle=True)


# In[90]:


model.evaluate(Train, TrainOneL)


# In[91]:


model.evaluate(Test, TestOneL)


# In[92]:



x_subset = Train[:]
y_subset = [np.argmax(x) for x in TrainOneL[:]]

print(np.unique(y_subset))
x_subset=x_subset.reshape((x_subset.shape[0],x_subset.shape[1]*x_subset.shape[2]))
get_ipython().run_line_magic('time', '')
tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300).fit_transform(x_subset)
plt.scatter(tsne[:, 0], tsne[:, 1], s= 50, c=y_subset, cmap='Spectral',edgecolors='black')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(2)-0.5).set_ticks(np.arange(2))
plt.title('Visualizing through t-SNE', fontsize=24);


# In[93]:


TrainOne.shape


# In[ ]:





# In[94]:



x_subset = Test[:]
y_subset = [np.argmax(x) for x in TestOneL[:]]

print(np.unique(y_subset))
x_subset=x_subset.reshape((x_subset.shape[0],x_subset.shape[1]*x_subset.shape[2]))
get_ipython().run_line_magic('time', '')
tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300).fit_transform(x_subset)
plt.scatter(tsne[:, 0], tsne[:, 1], s= 50, c=y_subset, cmap='Spectral',edgecolors='black')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(2)-0.5).set_ticks(np.arange(2))
plt.title('Visualizing through t-SNE', fontsize=24);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[95]:


f, axes = plt.subplots(1, 2, figsize=(15, 5), sharey='row')
Predict=[]
True_cls=[]
test=Train
y=TrainOneL

P=model.predict(test,verbose=0)
Predict=np.argmax(P,axis=1)
True_cls=np.argmax(y,axis=1)

cm = confusion_matrix(True_cls, Predict, labels=[0,1])
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No Mov','Mov'])
disp1.plot(ax=axes[0])
disp1.im_.colorbar.remove()

Predict2=[]
True_cls2=[]
test=Test
y=TestOneL
P=model.predict(test,verbose=0)
Predict2=np.argmax(P,axis=1)
True_cls2=np.argmax(y,axis=1)

cm = confusion_matrix(True_cls2, Predict2, labels=[0,1])
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No Mov','Mov'])
disp2.plot(ax=axes[1])
disp2.im_.colorbar.remove()



plt.subplots_adjust(wspace=0.40, hspace=0.1)

plt.show()


# In[43]:


# f, axes = plt.subplots(1, 2, figsize=(15, 5), sharey='row')


# Predict=[]
# True_cls=[]
# test=TrainOne
# y=TrainOneL
# for i in range (len(test)):
#     P=model2.predict(np.expand_dims(test[i], axis=0),verbose=0)
#     Predict.append(np.argmax(P))
#     True_cls.append(np.argmax(y[i]))

# cm = confusion_matrix(True_cls, Predict, labels=[0,1])
# disp1 = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No Mov','Mov'])
# disp1.plot(ax=axes[0])
# disp1.im_.colorbar.remove()

# Predict=[]
# True_cls=[]
# test=TestOne
# y=TestOneL
# for i in range (len(test)):
#     P=model2.predict(np.expand_dims(test[i], axis=0),verbose=0)
#     Predict.append(np.argmax(P))
#     True_cls.append(np.argmax(y[i]))

# cm = confusion_matrix(True_cls, Predict, labels=[0,1])
# disp2 = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No Mov','Mov'])
# disp2.plot(ax=axes[1])
# disp2.im_.colorbar.remove()

# plt.subplots_adjust(wspace=0.40, hspace=0.1)


# plt.show()


# In[ ]:




