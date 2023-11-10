#!/usr/bin/env python
# coding: utf-8

# In[1]:


seed=198


# In[2]:


# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import random
import pickle
import torch.nn.init as init
from torchviz import make_dot
# import seaborn as sns
# import torchtext
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import display
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
warnings.simplefilter("ignore")
print(torch.__version__)


# In[3]:


# Check if CUDA is available
if torch.cuda.is_available():
    # CUDA is available, so PyTorch can use a GPU
    device = torch.device("cuda")
    print("PyTorch is using GPU.")
else:
    # CUDA is not available, PyTorch will use CPU
    device = torch.device("cpu")
    print("PyTorch is using CPU.")

# Create a tensor and move it to the device
tensor = torch.randn(3, 4).to(device)

# Check the device of the tensor
print("Tensor device:", tensor.device)


# In[4]:


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
#     tf.random.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# In[5]:


scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()


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
    
    print(Stations)
    df = pd.read_csv(Stations, header=0, index_col=None)
    df = df.reset_index(drop=True)
    df = df[['Tem','Hum','Pressure','Rain','Light','Ax','Ay','Az','Wx','Wy','Wz','Moisture','Movements']]
    df.dropna(inplace=True)
    data=df.values
   

    Data.append(data)
           
    
        
    return makeArray(Data)


# In[8]:


Stations=os.listdir('Data')


# In[9]:


Stations.sort()


# In[10]:


Data=[[] for x in range(len(Stations))]
for i in range(len(Stations)):
    Data[i] = readData('Data/'+Stations[i])


# In[14]:


L_shape=[]
for i in range(len(Data)):
    L_shape.append(Data[i].shape[0])

D = np.concatenate(Data,axis=0)
D = scaler.fit_transform(D)
s=0
Normalize = []
for i in range(len(L_shape)):
    Normalize.append(D[s:s+L_shape[i]])
    s=L_shape[i]
    


# In[17]:


D.shape


# In[21]:


R=np.zeros((1,13))


# In[33]:


R[0]=0.7


# In[34]:


R


# In[35]:


scaler.inverse_transform(R)


# In[12]:


def packet(data,seq):
    D,L1,L2,L3 = [], [], [], []
    
    for i in range(len(data)-seq-5):
        D.append(data[i:i+seq])
#         L1.append(data[i+1:i+seq+1])
#         L2.append(data[i+3:i+seq+3])
        L3.append(data[i+6:i+seq+6])
        
    return np.array(D), np.array(L3)#, np.array(L2), np.array(L3)


# In[13]:


seq_len=144


# In[14]:


Total, One, Half, Hour = [], [], [], []
for i in range(len(Data)):
    D, L1=packet(Normalize[i],seq_len)
    Total.append(D)
    One.append(L1)
#     Half.append(L2)
#     Hour.append(L3)
    


# In[15]:


Train = np.concatenate(Total[:50],axis=0)
Test = np.concatenate(Total[50:],axis=0)
TrainL1 = np.concatenate(One[:50],axis=0)
TestL1 = np.concatenate(One[50:],axis=0)
# TrainL2 = np.concatenate(Half[:50],axis=0)
# TestL2 = np.concatenate(Half[50:],axis=0)
# TrainL3 = np.concatenate(Hour[:50],axis=0)
# TestL3 = np.concatenate(Hour[50:],axis=0)


# In[22]:


Train.shape, Test.shape


# In[30]:


np.where(TrainL1[:,-1,-1]==0)[0].shape


# In[49]:


# Sample data
import matplotlib.pyplot as plt
categories = ['No-Movement', 'Small Movement', 'Large Movement']
values = [1030997, 27236, 7776]

# Create a column diagram
bars=plt.bar(categories, values, color=['blue','darkorange','green'])
plt.xlabel('Movement Class')
plt.ylabel('Samples')
plt.title('Movement Class Distribution')

# Rotate x-axis labels for better readability (optional)
# plt.xticks(rotation=45)
plt.ticklabel_format(style='plain', axis='y')
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
             ha='center', va='bottom')
# Show the plot
plt.savefig('column_diagram.png')
plt.show()


# In[ ]:





# In[18]:


T = []
TL1 = []
TL2 = []
TL3 = []
I = []
a,b,c=0,0,0
cc=0
for i in range(len(Train)):
    a = TrainL1[i][0][0]
#     b = TrainL2[i][0][0]
#     c = TrainL3[i][0][0]
    
    if (a+b+c)!=0:
        cc=cc+1
        T.append(Train[i])
        TL1.append(TrainL1[i])
#         TL2.append(TrainL2[i])
#         TL3.append(TrainL3[i])
    else:
        I.append(i)

random.shuffle(I)
for i in I[:cc]:
    T.append(Train[i])
    TL1.append(TrainL1[i])
#     TL2.append(TrainL2[i])
#     TL3.append(TrainL3[i])

TrainData = np.array(T)
TrainLabel1 = np.array(TL1)
# TrainLabel2 = np.array(TL2)
# TrainLabel3 = np.array(TL3)

T = []
TL1 = []
TL2 = []
TL3 = []
I = []
cc=0
for i in range(len(Test)):
    a = TestL1[i][0][0]
#     b = TestL2[i][0][0]
#     c = TestL3[i][0][0]
    
    if (a+b+c)!=0:
        cc=cc+1
        T.append(Test[i])
        TL1.append(TestL1[i])
#         TL2.append(TestL2[i])
#         TL3.append(TestL3[i])
    else:
        I.append(i)

random.shuffle(I)
for i in I[:cc]:
    T.append(Test[i])
    TL1.append(TestL1[i])
#     TL2.append(TestL2[i])
#     TL3.append(TestL3[i])

TestData = np.array(T)
TestLabel1 = np.array(TL1)
# TestLabel2 = np.array(TL2)
# TestLabel3 = np.array(TL3)


# In[21]:


TrainData.shape


# In[20]:


TestLabel1.shape


# In[45]:


# Assuming you have TrainData, TrainLabel1, TrainLabel2, TrainLabel3, TestData, TestLabel1, TestLabel2, TestLabel3 defined

# Create a dictionary to store the data and labels
data_dict = {
    "TrainData": TrainData,
    "TrainLabel3": TrainLabel1,
#     "TrainLabel2": TrainLabel2,
#     "TrainLabel3": TrainLabel3,
    "TestData": TestData,
    "TestLabel3": TestLabel1,
#     "TestLabel2": TestLabel2,
#     "TestLabel3": TestLabel3
}

# Specify the file path to save the pickle file
pickle_file_path = "data60.pickle"

# Save the data dictionary to a pickle file
with open(pickle_file_path, "wb") as f:
    pickle.dump(data_dict, f)

print("Data saved to", pickle_file_path)


# In[ ]:




