#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

#For standardising the dat
from sklearn.preprocessing import StandardScaler

#PCA
from sklearn.manifold import TSNE

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[95]:


Stations = ['Batseri kinnaur','ddharmshalakangara','ghoda_farm3_mandi','ghoda_farm5_mandi','griffon peak_2','griffon peak5 mandi','kuppa_data','nigulasridata','pagalnala_data','purbani_kinnaur','sanarli_1_mandi','sanarli_3_mandi','sandhol kangra','Tattapani Mandi','urni_dhank_kinnaur']


# In[ ]:


file = Stations[14]+'.csv'
newfile = 'clean_'+file


# In[21]:


df = pd.read_csv('data.csv', header=0, index_col=0)
print('Number of rows and columns:', df.shape)
df = df.reset_index(drop=True)
df.head()

df.dtypes


# In[22]:


data=df.values[:,1:13]
count=df.values[:,13]
cumcount=df.values[:,14]


# In[23]:


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(data)
# datasetnew = np.append(dataset,count.reshape((count.shape[0],1)),axis=1)


# In[76]:


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
        if data[idx,-1]==0:
            hipcount+=1
            if hipcount==lag:
                countnew+=1
                hipcount=0
                DataY.append(data[idx-lag+1:idx+1,:-1])
                idx+=lag
        else:
            hipcount=0
            
        if countnew==count*5:
            break
        
        idx+=1
    
    return np.array(DataX).astype(np.float32), np.array(DataY).astype(np.float32)
    


# In[77]:


DataX, DataY = getData(datasetnew,5)


# In[78]:


print(len(DataX),len(DataY))


# In[79]:


A=np.ones(len(DataX))
B=np.zeros(len(DataY))
y_train=np.append(A,B)


# In[80]:


x_train=np.append(DataX,DataY,axis=0)


# In[81]:


x_train.shape


# In[82]:


y_train


# In[83]:


x = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
y = y_train


# In[84]:


x.shape


# In[85]:


## Standardizing the data
standardized_data = StandardScaler().fit_transform(x)
print(standardized_data.shape)


# In[86]:


# t-SNE is consumes a lot of memory so we shall use only a subset of our dataset. 

x_subset = x[0:60]
y_subset = y[0:60]

print(np.unique(y_subset))


# In[87]:


get_ipython().run_line_magic('time', '')
tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300).fit_transform(x_subset)


# In[88]:


plt.scatter(tsne[:, 0], tsne[:, 1], s= 5, c=y_subset, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(2))
plt.title('Visualizing MNIST through t-SNE', fontsize=24);


# In[89]:


from sklearn.decomposition import PCA
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x_subset)


# In[90]:


# Using the output of PCA as input for t-SNE
get_ipython().run_line_magic('time', '')
pca_tsne = TSNE(random_state = 42, n_components=2, verbose=0, perplexity=40, n_iter=300).fit_transform(pca_result_50)


# In[91]:


#visualising t-SNE again 

plt.scatter(pca_tsne[:, 0], pca_tsne[:, 1], s= 5, c=y_subset, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(2))
plt.title('Visualizing MNIST through t-SNE', fontsize=24);


# In[92]:


get_ipython().run_line_magic('time', '')
pca_tsne2 = TSNE(random_state = 42, n_components=3, verbose=0, perplexity=40, n_iter=300).fit_transform(pca_result_50)


# In[93]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pca_tsne2[:, 0], pca_tsne2[:, 1],pca_tsne2[:,2], s= 5, c=y_subset, cmap='Spectral')
plt.title('Visualizing MNIST through t-SNE in 3D', fontsize=24);
plt.show()


# In[94]:


x=pca_tsne2[:, 0]
y=pca_tsne2[:, 1]
z=pca_tsne2[:, 2]

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=x,                # set color to an array/list of desired values
        colorscale='Spectral',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# In[ ]:




