#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ## We are importing data

# In[2]:


data=pd.read_csv(r"C:\Users\ASUS\Desktop\c-dac files\Advance Analytics\Day 2\Bank_ Data2009-14.csv")


# In[3]:


data


# In[4]:


data.head()


# ## We are first two columns 'Obs','account_id' which has no impact on our output so droping that columns

# In[5]:


data.drop(['Obs','account_id'],axis=1, inplace=True)

# we want to know that what column has null and what type of data is

print(data.info())

# By this we know that dimension of data how many row's and how many columns are there in our data

print(data.shape)


# In[6]:


data.columns


# In[ ]:





# In[7]:


#print the categorical variables if any one of them contains too many unique values. 
#In such cases we have to do something to reduce the unique values by clubing some of them together.
print('Sex=',data['sex'].unique())
print('card=',data['card'].unique())
print('Second=',data['second'].unique())
print('Frequency=',data['frequency'].unique())
print('Region=',data['region'].unique())
print('good=',data['good'].unique())


# In[8]:


data.plot(kind='box', subplots=True, layout=(16,2), sharex=False, sharey=False,figsize=(20, 40))
plt.show()


# In[9]:


data.hist(layout=(16,2),figsize=(20, 40))
plt.show()


# In[10]:


data.describe()


# In[11]:


data.cov()


# In[12]:


data.kurtosis()


# In[13]:


data.skew()


# In[14]:


from scipy import stats
z1=stats.zscore(data['cardwdln'])
z2=stats.zscore(data['cardwdlt'])
z3=stats.zscore(data['bankcolt'])
z4=stats.zscore(data['bankrn'])
z5=stats.zscore(data['cardwdlnd'])
z6=stats.zscore(data['othcrnd'])
z7=stats.zscore(data['acardwdl'])
z8=stats.zscore(data['cashwdt'])
z9=stats.zscore(data['cardwdltd'])


# In[15]:


#insert the calculated z-Score into the dataframe
data.insert(0,"Z-Score_cardwdln", list(z1), True)
data.insert(0,"Z-Score_cardwdlt", list(z2), True) 
data.insert(0,"Z-Score_bankcolt", list(z3), True) 
data.insert(0,"Z-Score_bankrn", list(z4), True) 
data.insert(0,"Z-Score_cardwdlnd", list(z5), True) 
data.insert(0,"Z-Score_othcrnd", list(z6), True) 
data.insert(0,"Z-Score_acardwdl", list(z7), True) 
data.insert(0,"Z-Score_cashwdt", list(z8), True) 
data.insert(0,"Z-Score_cardwdltd", list(z9), True) 


# In[16]:


data.head()


# In[17]:


#testing How I can filter out the high z-scores from a single column
data[data['Z-Score_cardwdltd']>1.96]['cardwdltd']


# In[18]:


# Filtering out the extreme z-scores from the required columns 
# and imputing NaN values in the corresponding columns

data.loc[data['Z-Score_cardwdln']>1.96,'cardwdln']=np.nan
data.loc[data['Z-Score_cardwdln']<-1.96,'cardwdln']=np.nan

data.loc[data['Z-Score_cardwdlt']>1.96,'cardwdlt']=np.nan
data.loc[data['Z-Score_cardwdlt']<-1.96,'cardwdlt']=np.nan

data.loc[data['Z-Score_bankcolt']>1.96,'bankcolt']=np.nan
data.loc[data['Z-Score_bankcolt']<-1.96,'bankcolt']=np.nan

data.loc[data['Z-Score_bankrn']>1.96,'bankrn']=np.nan
data.loc[data['Z-Score_bankrn']<-1.96,'bankrn']=np.nan

data.loc[data['Z-Score_cardwdlnd']>1.96,'cardwdlnd']=np.nan
data.loc[data['Z-Score_cardwdlnd']<-1.96,'cardwdlnd']=np.nan

data.loc[data['Z-Score_othcrnd']>1.96,'othcrnd']=np.nan
data.loc[data['Z-Score_othcrnd']<-1.96,'othcrnd']=np.nan

data.loc[data['Z-Score_acardwdl']>1.96,'acardwdl']=np.nan
data.loc[data['Z-Score_acardwdl']<-1.96,'acardwdl']=np.nan

data.loc[data['Z-Score_cashwdt']>1.96,'cashwdt']=np.nan
data.loc[data['Z-Score_cashwdt']<-1.96,'cashwdt']=np.nan

data.loc[data['Z-Score_cardwdltd']>1.96,'cardwdltd']=np.nan
data.loc[data['Z-Score_cardwdltd']<-1.96,'cardwdltd']=np.nan


# In[19]:


data.info()


# In[20]:


# save a copy of the data to disc.
data.to_csv('see.csv')


# In[21]:


data['bankrn'].median()


# In[22]:


# imputing the median values in place of the NaN values
data['cardwdln']=data['cardwdln'].fillna(data['cardwdln'].median())
data['cardwdlt']=data['cardwdlt'].fillna(data['cardwdlt'].median())
data['bankcolt']=data['bankcolt'].fillna(data['bankcolt'].median())
data['bankrn']=data['bankrn'].fillna(data['bankrn'].median())
data['cardwdlnd']=data['cardwdlnd'].fillna(data['cardwdlnd'].median())
data['othcrnd']=data['othcrnd'].fillna(data['othcrnd'].median())
data['acardwdl']=data['acardwdl'].fillna(data['acardwdl'].median())
data['cashwdt']=data['cashwdt'].fillna(data['cashwdt'].median())
data['cardwdltd']=data['cardwdltd'].fillna(data['cardwdltd'].median())


# In[23]:


# getting rid of the z-Score columns
data=data.drop(['Z-Score_cardwdltd', 'Z-Score_cashwdt','Z-Score_acardwdl','Z-Score_othcrnd','Z-Score_cardwdlnd','Z-Score_bankrn','Z-Score_bankcolt','Z-Score_cardwdlt','Z-Score_cardwdln'], axis=1)
data.shape


# In[24]:


data.plot(kind='box', subplots=True, layout=(16,2), sharex=False, sharey=False,figsize=(20, 40))
plt.show()


# In[25]:


data.hist(layout=(16,2),figsize=(20, 40))
plt.show()


# In[26]:


# Split the features and the target variables first

X=data.iloc[:,0:37]

y=data.iloc[:,-1]


# ## By using get_dummy function we are converting character data columns into numerical 

# In[27]:


X1=pd.get_dummies(X)
X=X1
X2=X
X2


# In[28]:


X.head()


# In[ ]:





# In[ ]:





# In[29]:


data.columns


# In[30]:


X1.columns


# In[31]:


X.shape


# # Standarization

# In[32]:


from sklearn import preprocessing 

X=preprocessing.StandardScaler().fit(X).transform(X)


# In[33]:


X


# In[34]:


#Encode the y variable as well
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y_data_final=labelencoder_y.fit_transform(y)
y_data_final


# In[35]:


from sklearn.metrics import confusion_matrix , classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ## We are Training and Testing data

# In[36]:


X_train,X_test,y_train,y_test=train_test_split(X,y_data_final,test_size=0.4,random_state=42)
logreg=LogisticRegression(max_iter=500)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[37]:


X


# # Normalization

# In[38]:


from sklearn.preprocessing import Normalizer


# In[39]:


n=Normalizer().fit(X1)


# In[40]:


X1=n.transform(X1)


# In[41]:


X1


# In[42]:


X_train,X_test,y_train,y_test=train_test_split(X1,y_data_final,test_size=0.4,random_state=42)
logreg=LogisticRegression(max_iter=500)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # PCA

# In[43]:


from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd


# In[44]:


pca=PCA(n_components=2,svd_solver='randomized')


# In[45]:


a=pca.fit(X).transform(X)


# In[46]:


type(a)


# In[ ]:





# In[47]:


b=pd.DataFrame(data=a,columns=['P1','P2'])
b


# In[48]:


y=b


# In[49]:


X_train,X_test,y_train,y_test=train_test_split(X,y_data_final,test_size=0.4,random_state=42)
logreg=LogisticRegression(max_iter=500)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[53]:


logreg.score(X_test,y_test)


# In[ ]:




