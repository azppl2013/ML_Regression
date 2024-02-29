#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_train = pd.read_csv('train-v3.csv') #將資料變成dataframe
df_test  = pd.read_csv('test-v3.csv')
print(df_train) #印出資料查看資料內容


# In[2]:


#先看看基本資料們
print(df_train.shape) #查看trian的形狀，以方便後來訂定神經網路的輸入層
print(df_test.shape)  #查看test形狀
print('*'*50)
print(df_train.dtypes) #看一下訓練資料的各個欄位的型態


# In[3]:


df_train.drop("id",inplace = True,axis=1) #inplace = True會直接把原本的data改掉，axis=1才會刪除 行
df_test.drop("id",inplace = True,axis=1)  #inplace = True會直接把原本的data改掉，axis=1才會刪除 行
df_train.head(5) #.head()可以查看前幾筆資料，括號填入數字


# In[4]:


trian_corr =df_train.corr() #計算相關係數
plt.subplots(figsize=(15, 10))  # 設置長寬尺寸大小
sns.heatmap(trian_corr, annot=True, vmax=1, cmap="Blues")


# In[5]:


high_corr = trian_corr.index[abs(trian_corr["price"])>0.5]
#abs是取絕對值的意思
#abs(trian_corr["price"])>0.6 這句的意思是與SalePrice有關的係數>0.6的判別式，它會輸出True(大於0.6)或是False
#abs(trian_corr["pPrice"])>0.6 會丟回一堆True和False，放在原本的trian_corr.index[]就會把是大於0.6的傳回去
print(high_corr) 


# In[6]:


x = np.random.randint(0, 100, 10) #0~100隨機找10筆數據
y = 5 * x 
z = np.random.randint(0,100,10)
l = ['a','b','c','d','e','f','g','h','i','j'] #新增的

dataframe = pd.DataFrame({'x':x,'y':y,'z':z,'l':l})

df =dataframe.corr() #計算相關係數
plt.subplots(figsize=(10, 10))  # 設置長寬尺寸大小
sns.heatmap(df, annot=True, vmax=1, square=True, cmap="Blues")
#1.丟入的資料 2.是否輸出熱力圖數值大小 3.最大值顯示4.變成正方形5.要什麼顏色
plt.show()


# In[7]:


import pandas as pd
name = ['John','Jason','Jenny']
df_data = pd.DataFrame(name)
print(df_data)
#以上是基本的建立資料
one_hot_data = pd.get_dummies(df_data) 
#Pandas裡面有get_dummies函數，可以直接進行One-Hot Encoding
pd.DataFrame(one_hot_data)


# In[8]:


import pandas as pd
name = ['John','Jason','Jenny']
df_data = pd.DataFrame(name)
print(df_data)
#以上是基本的建立資料
from sklearn.preprocessing import LabelEncoder #導入LabelEncoder工具
df_data[0] = LabelEncoder().fit_transform(df_data[0])
#LabelEncoder()只能鎖定一行，所以要特別標註是哪行
df_data


# In[9]:


float_data = trian_corr.index #是float或int的的傢伙
all_col = df_train.columns #全部的col
object_data = []
for i in range(len(all_col)): #查找全部的all_col，len(all_col)是長度(要全部找過一遍)
    if all_col[i] not in float_data: #如果在float_data裡面沒有，表示它是object幫的
        object_data.append(all_col[i]) #不是就加上去
print(len(object_data))
print(object_data)


# In[10]:


from sklearn.preprocessing import LabelEncoder
#df_train[pd.isnull(df_train)]  = 'NaN'
for i in object_data: #將轉換是object的傢伙轉換，從object_data陣列一個一個抓出來改造
        df_train[i] = LabelEncoder().fit_transform(df_train[i].factorize()[0]) 
        #pd.factorize()[0]會給nans(缺失值)一個-1的值，若沒寫這個，會造成等號兩邊不等的情況
df_train


# In[11]:


import seaborn as sns
trian_corr =df_train.corr() #計算相關係數
print(trian_corr.shape) #查看形狀
plt.subplots(figsize=(30, 20))  # 設置長寬尺寸大小
sns.heatmap(trian_corr, annot=True, vmax=1, cmap="Blues")


# In[12]:


high_corr = trian_corr.index[abs(trian_corr["price"])>0.5]
#abs是取絕對值的意思
#abs(trian_corr["SalePrice"])>0.6 這句的意思是與SalePrice有關的係數>0.6的判別式，它會輸出True(大於0.6)或是False
#abs(trian_corr["SalePrice"])>0.6 會丟回一堆True和False，放在原本的trian_corr.index[]就會把是大於0.6的傳回去
print(high_corr) 


# In[13]:


import numpy as np
from sklearn.model_selection import train_test_split

x_data = np.array([55,27,33,70,13,26,89,21,10,67])
y_data = np.array([500000,275000,360000,780000,145000,280000,860000,200000,90000,680000])
print("x_data",x_data)
print("y_data",y_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
#X_train,X_test, y_train, y_test = model_selection.train_test_split(train_data,train_target,test_size=0.2, random_state=0)
#train_test_split的括號裡面，1.資料的x那些種類2.y的部分3.test的部分要佔全部資料的多少4.指定一個隨機種子，0的話就是隨機囉
#就是一種隨機切分資料的功能
#隨機種子相同的話，在另一個框框在跑一次一樣的code還是會出現相同的隨機數
print("X_train",X_train)
print("y_train",y_train)
print("X_test",X_test)
print("y_test",y_test)


# In[14]:


for i in df_train.columns: #查找原本資料中所有columns
    if i not in high_corr: #如果沒有相關係數大於0.6的話
        df_train = df_train.drop(i,axis=1) #就把它拔掉
print(df_train)


# In[15]:


train_targets = df_train["price"].values #把SalePrice這行數值整個拉出來
train_data = df_train.drop(columns=["price"]) #刪除SalePrice這行
print(train_targets)
print("*"*50)
print(train_data)


# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_validation,Y_trian,Y_validation = train_test_split(train_data, train_targets, test_size=0.2, random_state=0)
#X_train,X_test, y_train, y_test = model_selection.train_test_split(train_data,train_target,test_size=0.2, random_state=0)
#train_test_split的括號裡面，1.資料的x那些種類2.y的部分3.X_test的部分要佔全部資料的多少(我們這裡是驗證集唷)4.指定一個隨機種子，0的話就是隨機囉
#就是一種隨機切分資料的功能
#隨機種子相同的話，在另一個框框在跑一次一樣的code還是會出現相同的隨機數
print(X_train.shape)
print(Y_trian.shape)
print('*'*50)
print(X_validation.shape)
print(Y_validation.shape)
X_train_dataset = X_train.values #取出數值，轉換回list
X_validation_dataset = X_validation.values
print(X_train_dataset)


# In[17]:


from sklearn import preprocessing #引入所需函式庫
normalize = preprocessing.StandardScaler() #取一個短的名字
# 標準化處理
X_trian_normal_data = normalize.fit_transform(X_train_dataset) #將訓練資料標準化
X_validation_normal_data = normalize.fit_transform(X_validation_dataset) #將驗證資料標準化

print(X_trian_normal_data)
print('*'*70)
print(X_validation_normal_data)


# In[18]:



import tensorflow as tf
from keras.layers import Dense, Dropout, Activation
from keras.layers import LayerNormalization
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.layers import BatchNormalization,Dropout,Dense
from keras.callbacks import ModelCheckpoint
import keras
import keras.utils
from keras import utils as np_utils
#上方為引進所需model

#建立model
def model():
    model = Sequential()
    model.add(layers.Dense(1024,kernel_initializer = 'random_normal',
             activation = 'relu',
             input_shape = (X_trian_normal_data.shape[1],)))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(1024, kernel_initializer = 'random_normal', 
                        activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(512, kernel_initializer = 'random_normal', 
                        activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(256, kernel_initializer = 'random_normal', 
                           activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(128, kernel_initializer = 'random_normal', 
                           activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(32, kernel_initializer = 'random_normal', 
                           activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(16, kernel_initializer = 'random_normal', 
                           activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(1, kernel_initializer = 'random_normal',
                          activation = 'linear'))
    
    adam = tf.optimizers.Adam(learning_rate=0.001)
    #lr學習率
    model.compile(optimizer = adam, loss = 'mae')
    #設定要的優化器以及loss函數
    return model     


# In[19]:


call = ModelCheckpoint('good.h5',
                     monitor = 'val_loss',
                     verbose = 0,
                     save_best_only = True,
                     save_weights_only = True,
                     mode = 'auto',
                     period = 1)

model = model()    
history = model.fit(X_trian_normal_data, Y_trian,
                    validation_data = [X_validation_normal_data, Y_validation],
                    callbacks = [call],
                    epochs = 600, 
                    batch_size = 512, verbose = 1)


# In[20]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.show()


# In[21]:


for i in df_test.columns: #查找原本資料中所有columns
    if i not in high_corr: 
        #如果沒有在訓練時相關係數大於0.6的話就拔掉，因為真正影響房價的不是小於0.6的
        df_test = df_test.drop(i,axis=1) 
        #i是跑過test裡面的columns所有人，axis=1是指定要丟掉"行"
print(df_test) #看一下刪掉相關係數小的人樣子
X_test_dataset = df_test.values #只要數值就好不要dataframe形式
from sklearn import preprocessing
normalize = preprocessing.StandardScaler() 
# 標準化處理
X_test_normal_data = normalize.fit_transform(X_test_dataset)

print(X_test_normal_data) #看一下標準化的結果


# In[24]:


model.load_weights('good.h5') #召喚之前儲存的model
pred = model.predict(X_test_normal_data) #括號內填入要預測的資料
print(pred)


# In[28]:


with open('house_predict.csv', 'w') as f: #開啟一個檔案，house_predict.csv是名字；w是寫入
    f.write('id,price\n') #寫入最上方的列，並用\n往下一列
    for i in range(len(pred)): #len(pred)看整個test有多長，然後用for去跑全部
        f.write(str(i+1) + ',' + str(float(pred[i])) + '\n')
        #寫入，這邊特別注意「+1461」，因為它給的sample_submission是從1461開始
        #中間加上逗號是因為前面有介紹過csv檔是以逗號為間隔的檔案
        #str(float(pred[i]):先把它變成浮點數，再用str()把它變回字串
        #後面的'\n'是要換下一列的意思
        #注意到中間全部是用「+號」隔開，這是字串的基本招式唷


# In[ ]:




