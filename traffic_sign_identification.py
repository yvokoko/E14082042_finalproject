#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import cv2
import tensorflow_datasets as tfds
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tqdm import tqdm,trange
from tensorflow.keras.layers import Dense,Dropout,Flatten,Activation,BatchNormalization
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint


# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[3]:


import pickle
dirpath='C:\\Users\\PCUSER\\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\GTSRB\\traffic_sign\\'
os.listdir(dirpath) 
training_file = "C:\\Users\PCUSER\\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\train.p"
validation_file= "C:\\Users\PCUSER\\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\valid.p"
testing_file = "C:\\Users\PCUSER\\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    


# In[4]:


train_path = 'C:\\Users\\PCUSER\\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\GTSRB\\traffic_sign\\Train'
test = 'C:\\Users\\PCUSER\\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\GTSRB\\traffic_sign\\Test\\'


tr_data = []
tr_name = sorted(os.listdir(train_path)) 

for i in range(0,len(tr_name)):
  tr_data.append(train_path+'\\'+tr_name[i]+'\\')

print(tr_data[3])
print(tr_name)


# In[5]:


print('train length for each train data: ',len(os.listdir(tr_data[0])))
print('test length: ',len(os.listdir(test)))
# for i in range(0,len(tr_name)):
#   tr_sum = tr_sum + len(os.listdir(tr_data[i]))

tr_sum =0
for i in range(0,len(tr_name)):
  tr_sum = tr_sum + len(os.listdir(tr_data[i]))

print(tr_sum)


# In[6]:


import skimage
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank


def gray_scale(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def low_contrast(image):
    kernel=disk(30)
    low_contrast_img=rank.equalize(image,selem=kernel)
    return low_contrast_img

def normalization(image):
    image.np.divide(image,255)
    return image


# In[39]:


# convert the train data to numpy
tr_label=list()
tr_traffic=np.empty(shape=(tr_sum,32,32))
x=0
for i in range(0,len(tr_name)):
  for j in range(0,len(os.listdir(tr_data[i]))):
    f=os.listdir(tr_data[i])[j]
    print(tr_data[i]+f)
    img=cv2.imread(tr_data[i]+f)
    img=cv2.resize(img,(32,32))
    img=img[:,:,::-1]/255
    img = img.astype('float32')
    img = gray_scale(img)
    img = low_contrast(img)
    tr_traffic[x]=img
    tr_label.append(tr_name[i])
    x+=1

tr_label=np.array(tr_label)

print(tr_label)


# In[8]:


print(tr_label.shape)


# In[9]:


# convert the test data to numpy
from skimage.util import img_as_ubyte
te_filename=list()
te_data=np.empty(shape=(len(os.listdir(test)),32,32))
for i in range(len(os.listdir(test))):
  f=os.listdir(test)[i]
  te_filename.append(f)
  img=cv2.imread(test+f)
  img =  img.astype(np.uint8)
  img=cv2.resize(img,(32,32))
  img=img[:,:,::-1]/255
  img = img.astype('float32')
  img = gray_scale(img)
  img = low_contrast(img)
  te_data[i]=img

print(len(te_data))


# In[35]:


import matplotlib.pyplot as plt
import random
def list_images(dataset, dataset_y, ylabel="", cmap=None):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            images: An np.array compatible with plt.imshow.
            lanel (Default = No label): A string to be used as a label for each image.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(15, 16))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        indx = random.randint(0, len(dataset))
        #Use gray scale color map if there is only one channel
        cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
        plt.imshow(dataset[indx], cmap = cmap)
        plt.xlabel(dataset_y[indx])
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


# In[37]:


list_images(tr_traffic,tr_label,'training example')


# In[12]:


print('train data size:',tr_traffic.shape)
print('train label size:',len(tr_label))
print('test data size:',te_data.shape)


# In[13]:


# on-hot label
from tensorflow.keras.utils import to_categorical
tr_label=to_categorical(tr_label)
# 打亂資料順序
from sklearn.utils import shuffle
train_data,train_label=shuffle(tr_traffic,tr_label,random_state=0)
print(train_label.shape)


# In[14]:


train_data_gen = train_data[0:22656]
val_data_gen = train_data[22656:]
train_label_gen = train_label[0:22656,:]
val_label_gen =  train_label[22656:,:]


train_data_gen = train_data_gen.reshape(train_data_gen.shape[0],32,32,1)
val_data_gen = val_data_gen.reshape(val_data_gen.shape[0],32,32,1)

print('train data size: ',train_data_gen.shape)
print('validation data size: ',val_data_gen.shape)
print('train label size: ',train_label_gen.shape)
print('validation label size: ',val_label_gen.shape)


# In[15]:


imgSize=(32,32)
imgShape=(32,32,1)
batchSize=32

from keras.preprocessing.image import ImageDataGenerator
# Data Augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # 以每一張feature map為單位將平均值設為0
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # 以每一張feature map為單位將數值除以其標準差(上述兩步驟就是我們常見的Standardization)
        samplewise_std_normalization=False,  #  將输入的每個樣本除以其自身的標準差。
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.1,  # 隨機旋轉圖片
        zoom_range = 0.1, #  隨機縮放範圍
        width_shift_range=0.1,  #  水平平移，相對總寬度的比例
        height_shift_range=0.1,  # 垂直平移，相對總高度的比例
        horizontal_flip=False,  # 一半影象水平翻轉
        vertical_flip=False)  # 一半影象垂直翻轉
        
datagen.fit(tr_traffic.reshape(39209,32,32,1))


# In[16]:



# Label Overview
classes = { 0:'Speed limit (5km/h)',
            1:'Speed limit (15km/h)', 
            2:'Speed limit (30km/h)', 
            3:'Speed limit (40km/h)', 
            4:'Speed limit (50km/h)', 
            5:'Speed limit (60km/h)', 
            6:'Speed limit (70km/h)', 
            7:'Speed limit (80km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' ,
}


# ## Training Model

# In[17]:


from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.initializers import he_normal
from tensorflow.keras import regularizers
# define model
cnn=models.Sequential() # name the network
# feature extraction
#1 layer
cnn.add(layers.Conv2D(filters=32,kernel_size=(3,3), input_shape=(32,32,1),padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))
#2 layer
cnn.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))

cnn.add(layers.MaxPooling2D(pool_size=(2,2),padding='same'))

#3 layer
cnn.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))

#3 layer
cnn.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))



cnn.add(layers.GlobalAveragePooling2D())
# neron network
cnn.add(layers.Dense(units=512,activation='relu',kernel_initializer = he_normal(),kernel_regularizer = regularizers.l2(l=0.001)))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))
cnn.add(Dropout(0.25))
cnn.add(layers.Dense(units=512,activation='relu',kernel_initializer = he_normal(),kernel_regularizer = regularizers.l2(l=0.001)))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))
cnn.add(Dropout(0.5))
cnn.add(layers.Dense(units=256,activation='relu',kernel_initializer = he_normal(),kernel_regularizer = regularizers.l2(l=0.001)))
cnn.add(BatchNormalization())
cnn.add(Activation("relu"))
cnn.add(Dropout(0.5))
cnn.add(layers.Dense(units=128,activation='relu',kernel_initializer = he_normal(),kernel_regularizer = regularizers.l2(l=0.001)))
cnn.add(Dropout(0.5))
cnn.add(layers.Dense(units=43,activation='softmax'))
# show the model structure
cnn.summary()


# In[18]:


from tensorflow.keras.optimizers import Adam
cnn.compile(optimizer = Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[19]:


estop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

# # 設定模型儲存條件
# checkpoint = ModelCheckpoint('InceptionResNetV2_checkpoint_v2.h5', verbose=1,
#                           monitor='val_loss', save_best_only=True,
#                           mode='min')

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint("traffic_sign_identification_new2.h5", verbose=1, save_best_only=True)

# 設定lr降低條件(0.001 → 0.0005 → 0.00025 → 0.000125 → 0.0001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                           patience=5, mode='min', verbose=1,
                           min_lr=1e-5)


# In[21]:



epoch=50
# with tf.device('/gpu:0'):
#     history = cnn.fit(
#       trainBatch,
#       steps_per_epoch = trainBatch.samples // batchSize,
#       validation_data = valBatch,
#       validation_steps = valBatch.samples // batchSize,
#       epochs=epoch,callbacks =[checkpointer,estop, reduce_lr]
#     )
with tf.device('/gpu:0'):
    history = cnn.fit_generator(datagen.flow(train_data_gen, train_label_gen, batch_size=32),shuffle=True,epochs=epoch, 
                                  validation_data = (val_data_gen, val_label_gen),
                                  verbose = 2, #verbose=2過程全顯示
                                  steps_per_epoch=train_data.shape[0] // 32,
                                  callbacks=[checkpointer,estop]) #we save the best weights with checkpointer


# In[22]:


plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('loss curve')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.title('accuracy curve')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[26]:


from sklearn.metrics import confusion_matrix
pre=cnn.predict(val_data_gen)
pre=np.argmax(pre,axis=1)


# In[32]:


#confusion matrix
import seaborn as sn
cm=confusion_matrix(pre,np.argmax(val_label_gen,axis=1))
fit=plt.figure(figsize=(20,16))
plt.title('confusion matrix')
sn.heatmap(cm,annot=True,cmap='OrRd',fmt='g')
plt.xlabel('prediction')
plt.ylabel('true label')
plt.show()


# In[46]:


prediction_label=prediction.argmax(axis=1)
filename=testBatch.filenames
for i in range(len(filename)):
    filename[i] = filename[i].replace('test\\','')
outputdf=pd.DataFrame()
outputdf['Name']=filename
outputdf['Label']=prediction_label
outputdf.to_csv('C:\\Users\\PCUSER\\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\traffic_sign_predict.csv',index=False)


# In[ ]:


# C:\\Users\\PCUSER\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\GTSRB\\traffic_sign\\Test


# In[8]:


# data_path =[]

# dataset = pd.read_csv("C:\\Users\\PCUSER\\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\GTSRB\\Test.csv", delimiter=",")
# data_id ['ClassId'] = dataset['ClassId'].astype(str).str.zfill(5)
# for i in range(12630):
#     data_path.append(dataset.iat[i,7])
#     data_id.append(int((dataset.iat[i,6])))
# # print(data_id)


# In[39]:


with tf.device('/gpu:0'):
    val_score = classification_model.evaluate_generator(valBatch)
    print('Test loss:', val_score[0])
    print('Test accuracy:', val_score[1])


# In[40]:


import sys
with tf.device('/gpu:0'):
    score = classification_model.predict_generator(testBatch)
    y_classes = score.argmax(axis=-1)
print(y_classes)


# In[41]:


import numpy as np
import pandas as pd
def load_csv_data(file):
    return pd.read_csv(file)

data = load_csv_data('C:\\Users\\PCUSER\\Desktop\\junior\\numerical_method\\final_project\\traffic_identification\\GTSRB\\Test.csv')
class_id = data.iloc[:,6].values
print(class_id)


# In[ ]:





# In[43]:


data_id = np.array(class_id)
different =0

for i in range(len(data_id)):
    if y_classes[i] != data_id[i]:
        different+=1
        

print('accuracy:',1-different/len(data_id))


# In[ ]:




