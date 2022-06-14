#!/usr/bin/env python
# coding: utf-8

# In[8]:


# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from tqdm import tqdm,trange
# from tensorflow.keras import models
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image


# In[9]:


# from tkinter import *
# from tkinter import filedialog
# from PIL import ImageTk, Image  


# In[10]:



# # Label Overview
# classes = { 0:'Speed limit (20km/h)',
#             1:'Speed limit (30km/h)', 
#             2:'Speed limit (50km/h)', 
#             3:'Speed limit (60km/h)', 
#             4:'Speed limit (70km/h)', 
#             5:'Speed limit (80km/h)', 
#             6:'End of speed limit (80km/h)', 
#             7:'Speed limit (100km/h)', 
#             8:'Speed limit (120km/h)', 
#             9:'No passing', 
#             10:'No passing veh over 3.5 tons', 
#             11:'Right-of-way at intersection', 
#             12:'Priority road', 
#             13:'Yield', 
#             14:'Stop', 
#             15:'No vehicles', 
#             16:'Veh > 3.5 tons prohibited', 
#             17:'No entry', 
#             18:'General caution', 
#             19:'Dangerous curve left', 
#             20:'Dangerous curve right', 
#             21:'Double curve', 
#             22:'Bumpy road', 
#             23:'Slippery road', 
#             24:'Road narrows on the right', 
#             25:'Road work', 
#             26:'Traffic signals', 
#             27:'Pedestrians', 
#             28:'Children crossing', 
#             29:'Bicycles crossing', 
#             30:'Beware of ice/snow',
#             31:'Wild animals crossing', 
#             32:'End speed + passing limits', 
#             33:'Turn right ahead', 
#             34:'Turn left ahead', 
#             35:'Ahead only', 
#             36:'Go straight or right', 
#             37:'Go straight or left', 
#             38:'Keep right', 
#             39:'Keep left', 
#             40:'Roundabout mandatory', 
#             41:'End of no passing', 
#             42:'End no passing veh > 3.5 tons' }


# In[1]:


# import skimage
# from skimage import exposure
# from skimage.morphology import disk
# from skimage.morphology import ball
# from skimage.filters import rank


# def gray_scale(image):
#     return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# def low_contrast(image):
#     kernel=disk(30)
#     low_contrast_img=rank.equalize(image,selem=kernel)
#     return low_contrast_img

# def normalization(image):
#     image.np.divide(image,255)
#     return image


# In[2]:


#import stuff
import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm,trange
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image  

import skimage
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank
#----------------------------------------------
# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
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
            42:'End no passing veh > 3.5 tons' }
#----------------------------------------------
#image pre-processing
def gray_scale(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def low_contrast(image):
    kernel=disk(30)
    low_contrast_img=rank.equalize(image,selem=kernel)
    return low_contrast_img

def normalization(image):
    image.np.divide(image,255)
    return image
#----------------------------------------------
classification_model = load_model('traffic_sign_identification_new2.h5')

win = Tk()
win.title('traffic identification')
win.geometry('800x600')
image_label = Label(win)
yolo_label = Label(win)

def openfilename():
    filename = filedialog.askopenfilename(title = 'Select Image')
    return filename

def open_img_identification():
    x = openfilename()
    img = Image.open(x)
    img_convert = cv2.imread(x)
    img_convert=cv2.resize(img_convert,(32,32))
    img_convert=img_convert[:,:,::-1]/255
    img_convert = img_convert.astype('float32')
    img_convert = gray_scale(img_convert)
    img_convert = low_contrast(img_convert)
    cv2.imwrite('casting.jpeg',img_convert)
    
#     img = cv2.imdecode(np.fromfile(x,dtype=np.uint8), cv2.IMREAD_COLOR)
#     resize_img = Image.fromarray(img)
    resize_img = img.resize((200,200),Image.ANTIALIAS)
    resize_img= ImageTk.PhotoImage(resize_img)
    image_label.imgtk =resize_img
    image_label.place(relx=0.3,rely=0.3)
    image_label.configure(image = resize_img)
    image_label.image = resize_img
#     win.update_idletasks()   #更新圖片，必須update
    
text = StringVar()
text.set("input photo to predict")    
result = Label(win,textvariable = text)
result.place(bordermode=OUTSIDE,relx=0.7,rely=0.5)    

def prediction():
    img = 'casting.jpeg'
    test_image=cv2.imread(img)
#     test_image = image.load_img(img,target_size=(32,32))
    test_image=cv2.imread(img,-1)
    test_image = test_image.astype('float32')
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = np.expand_dims(test_image, axis = 3)
#     test_image = image.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis = 0)
    result = classification_model.predict(test_image)
    result = np.round(result,0)
    predict_label = np.argwhere(result==1)
    sign = classes[predict_label[0][1]]
    #prediction = classes[result]
    text.set(sign)
           
identification = Button(win,text = 'open image',font = ('Arial',16),                  command = open_img_identification).place(relx=0.3,rely=0.7)
predict = Button(win,text = 'Predict',command=prediction).place(relx=0.7,rely=0.7)

result_hd = Label(win, text = "RESULT")
result_hd.place(bordermode=OUTSIDE,relx=0.7,rely=0.3)

win.mainloop()


# In[ ]:




