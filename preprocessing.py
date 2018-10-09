
# coding: utf-8

# In[1]:

import numpy as np
import os
import random
#import tqdm
import cv2


# In[2]:

DATA_DIR='/home/jabir/Project/MHCR/DATASET'
DIR=''
IMG_SIZE=86
LR=1e-3
MODEL_NAME='MHCR.{}-{}-model'.format(LR,'2dconv')


# In[3]:

def create_all_data():
    dataset=[]
    op_class_num=0
    k=0
    for subdir in os.listdir(DATA_DIR):
        k=0
        label=[0 for i in range(44)]
        print('procesing folder:',subdir)
        index=int(subdir.split('R')[1])
        label[index-1]=1
        #print('label of ',subdir, label)
        #if k<4:
        dir_path=os.path.join(DATA_DIR,subdir)
        for img in os.listdir(dir_path):
            k+=1
            if k<=50:
            #print(img)
                img_path=os.path.join(dir_path,img)
                image=cv2.imread(img_path,-1)
                dataset.append([np.array(image),np.array(label)])
                print('processing image:',k)
               
    random.shuffle(dataset)
    np.save('/home/jabir/Project/MHCR/datasetsmall.npy',dataset)
    #dataset=[]
    #print(dataset)
        


# In[4]:

create_all_data()


# In[ ]:



