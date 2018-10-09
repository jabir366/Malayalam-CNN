#!/usr/bin/env python
import os 
import numpy as np 
x=0
DATA_DIR='/home/jabir/Project/MHCR/DATASET'
DIR=''
min1=min2=9999
sum1=0
for i in os.listdir(DATA_DIR):
    print('Folder Name:',i)
    x+=1
    DIR=DATA_DIR+'/'+i 
    k=0
    length= len([name for name in os.listdir(DIR) if os.path.isdir(DIR)]) 
    print('lenght:',length) 
    for image in os.listdir(DIR):
        k+=1
    if k<min1:
        min2=min1
        min1=k 
    elif k<min2:
        min2=k 
    sum1+=k
print('min1=',min1)
print('min2=',min1)
print('x=',x)
print('sum is:',sum1)


