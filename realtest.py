#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import argparse
def mplot(img,img2=None):
     
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.moveWindow('img', 600,300)
    cv2.imshow('img',img)
    if img2 is not None:
        cv2.namedWindow('img2',cv2.WINDOW_NORMAL)
        cv2.moveWindow('img', 600,600)
        cv2.imshow('img2',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
function to rearrange the contour bounding boxes. in default the contour bounding boxes comes in the sorted order of
their y co-ordinates . this function returns a list of rectangles [(x1,y1,w1,h1),(x2,y2,w2,h2)...] which are sorted in
the order of x axis on each line. a line will have all recangles of y coordinates between y and y+h of first rectangle '''
def rearrange(cnt):
    b_rect=[]
    for c in cnt:
        rect=cv2.boundingRect(c)
        if rect[2] <=20 or rect[3] <= 20:
            continue
        b_rect.append(rect)
    if b_rect==[]:
        return []
    p=b_rect[0][1]+b_rect[0][3]
    #print('length of brect:',len(b_rect))
    s_rect=[]
    i=0
    length=len(b_rect)
    while i<length:
        p=b_rect[i][1]+b_rect[i][3]
        elem_on_line=[]#elements on a line
        outer=True
        while i<length and p>b_rect[i][1]:
            elem_on_line.append(b_rect[i])
            i+=1
            outer=False
        if outer:
            i+=1
        elem_on_line=sorted(elem_on_line) #,key=lambda x:x[0]
        #print(elem_on_line,i)
        s_rect.extend(elem_on_line)
    return s_rect
        

'''
clear_noice method clear noice from list of images and retrun cleared list of images
'''
def clear_noice(image,ellipse=(7,7),rect=(16,7)):
    
    threshold=128
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ellipse )
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, rect )
    cnt=[]
    grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, threshold, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel2)
    _,contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE )
    #reversing contour list to start processig from top
    contours.reverse()
    cnt.extend(contours)
    #print('contour length:',len(contours))
    #mplot(grad,connected)
    return cnt

def printChars():
    for i,word in enumerate(seperated):
        for j,char in enumerate(word):
            mplot(char)

def splMean(img,thresh):
    sum=0
    nt=0
    for row in img:
        for elem in row:
            if elem>thresh:
                sum+=elem
            else:
                nt+=1
    if sum!=0:
        avg=sum/(img.size-nt)
    else:avg=0
    #print(avg)
    return avg

# In[75]:
#this is where the execution begins
parser=argparse.ArgumentParser()
parser.add_argument('image',help=" the image of handwritten character document ")
args=parser.parse_args()

img=cv2.imread(args.image,0)
print('reading done')
img=img[190:-200,30:-30]
img2=img.copy()
contours=clear_noice(img)
#cutting the image into list of words
s_rect=rearrange(contours)
words=[]
i=0
for rect in s_rect:
    x,y,w,h = rect
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,0),2)
    cv2.putText(img2,str(i),(x+w+10,y+h),0,0.3,(0,0,0))
    word=(img[y:y+h,x:x+w])
    ret,thresh4 = cv2.threshold(word,127,255,cv2.THRESH_TOZERO)
    inc=1*(255-splMean(thresh4,90))
    #ret,thresh4 = cv2.threshold(word,127,255,cv2.THRESH_BINARY)
    #thresh4 = cv2.adaptiveThreshold(word, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    word=np.array([[min(j+inc,255) if j>90 else j for j in thresh4[k]] for k in range(len(thresh4))],dtype=np.uint8)
    #word=np.array([[min(j+80,255) if j>130 else j for j in thresh4[k]] for k in range(len(thresh4))],dtype=np.uint8b)
    #word=thresh4
    #mplot(word)
    words.append(word)
    i+=1
mplot(img2)


# <h4> Drawing bounding boxes on each words</h4><br/>
# <p> Here, word consists of list of words cropped.we are rearranging each word and drawing bounding rectangle to each characters  </p>

# In[73]:

seperated=[]
for i in range(len(words)):
    
    c=clear_noice(words[i],ellipse=(3,3),rect=(2,2))
    s_rect=rearrange(c)
    #print("rect:",s_rect)
    chars=[]
    if s_rect==[]:
        continue
    for rect in s_rect:
        x,y,w,h = rect
        char=words[i][y:y+h,x:x+w]
        char= cv2.copyMakeBorder(char,20,20,20,20,cv2.BORDER_CONSTANT,value=[255,255,255])
        char=cv2.resize(char,(86,86))
        chars.append(char)
    #print(chars)
    
    
        #cv2.rectangle(words[i],(x,y),(x+w,y+h),(0,0,0),2)
        #cv2.putText(words[i],str(i),(x+w+10,y+h),0,0.3,(0,0,0))
           
    #mplot(words[i])
    seperated.append(chars)
    


# In[7]:



# In[8]:
print('loading model...')
model2=load_model('/home/jabir/Project/MHCR/modelMHCR_gray_2.8.96.97.h5')


# In[9]:

mal=np.load('/home/jabir/Project/MHCR/malchar.npy')


# In[16]:

#predict word when a list of predicted classnames are given
def predict_word(p_word):
    pred='' 
    for i in p_word:
        pred+=chr(mal[i])
    return pred


# In[76]:

#pred=model2.predict_class
with open('output.txt','w') as file:
    for i,word in enumerate(seperated):
        if word ==[]:
            continue
        word=np.array(word)
        word2=word
        #print(word.shape)
        pred=model2.predict_classes(word.reshape(-1,86,86,1))
        #cv2.putText(words[i],predict_word(pred),(10,10),0,0.3,(0,255,0))   
        print(predict_word(pred),'\n')
        file.write(predict_word(pred)+'\n')
        # mplot(words[i])


# In[168]:

#class and images
#cls_and_img=np.load('/home/jabir/Project/MHCR/cls_labels.npy',cls_label)


# In[159]:

#npy file for prediction
#mal=np.load('/home/jabir/Project/MHCR/malchar.npy')


# In[80]:
print('Prediction done... \n Output written to output.txt ')
#printChars()


# In[ ]:



