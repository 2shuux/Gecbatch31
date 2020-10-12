import os
import numpy as np
#import cv2
import matplotlib.pyplot as plt
from PIL import Image
#from keras.preprocessing.image import ImageDataGenerator

#insert ur path here ( should be in directory where train and test folder is there)
input_path = 'C:/Users/vasudha/Desktop/dummy/'

def process_data(img_dims, batch_size,t):
    # Data generation objects
   
    # I will be making predictions off of the test set in one batch size
    # This is useful to be able to get the confusion matrix
    test_data = []
    test_labels = []

    for cond in ['/normal/', '/pneumonia/']:
        for img in (os.listdir(input_path + t + cond)):
            #img = plt.imread(input_path+'test'+cond+img)
            img = Image.open(input_path+ t +cond+img).convert('L')
            #img = cv2.resize(img, (img_dims, img_dims))
            img=img.resize((img_dims, img_dims),)
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            if cond=='/normal/':
                label = 0
            elif cond=='/pneumonia/':
                label = 1
            test_data.append(img)
            test_labels.append(label)
    #test_data=np.reshape(test_data,(280,280,1))   
    #temp=np.array(test_data)
    test_data=np.array(test_data)
    #test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    return  test_data, test_labels

img_dims = 28
#epochs = 10
batch_size = 32

train_x, train_y= process_data(img_dims, batch_size,"train")
test_x, test_y = process_data(img_dims, batch_size,"test")

print("train labels")
print(train_y)
print("test labels")
print(test_y)
print(test_x.shape)
test_x=np.reshape(test_x,(-1,28,28,1))
train_x=np.reshape(train_x,(-1,28,28,1))
print(train_x.shape)
print(test_x.shape)

np.savez_compressed('C:/Users/Vasudha/Desktop/dummy/dataset.npz',x_train=train_x,y_train=train_y,x_test=test_x,y_test=test_y)
