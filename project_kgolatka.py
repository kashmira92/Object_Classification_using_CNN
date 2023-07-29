#!/usr/bin/env python
# coding: utf-8

# In[78]:


import sys
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras import models



# In[ ]:


def run(train_image, train_label, test_image):
# =============================================================================
#     #Train-Test split
#     X_train, X_test, y_train, y_test = train_test_split(train_image, train_label, test_size = 0.2, random_state = 0)
#     #reshaping
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
#     X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1)) 
#     #normalizing the pixel values
#     X_train=X_train/255
#     X_test=X_test/255
#     #Applying CNN Model
#     #defining model
#     # referred  https://www.analyticsvidhya.com/blog/2021/08/beginners-guide-to-convolutional-neural-network-with-implementation-in-python/ for understanding CNN: 
#     model=Sequential()
#     #adding convolution layer
#     model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
#     #adding pooling layer
#     model.add(MaxPool2D(2,2))
#     #adding fully connected layer
#     model.add(Flatten())
#     model.add(Dense(100,activation='relu'))
#     #adding output layer
#     model.add(Dense(100,activation='softmax'))
#     #compiling the model
#     model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#     #fitting the model
#     model.fit(X_train,y_train,epochs=10)
#     #Accuracy on the training data
#     model.evaluate(X_test,y_test)
#     #Predicting the results of train data
# =============================================================================
# y_pred = model.predict(X_test)
# resultT=[]
# for i in range(0,20000): 
#     resultT.append(np.argmax(y_pred[i]))
#     target=np.array(resultT)
# 
# =============================================================================
# =============================================================================

    model = models.load_model("./model_kgolatka.h5")
    
    
    
    #Reshaping on the test data
    X_test_ = test_image.reshape((test_image.shape[0], test_image.shape[1], test_image.shape[2], 1))
    #normalizing the pixel values   
    X_test_=X_test_/255
    #Predicting on the Test data
    y_predict = model.predict(X_test_)
    result=[]
    for i in range(0,100000): 
        result.append(np.argmax(y_predict[i]))
    result=np.array(result)
    print(result)
    with open('project_kgolatka.txt', 'w') as file:  # edit here as your username
        file.write('\n'.join(map(str, result)))
        file.flush()
        return True
    return False   
    


# In[ ]:


if __name__ == "__main__":
    # we will run your code by the following command
    # python project_xiaoq.py argv[1] argv[2]
    # argv[1] is the path of training set
    # argv[2] is the path of test set
    # for example, python demo.py train100c5k_v2.pkl test100c5k_nolabel.pkl

    try:
        train_df = pd.read_pickle(sys.argv[1])
        train_sam = train_df.sample(frac=0.50, random_state=42, replace=True)
        train_dt = train_sam['data'].values
        shapeI=[]
        for i in train_dt:
            reshapeI = i.reshape(28,28)
            shapeI.append(reshapeI)
        train_label = train_sam['target'].values
        features = np.array(shapeI)
        label = np.array(train_label)
        
        test_df = pd.read_pickle(sys.argv[2])
        test_dt = test_df['data'].values
        shapeIT=[]
        for i in test_dt:
            reshapeI = i.reshape(28,28)
            shapeIT.append(reshapeI)
        featuresT = np.array(shapeIT)
        
        
        info = run(features, label, featuresT)
        if not info:
            print(sys.argv[0] + ": Return False")
    except RuntimeError:
        print(sys.argv[0] + ": An RuntimeError occurred")
    except:
        print(sys.argv[0] + ": An exception occurred")


# In[ ]:


# if __name__ == "__main__":
#     # we will run your code by the following command
#     # python project_xiaoq.py argv[1] argv[2]
#     # argv[1] is the path of training set
#     # argv[2] is the path of test set
#     # for example, python demo.py train100c5k_v2.pkl test100c5k_nolabel.pkl

#     try:
#         df = pd.read_pickle('train100c5k_v2.pkl')  # training set path
#         train_data = df['data'].values
#         train_target = df['target'].values
#         df = pd.read_pickle('test100c5k_nolabel.pkl')  # test set path
#         test_data = df['data'].values
#         info = run(train_data, train_target, test_data)
#         if not info:
#             print(sys.argv[0] + ": Return False")
#     except RuntimeError:
#         print(sys.argv[0] + ": An RuntimeError occurred")
#     except:
#         print(sys.argv[0] + ": An exception occurred")
