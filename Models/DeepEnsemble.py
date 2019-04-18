import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import Adam # I believe this is better optimizer for our case

from keras.utils.vis_utils import plot_model


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

def DNN_model(trainX,trainy): ## need to setup cross-validation
    #define model
    model = Sequential()
    model.add(Dense(20, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #fit model
    model.fit(trainX, trainy, epochs=100, verbose=0)

    return model

def CNN_model(trainX,trainy):

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3,3), activation='relu', input_shape = None))
    model.add(Conv2D(32, kernel_size = (3,3), activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, kernel_size = (3,3), activation='relu'))    
    model.add(Flatten())
    model.add(Dropout(0.4))
    # Lets add softmax activated neurons as much as number of classes
    model.add(Dense(2, activation = "softmax"))
    # Compile the model with loss and metrics
    model.compile(optimizer =  Adam() , loss = "categorical_crossentropy", metrics=["accuracy"])
    
    return model

def bi_LSTM_Attention(trainX,trainy):
    return None


def traditional_classifiers(trainX, trainy):

    kfold = KFold(n_splits=10, random_state=42)
    randomf_clf=RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=100, max_depth=3)
    randomf_pred = cross_val_score(randomf_clf, trainX, trainy, cv=kfold)
    randomf_pred = randomf_pred.mean()


    svm_clf = SVC(kernel='linear', C=1)
    svm_pred = cross_val_score(svm_clf,trainX,trainy, cv=KFold)
    svm_pred=svm_pred.mean()

    GBoost_model = XGBClassifier()
    GBoost_pred=cross_val_score(GBoost_model, trainX, trainy, cv=KFold)
    GBoost_pred=GBoost_pred.mean()

    Gaussian_clf = GaussianNB()
    Gaussian_pred=cross_val_score(randomf_clf, trainX, trainy, cv=kfold)
    Gaussian_pred=Gaussian_pred.mean()
        
        
def ensemble_predictions(models, testX):
    y=[]

    for m in models :
        y.add(m.predict(testX))
    
    return y
# ensemble by stacking 
def meta_classifier(predictions):
    final_prediction=0

    return final_prediction
