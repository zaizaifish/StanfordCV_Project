from sklearn import svm
import time
import numpy as np

def SVM(x_train, y_train, x_test):
    
    # reshape input for svm
    x_train = x_train.reshape(x_train.shape[0],-1)
    
    y_train = y_train.ravel()
    
    x_test = x_test.reshape(x_test.shape[0],-1)
    
    # set parameters
    kernel = 'poly' # radical basis function
    C = 10 # punishment rate
    gamma = 100 # coefficient for kernel function   
    decision_function_shape = 'ovo' # one-vs-one strategy
    model = svm.SVC(kernel = kernel, C = C, gamma = gamma, decision_function_shape = decision_function_shape)
    
    # train model
    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = time.time()-start_time
    
    # test
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    return [y_train_pred, y_test_pred, training_time]