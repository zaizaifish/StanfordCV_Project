from sklearn import svm
import time
import numpy as np

def SVM(train, val, test):
    # reshape input for svm
    x_train = list()
    y_train = list()
    for data in train:
        image = data[0]
        label = data[1]
        x_train.append(image)
        y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0],-1)
    
    x_val = list()
    y_val = list()
    for data in train:
        image = data[0]
        label = data[1]
        x_val.append(image)
        y_val.append(label)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_val = x_val.reshape(x_val.shape[0],-1)
    
    test = test.reshape(test.shape[0],-1)
    
    # set parameters
    kernel = 'linear' # radical basis function
    C = 0.1 # punishment rate
    gamma = 100 # coefficient for kernel function   
    decision_function_shape = 'ovo' # one-vs-one strategy
    model = svm.SVC(kernel = kernel, C = C, gamma = gamma, decision_function_shape = decision_function_shape)
    print('kernel: ', kernel, ' C: ', C, ' gamma: ', gamma)
    # train model
    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = time.time()-start_time
    print("Training time: {}".format(round(training_time,2)))

    # val 
    y_val_pred = model.predict(x_val)
    result_val = 1 - np.sum(np.abs(y_val - y_val_pred)) / y_val.shape[0]
    # test
    y_test_pred = model.predict(test)
    y_true = np.array([0,0,1,0,1,1,0,1,1,0])
    result_test = 1 - np.sum(np.abs(y_test_pred - y_true)) / y_true.shape[0]
    return [result_val, result_test]