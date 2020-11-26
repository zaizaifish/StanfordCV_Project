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
    
    #y_train = y_train.ravel()
    
    test = test.reshape(test.shape[0],-1)
    
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
    print("Training time: {}".format(training_time))
    # test
    y_test_pred = model.predict(test)
    return y_test_pred