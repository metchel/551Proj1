import numpy as np

def momentum(data, y, lr = 0.001, target_lr = 0.0001):
    n = data.shape[0]  #rows
    m = data.shape[1]  #columns
    weights = np.random.rand(m,1)
    bias = np.random.rand(n,1)
    keepUpdating = True
    transp = np.transpose(data)
    xt_x = np.dot(transp, data) 
    xt_y = np.dot(transp, y)
    num_iter = 50   #          # iters desired before lr == target_lr 
    tmp = np.log10(lr/target_lr)/num_iter
    decay = np.power(10, tmp) 
    velocity = np.zeros(weights.shape, dtype = float)
    rho = 0.9 #or try 0.99
    iter = 0
    print('the initial lr = %f, target_lr = %f, and decay = %f' %(lr, target_lr, decay))
    print(' ')
    while(keepUpdating):
        dErr_dW = 2*(np.dot(xt_x, weights) + np.dot(transp, bias)  - xt_y)
        velocity = rho*velocity + dErr_dW
        nxtWeights = weights - lr*velocity
        bias -= 2*lr*(np.dot(data, weights) + bias - y) 
        lr = lr/decay   
        iter += 1
        chng = weights - nxtWeights
        weights = nxtWeights
        print('updates done: %d -----  lr: %f ----- AVG weights chng: %s ---- AVGerror: %s' %(iter, lr, np.mean(chng), np.mean(y-np.dot(data,weights) -bias)))
        if abs(np.mean(chng)) < 0.000006:
            keepUpdating = False
            print(' ')
            print('-------------done')
            #print(abs(np.mean(chng)))
            return weights

def sgd(data, y, lr = 0.001, target_lr = 0.0001):
    n = data.shape[0] #rows
    m = data.shape[1] #columns
    weights = np.random.rand(m,1)
    bias = np.random.rand(n,1)
    keepUpdating = True
    transp = np.transpose(data)
    xt_x = np.dot(transp, data) 
    xt_y = np.dot(transp, y)
    num_iter = 100  #         # iters desired before lr == target_lr  
    tmp = np.log10(lr/target_lr)/num_iter
    decay = np.power(10, tmp) 
    iter = 0
    while(keepUpdating):
        nxtWeights = weights -  2*lr*(np.dot(xt_x, weights) + np.dot(transp, bias)  - xt_y)
        bias -= 2*lr*(np.dot(data, weights) + bias - y) 
        lr = lr/decay   
        iter += 1
        chng = weights - nxtWeights
        weights = nxtWeights
        print('updates done: %d -----  lr: %f----- AVG weights chng: %s ---- AVGerror: %s' %(iter, lr, np.mean(chng), np.mean(y-np.dot(data,weights) -bias)))

        if abs(np.mean(chng)) < 0.000006:
            keepUpdating = False
            print(' ')
            print('-------------done')
            #print(abs(np.mean(chng)))
            return weights


data = np.array([[1,2,3], [5,6,7], [9,10,11], [7,8,9]])
y =  np.array([[9],[10],[11],[12]])    #1 by 4
weights = sgd(data, y)
