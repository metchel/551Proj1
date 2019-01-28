import numpy as np
from preprocessing import read_json_file, count_words, process_features, train_validate_test_split

def initVars(optimizer, dataShape, lr, target_lr, num_iter):
    
    tmp = np.log10(lr/target_lr)/num_iter            # decay calculated s.t lr == target_lr after 
    decay = np.power(10, tmp)                        # dividing by decay num_iter ammount of times 
    weights = np.random.rand(dataShape[1], 1)
    
    if optimizer == 'sgd':
        return decay, weights
    
    if optimizer == 'momentum':
        velocity = np.zeros(weights.shape, dtype = float)
        return decay, weights, velocity
    
    if optimizer == 'adam':
        velocity = np.zeros(weights.shape, dtype = float)
        return decay, weights, velocity, velocity   
            

''' 
the velocity term means that our nxtWeights var is updated based on a accumulation of dErr_dW's.
the rho var decides how much we decay the influence of previous dErr_dW's

ie rho = 0.9 means that previously calculated dErr_dW's have more influence on the velocity var than rho = 0.5

'''

def momentum(data, y, lr =0.000001, target_lr =0.0000001, num_iter = 200, threshold =0.0000001, rho =0.99):
    #iter is the ammount of iterations before lr is decayed to target_lr
    rows = data.shape[0]  
    cols = data.shape[1]  
    decay, weights, velocity = initVars('momentum', (rows,cols), lr, target_lr, num_iter)
    
    transp = np.transpose(data)
    xt_x = np.dot(transp, data) 
    xt_y = np.dot(transp, y)
    
    keepUpdating = True
    iter = 0
    while(keepUpdating):
        dErr_dW = 2*(np.dot(xt_x, weights) - xt_y)
        velocity = rho*velocity + dErr_dW
        
        nxtWeights = weights - lr*velocity
        chng = weights - nxtWeights
        weights = nxtWeights
        
        iter += 1
        lr = lr/decay   
        print('updates done: %d -----  lr: %f ----- AVG weights chng: %s ---- AVGerror: %s' %(iter, lr, np.mean(chng), np.mean(y-np.dot(data,weights))))
        if abs(np.mean(chng)) < threshold:
            return weights

        
def sgd(data, y,  lr = 0.00001, target_lr = 0.000001, num_iter = 220, threshold = 0.0000001):
    #iter is the ammount of iterations before lr is decayed to target_lr
    rows = data.shape[0]
    cols = data.shape[1]
    decay, weights = initVars('sgd', (rows,cols), lr, target_lr, num_iter)
    
    transp = np.transpose(data)
    xt_x = np.dot(transp, data) 
    xt_y = np.dot(transp, y)
    
    keepUpdating = True
    iter = 0
    while(keepUpdating):
        nxtWeights = weights -  2*lr*(np.dot(xt_x, weights) - xt_y)
        chng = weights - nxtWeights
        weights = nxtWeights
        
        iter += 1
        lr = lr/decay   
        print('updates done: %d -----  lr: %f----- AVG weights chng: %s ---- AVGerror: %s' %(iter, lr, np.mean(chng), np.mean(y-np.dot(data,weights))))
        if abs(np.mean(chng)) < threshold: 
            return weights

        
def adam(data, y, lr = 0.5, target_lr = 0.05, num_iter = 130, threshold = 0.0000001, rho = 0.93, momentDecay=0.999):
    #num_iter is the ammount of iterations before lr is decayed to target_lr
    rows = data.shape[0]
    cols = data.shape[1]
    decay, weights, moment1, moment2 = initVars('adam', (rows,cols), lr, target_lr, num_iter)

    transp = np.transpose(data)
    xt_x = np.dot(transp, data) 
    xt_y = np.dot(transp, y)
    
    keepUpdating = True
    iter = 1
    while(keepUpdating):
        dErr_dW = 2*(np.dot(xt_x, weights) - xt_y)   #derivative of Error with respect to the weight vector
        moment1 = rho*moment1 + (1-rho)*dErr_dW       
        moment2 = momentDecay*moment2 + (1-momentDecay)*np.multiply(dErr_dW,dErr_dW)
        
        unbiased_m1 = moment1/(1 - rho**iter)         #counteracts the fact that we initialized the moments to zero.
        unbiased_m2 = moment2/(1 - momentDecay**iter)
        
        nxtWeights = weights - lr*unbiased_m1 / (np.sqrt(unbiased_m2) + 1e-7) #add small constant to avoid divide by zero.
        chng = weights - nxtWeights
        weights = nxtWeights
        
        iter += 1
        lr = lr/decay   
        print('updates done: %d -----  lr: %f ----- AVG weights chng: %s ---- AVGerror: %s' %(iter-1, lr, np.mean(chng), np.mean(y-np.dot(data,weights) )))
        #note I print: (iter-1) since I had to initialize iter to 1 to avoid div by zero in first unbiased_m calculation
        if abs(np.mean(chng)) < threshold:
            return weights
    


data = read_json_file("../data/proj1_data.json")
most_frequent_words = count_words([row["text"] for row in data], 160)
X, y = process_features(data, most_frequent_words)
train_X, train_y, validate_X, validate_y, test_X, test_y = train_validate_test_split(X, y)

train_y = np.reshape(train_y, (train_X.shape[0],1)) ##### do this in preprocessing.py for every Y variable s.t shape = (x,1)

x_with_bias = np.insert(train_X, train_X.shape[1], 1, axis = 1) ###### add this to train,validate,test X ONLY, NOT to Y's

weights = momentum(x_with_bias, train_y,  lr = 0.000001, target_lr = 0.0000001, num_iter = 200, rho = 0.99)
newValid = np.insert(validate_X, validate_X.shape[1],1, axis = 1)

print('avg validation err = %f ' %(np.mean(np.dot(newValid, weights) - validate_y)) ) 
