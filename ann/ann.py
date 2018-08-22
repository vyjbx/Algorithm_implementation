import numpy as np 
import matplotlib.pyplot as plt 
import copy

class network(object):
    '''network = network(configs)
    configs = [n1, n2, n3, n4, ..., n_m]
    n1 is the vector length (including bias term)
    n2 to n_m are number of nodes in each layers'''
    def __init__(self, configs, **kwargs):
        self.seed = 5
        self.batch = 19
        self.enta = 0.25
        self.enta_beta = 0.005
        self.err = 1e-7
        self.max_iter = 5000
        self.mom_alpha = 0.9
        self.split = 0.5 #boosting split
        self.grand_loop = 10
        
        self.__dict__.update(kwargs)
        self.RS = np.random.RandomState(self.seed)
        self.configs = configs
        self.n_layers = len(configs)
        self.W = [self.RS.randn(self.configs[i-1], self.configs[i]) for i in range(1, len(self.configs))]
        self.WT = [*map(lambda x: zip(*x), self.W)]
        
    def activate(self, x, w):
        f = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
        return f(np.dot(x, w))
    
    def actibar(self):
        f = lambda x: 1/(1 + np.exp(-x))
        return np.vectorize(lambda x: f(x)*(1-f(x)))
    
    def ffeed(self, x):
        A = []
        x0 = copy.copy(x)
        for mat in self.W:
            A.append(x0)
            x0 = self.activate(x0, mat)
        return x0, A
    
    def predict(self, x):
        return self.ffeed(x)[0]
    
    def SGD(self, tranx, trany, **kwargs):
        self.__dict__.update(kwargs)
        assert tranx.shape[0] == trany.shape[0]
        s = 0
        i = 0
        abar = self.actibar()
        ## training for max_iter epochs, will not use error as a creterian as it is SGD
        ti, C = [], []
        mom = [0]*len(self.W)
        
        while i < self.max_iter:
            input_index = np.array(range(s, s + self.batch)) % tranx.shape[0]
            inputx = tranx[input_index, :]
            inputy = trany[input_index, :]
            yp, A = self.ffeed(inputx)
            
            err = (yp - inputy)

            for (a, w, k) in zip(A[::-1], self.W[::-1], [*range(len(self.W))]):
                zb = abar(np.dot(a, w))
                gradient = np.dot(a.T, (err*zb))
                err = np.dot(err, w.T)
                
                # momentum term
                mom[k] = mom[k]*self.mom_alpha - self.enta/self.batch*gradient
                w += mom[k]
                
                # pure GD 
                # w -= self.enta/self.batch*gradient
            i += 1
            self.enta = self.enta*(1 - self.enta_beta)
            if i%10 == 0:
                ti.append(i)
                C.append(self.RMSE(tranx, trany))
        return ti, C    
            
            
    def RMSE(self, trainx, trainy):
        # this is equivalent to margin RMSE
        py = self.predict(trainx) - trainy
        C = np.dot(py.T, py).mean()/2
        return C
    
    def accuracy(self, trainx, trainy):
        py = (self.predict(trainx) > 0.5).astype(int)
        t = trainy.shape[0]
        right = (py==trainy).sum()
        return right/t
        
    
    def SGD2(self, trainx, trainy, **kwargs):
        ## this is the grand_loop version
        self.__dict__.update(kwargs)
        assert trainx.shape[0] == trainy.shape[0]
        i = 0
        ti, C = [], []
        mom = [0]*len(self.W) # momentum
        
        batch_W = int(self.batch*self.split) # sample number from wrong side
        teeN = trainx.shape[0]

        while i < self.max_iter:
            if (i % self.grand_loop ) == 0: # if to trigger evaluation
                pred_y = self.predict(trainx)
                indices = np.arange(teeN)
                bag_W = indices[np.abs(pred_y - trainy)[:, 0] >= 0.5]
                bag_R = indices[np.abs(pred_y - trainy)[:, 0] < 0.5]
            if bag_W.shape[0] <= batch_W:
                batch_W_select = bag_W
            else:
                batch_W_select = np.random.choice(bag_W, batch_W, replace = False)
                
            batch_R = self.batch - batch_W
            batch_R_select = bag_R[np.random.randint(0, bag_R.shape[0], batch_R)]
            batch_sel = np.hstack([batch_W_select, batch_R_select])
            
            inputx = trainx[batch_sel, :]
            inputy = trainy[batch_sel, :]
            yp, A = self.ffeed(inputx)
            err = yp - inputy
            
            for (a, w, k) in zip(A[::-1], self.W[::-1], [*range(len(self.W))]):
                zb = abar(np.dot(a, w))
                gradient = np.dot(a.T, (err*zb))
                err = np.dot(err, w.T)
                
                # momentum term
                mom[k] = mom[k]*self.mom_alpha - self.enta/self.batch*gradient
                w += mom[k]
                
                # pure GD 
                # w -= self.enta/self.batch*gradient
            if i % 100 == 0:
                print(i)
            i += 1
            self.enta = self.enta*(1 - self.enta_beta)
            if i%10 == 0:
                ti.append(i)
                C.append(self.RMSE(trainx, trainy))
        return ti, C       