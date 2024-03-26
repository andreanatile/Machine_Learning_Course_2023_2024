import numpy as np


class RegMetrics:
    def __init__(self,y_true,y_pred):
        self.y_true=y_true
        self.y_pred=y_pred
        
        
    def mean_squared_error(self):
        mse=np.mean((self.y_pred-self.y_true)**2)
        return mse
    
    def mean_absolute_error(self):
        mae=np.mean(np.abs(self.y_pred-self.y_true))
        return mae
    
    def r2(self):
        mean_true=np.mean(self.y_true)
        ss_total=np.mean((self.y_true-mean_true)**2)
        ss_residual=np.mean((self.y_pred-self.y_true)**2)
        
        r2=1-ss_residual/ss_total
        return r2
    
    
        