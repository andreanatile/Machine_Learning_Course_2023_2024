import numpy as np

np.random.seed(42)

class LinearRegression:

    def __init__(self,learning_rate=1e-2,n_steps=2000,n_features=1):
        self.learning_rate=learning_rate
        self.n_steps=n_steps
        self.theta=np.random.rand(n_features)

    def fit_full_batch(self,X,y):
        m=len(X)
        theta_history=np.zeros((self.n_steps,self.theta.shape[0]))
        cost_history=np.zeros(self.n_steps)

        for step in range(0,self.n_steps):
            pred=np.dot(X,self.theta)
            error=pred-y

            self.theta=self.theta-self.learning_rate*np.dot(X.T,error)*(1/m)
            theta_history[step]=self.theta.T

            cost=(1/m)*np.dot(error.T,error)
            cost_history[step]=cost
    
        return cost_history,theta_history
    
    def fit_mini_batch(self,X,y,b=8):
        m=len(X)
        theta_history=np.zeros((self.n_steps,self.theta.shape[0]))
        cost_history=np.zeros(self.n_steps)

        for step in range(0,self.n_steps):
            total_error=np.zeros((self.theta.shape[0]))
            for i in range(0,m,b):
                xi=X[i:i+b]
                yi=y[i:i+b]

                pred=np.dot(xi,self.theta)
                error=pred-yi

                total_error +=np.dot(xi.T,error)
            
            self.theta=self.theta-self.learning_rate*total_error*(1/b)
            theta_history[step]=self.theta.T

            error=np.dot(X,self.theta)-y
            cost=(1/m)*np.dot(error.T,error)
            cost_history[step]=cost

        return cost_history,theta_history
    
    def fit_sgd(self,X,y):
        m=len(X)
        theta_history=np.zeros((self.n_steps,self.theta.shape[0]))
        cost_history=np.zeros(self.n_steps) 

        for step in range(0,self.n_steps):
            random_index=np.random.randint(m)
            
            xi=X[random_index]
            yi=y[random_index]

            pred=np.dot(xi,self.theta)
            error=pred-yi

            self.theta=self.theta-self.learning_rate*np.dot(xi.T,error)
            theta_history[step]=self.theta.T

            error=np.dot(X,self.theta)-y
            cost=(1/m)*np.dot(error.T,error)
            cost_history[step]=cost
        
        return cost_history,theta_history