import pandas as pd
import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

# read the dataset of houses prices
houses = pd.read_csv('datasets/houses_portaland_simple.csv')

# print dataset stats
print(houses.describe())
houses.drop('Bedroom', axis=1, inplace=True)

# shuffling all samples to avoid group bias, in simple it ordered in a random way the row
houses = houses.sample(frac=1, random_state=42).reset_index(drop=True)

""".sample(frac=1): The .sample() method is used to extract a 
random sample from the DataFrame. When frac=1, it means you want
 to sample the entire DataFrame, effectively shuffling all the rows.
 
 random_state=42: The random_state parameter is set to 42 to ensure 
 that the random shuffling is reproducible. Setting a specific random_state 
 value means that the randomization will produce the same results every 
 time you run the code, which can be useful for consistency in data analysis.

 reset_index(drop=True): After shuffling the DataFrame, the .reset_index()
   method is called to reset the index of the DataFrame. 
   The drop=True parameter is used to discard the previous index,
     and a new sequential index is assigned. This can be helpful to 
     avoid issues with the old index when working with the shuffled data.
 """

plt.plot(houses.Size, houses.Price, 'r.')
plt.title("First plot, x is size and y is price")
plt.show()

# another way to test the correlation
print(houses.corr()) # Calculate the correletion coefficient

houses = houses.values # Convert from a dataframe in NumpyArray
print(type(houses))

# compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = houses.mean(axis=0)
std = houses.std(axis=0)

# apply mean and std (standard deviation) compute on training sample to training set and to test set
houses = (houses - mean) / std # Z score scaling standardization

# in order to perform hold-out splitting 80/20 identify max train index value
train_index = round(len(houses) * 0.8)

x = houses[:, 0]
y = houses[:, 1]

plt.plot(x, y, 'r.')
plt.show()

# add bias column
"""By adding a bias column of ones to your independent variables, 
the linear equation can be written in matrix form, 
making it easier to work with and compute: Y=X A"""
x = np.c_[np.ones(x.shape[0]), x]

# create a regressor with specific characteristics
linear = LinearRegression(n_features=x.shape[1], n_steps=1000, learning_rate=0.01)

lineX = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
liney = [linear.theta[0] + linear.theta[1]*xx for xx in lineX]

plt.plot(x[:, 1], y, 'r.', label='Training data')
plt.plot(lineX, liney, 'b--', label='Current hypothesis')
plt.legend()
plt.show()

# fit (try different strategies) your trained regressor
cost_history, theta_history = linear.fit(x, y)

print(f'''Thetas: {*linear.theta,}''')
print(f'''Final train cost:  {cost_history[-1]:.3f}''')

lineX = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
liney = [theta_history[-1, 0] + theta_history[-1, 1]*xx for xx in lineX]

plt.plot(x[:, 1], y, 'r.', label='Training data')
plt.plot(lineX, liney, 'b--', label='Current hypothesis')
plt.legend()
plt.show()

plt.plot(cost_history, 'g--')
plt.show()

#Grid over which we will calculate J
theta0_vals = np.linspace(-2, 2, 100)
theta1_vals = np.linspace(-2, 3, 100)

#initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

#Fill out J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = np.zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        h = x.dot(thetaT.flatten())
        j = (h - y)
        J = j.dot(j) / 2 / (len(x))
        J_vals[t1, t2] = J

#Contour plot
J_vals = J_vals.T

A, B = np.meshgrid(theta0_vals, theta1_vals)
C = J_vals

cp = plt.contourf(A, B, C)
plt.colorbar(cp)
plt.plot(theta_history.T[0], theta_history.T[1], 'r--')
plt.title("")
plt.show()