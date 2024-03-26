import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch

# Load dataset
diabetes = pd.read_csv('FML_9\diabetes.csv')
print(diabetes.head())
X = diabetes.drop(['Outcome'], axis=1).values
y = diabetes['Outcome'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature normalization
scaler = RobustScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from torch import nn
from torch import optim
device = "cuda" if torch.cuda.is_available() else "cpu"
device


class SimpleNN_1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN_1, self).__init__() # SimpleNN is a sub-class of nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size) # First layer -> Hidden Layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,output_size)# Hidden Layer -> Output
        self.sigmoid = nn.Sigmoid() # Activation (it is a classification task) -> We produce predictions

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x=self.fc3(x)
        x=self.sigmoid(x)
        x=self.fc4(x)
        x=self.sigmoid(x)
        
        return x
    
class SimpleNN_2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN_2, self).__init__() # SimpleNN is a sub-class of nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size) # First layer -> Hidden Layer
        # self.sigmoid1 = nn.Sigmoid() # Activation
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Hidden Layer -> Hidden Layer
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,output_size) # Hidden Layer -> Output
        self.sigmoid = nn.Sigmoid() # Activation (it is a classification task) -> We produce predictions

    def forward(self, x):
        x = self.fc1(x)
        # x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x=self.fc4(x)
        x = self.sigmoid(x)
        return x
    
    
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


torch.manual_seed(42)
non_linearity = False
input_size = X.shape[1]
hidden_size = 125
output_size = 1
if non_linearity == True: # Use NN with non-linear activation function
  model = SimpleNN_1(input_size, hidden_size, output_size).to(device)
else: # Use NN without non-linear activation function
  model = SimpleNN_2(input_size, hidden_size, output_size).to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.001)  # SGD optimizer
X_train_std = torch.from_numpy(X_train_std).float().to(device)
X_test_std = torch.from_numpy(X_test_std).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test= torch.from_numpy(y_test).float().to(device)


epochs = 4000
for epoch in range(epochs):
  model.train() # we just inform the model that we are training it.
  outputs = model(X_train_std) # we obtain the predictions on training set
  outputs = outputs.squeeze() # we adapt prediction size to our labels
  # print(outputs)
  loss = criterion(outputs, y_train) # compute loss function
  outputs = torch.round(outputs).float() # transform predictions in labels
  acc = accuracy_fn(y_true=y_train,
                          y_pred=outputs)
  # compute loss gradients with respect to model's parameters
  loss.backward()
  # update the model parameters based on the computed gradients.
  optimizer.step()
  # In PyTorch, for example, when you perform backpropagation to compute
  # the gradients of the loss with respect to the model parameters, these
  # gradients accumulate by default through the epochs. Therefore, before
  # computing the gradients for a new batch, it's a common practice to zero
  # them using this line to avoid interference from previous iterations.
  optimizer.zero_grad()
  model.eval() # we just inform the model that we are evaluating it.
  with torch.inference_mode(): # we are doing inference: we don't need to compute gradients
    # 1. Forward pass
    test_outputs = model(X_test_std)
    test_outputs = test_outputs.squeeze()
    test_loss = criterion(test_outputs,
                        y_test)
    test_outputs = torch.round(test_outputs).float()
    # print(test_outputs)
    test_acc = accuracy_fn(y_true=y_test,
                            y_pred=test_outputs)
    # 2. Caculate loss/accuracy


  if (epoch + 1) % 20 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")