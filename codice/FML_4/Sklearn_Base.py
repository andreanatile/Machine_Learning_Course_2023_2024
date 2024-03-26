import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler #RobustScaler more robust on outliers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the diabetes dataset
diabetes = pd.read_csv('FML_4\diabetes.csv')

# Print dataset stats
print(diabetes.describe())
print(diabetes.columns)

# Shuffling all samples to avoid group bias
#diabetes = diabetes.sample(frac=1).reset_index(drop=True)

# Select features and target variable
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes[selected_features].values
y = diabetes['Outcome'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Random state is like a seed, if i run this code in two computer with a stame random_state
#it will produce the same result.It said that is reproducible. In this case it will
#happen only if we remove the shuffling of our data


# More robust to the outliers
#scaler = RobustScaler()
# Standardize the features using StandardScaler
scaler = StandardScaler()

#we evaluate only the mean and std of the training data using fit_transform
#if we want to use the same training std and mean to the test we use .transform
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Create a logistic regression model
logistic_model = LogisticRegression(random_state=42)

# Train the logistic regression model on the standardized training data
logistic_model.fit(X_train_std, y_train)

# Make predictions on the standardized test data
y_pred = logistic_model.predict(X_test_std)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report_str)
print('Confusion Matrix:\n', conf_matrix)
