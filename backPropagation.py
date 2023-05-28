import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

file1 = 'result.txt'  # loading the file with all the vectors represents the letters ב,ל,מ

# This function will use us to read and create DataFrame from the vectors
def createDF(file_name):
    data = []
    with open(file_name, 'r', encoding='latin-1') as f:
        file_content = f.read()

    lines = file_content.splitlines()
    for line in lines:
        values = line.strip().strip('()').replace('[', '').replace(']', '').split(',')
        data.append(values)

    dataframe = pd.DataFrame(data)
    return dataframe



dataFrame = createDF(file1)
dataFrame.columns = ['category'] + list(dataFrame.columns[1:])

# Preparing the data for training
X = dataFrame.iloc[:, 1:].values
y = dataFrame['category'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the input features, we used in standardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), activation='relu', solver='adam', random_state=42)
mlp.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test_scaled)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
