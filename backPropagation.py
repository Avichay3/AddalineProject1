import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

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

    dataframe = pd.DataFrame(data)  # pandas function
    return dataframe






dataFrame = createDF(file1)  ## we us the function above to create the data frame using pandas library
dataFrame = dataFrame.rename(columns={dataFrame.columns[0]: 'category'})  # rename the first column of the dataFrame to 'category'

# preparing the data for training and convert the selected DataFrame columns to numpy arrays
X = dataFrame.iloc[:, 1:].to_numpy()  # independent variables that will be used to train a model
y = dataFrame['category'].to_numpy()  # represent the corresponding categories or target variable that the model will try to predict

# split the data into training and testing sets using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



# scale the input features, we used in standardScaler
scaler = StandardScaler()  # this scaling ensures that the features have similar ranges and helps in achieving better model performance
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), activation='relu', solver='adam', random_state=42)
mlp.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test_scaled)




from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def find_best_test_size(X, y, test_sizes):
    best_test_size = None
    best_accuracy = 0.0

    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        # Replace 'model' with your MLP model
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_test_size = test_size

    return best_test_size, best_accuracy


test_sizes = [0.1, 0.15, 0.2, 0.25]
best_test_size, best_accuracy = find_best_test_size(X, y, test_sizes)
print("Best Test Size:", best_test_size)
print("Best Accuracy:", best_accuracy)




# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("---------------- In summary, the accuracy is ---------------")
print("                  ", accuracy)
