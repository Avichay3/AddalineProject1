import numpy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from Addaline import Adaline_fit, Adaline_predict


def get_x_y(chars:list):
    with open("./result.txt", "r") as f:
        lines = f.readlines()  # read all lines from file
    vectors = []
    Y = []
    for line in lines:
        line = line.replace("(", "").replace(")", "")
        label = line[0]
        if label in chars:
            label = 0 if label == chars[0] else 1
            Y.append(int(label))
            # split the line by whitespace and convert values to float
            values = list(map(np.float64, line.split(',')[1:]))
            # append the values as a numpy array to the vectors list
            vectors.append(np.array(values))

    # stack all vectors into a 2D numpy array
    X = np.vstack(vectors)
    Y = np.array(Y)

    return X, Y

def main(chars:list, lr, stop, debug=False):
    _X, _Y = get_x_y(chars)
    # using the train test split function
    X_train, X_test, y_train, y_test = train_test_split(_X, _Y,
                                                        random_state=42,
                                                        test_size=0.2,
                                                        shuffle=True)
    # Define the number of folds for cross-validation
    num_folds = 5
    # Create a KFold object for cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True)
    # Initialize lists to store the accuracy scores for each fold
    accuracy_scores = []
    W = []
    B = []
    #--------------------- Train + Validation -------------------------------
    # Loop over each fold of the data
    for train_index, test_index in kf.split(X_train):
        # Split the data into training and testing sets
        X_train_fd, X_test_fd = X_train[train_index], X_train[test_index]
        y_train_fd, y_test_fd = y_train[train_index], y_train[test_index]

        # Train the Adaline classifier on the training data
        w, b = Adaline_fit(X_train_fd, y_train_fd, lr, stop, debug)
        W.append(w)
        B.append(b)

        # Predict the labels for the testing data
        y_pred = Adaline_predict(X_test_fd, w, b)

        # Compute the accuracy score for the validation data
        accuracy = accuracy_score(y_test_fd, y_pred)
        print(f"validation accuracy: {accuracy}")

        # Append the accuracy score to the list
        accuracy_scores.append(accuracy)

    # Compute the mean accuracy score over all folds
    mean_accuracy = np.mean(accuracy_scores)
    print(f"\tvalidation mean accuracy: {mean_accuracy}")

    #--------------------- Test -------------------------------
    W = np.array(W)
    # Compute the vector of means along axis 0
    mean_w = np.mean(W, axis=0)
    B = np.array(B)
    # Compute the vector of means along axis 0
    mean_b = np.mean(B, axis=0)

    y_pred_test = Adaline_predict(X_test, mean_w, mean_b)

    # Compute the accuracy score for the testing data
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"\t->Final accuracy on test set: {accuracy}")

if __name__ == "__main__":
    print("\n-------------------------- mem & bet ---------------------------------------")
    main(['2', '1'], 0.0001, 130, debug=False)   #Best: 0.0001, 130 -> test-0.84
    print("-------------------------- bet & lamed ---------------------------------------")
    main(['1', '3'], 0.0001, 130, debug=False)   #Best: 0.0001, 130 -> test-0.84
    print("\n-------------------------- mem & lamed ---------------------------------------")
    main(['2', '3'], 0.0001, 130, debug=False)   #Best: 0.0001, 130 -> test-0.84

