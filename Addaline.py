# Adaline neural network
import numpy as np


def Adaline_fit(Input, Target, lr=0.2, stop=0.001, debug=False):
    weight = np.random.random(Input.shape[1])  ## משקולות לכל פרמטר אפשרי
    bias = np.random.random(1)  ## b
    Error = [stop + 1]
    i=0
    # check the stop condition for the network
    while Error[-1] > stop: # and (i<=2 or Error[-1]-Error[-2] > 0.0012):
        i+=1
        if i>2 and debug and i%10==0:
            print(f"Error[-1]-Error[-2]: {Error[-1]-Error[-2]}")
        error = []
        for i in range(Input.shape[0]):
            Y_pred = sum(weight * Input[i]) + bias
            # Update the weight
            for j in range(Input.shape[1]):
                weight[j] = weight[j] + lr * (Target[i] - Y_pred) * Input[i][j]

            # Update the bias
            bias = bias + lr * (Target[i] - Y_pred)

            # Store squared error value
            error.append((Target[i] - Y_pred) ** 2)
        # Store sum of square errors
        Error.append(sum(error))   ## בכל אפוק - אנחנו נסכום את הסכום של הטעיות שיצא לנו במעבר על כל דגימה באימון - אם הסכום הכולל יהיה קטן מערך כלשהו - נדע לעצור
        if debug and i%10==0:
            print('Error :', Error[-1])
    return weight, bias

def Adaline_predict(X,w,b):
    y=[]
    for i in range(X.shape[0]):
        x = X[i]
        pred = sum(w*x)+b
        y.append(pred)
    median = np.median(y)
    y_label = [0 if x < median else 1 for x in y]
    return y_label

