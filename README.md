<H3>Name:MAHASHREE S</H3>
<H3>Register no.: 212225230163</H3>
<H3>Date:30-10-2026</H3>
<H3>Experiment No. 2 </H3>

Implementation of Perceptron for Binary Classification


# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
```python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._b = 0.0
        self._w = None
        self.misclassified_samples = []

    def fit(self, x, y, n_iter=10):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []

        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)

    def f(self, x):
        return np.dot(x, self._w) + self._b

    def predict(self, x):
        return np.where(self.f(x) >= 0, 1, -1)


df = pd.read_csv("breast-cancer.csv")

print(df.head())

x3 = df[["radius_mean", "texture_mean", "perimeter_mean"]].values
y = np.where(df["diagnosis"] == "B", 1, -1)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_title("Breast Cancer Dataset")
ax.set_xlabel("Radius Mean")
ax.set_ylabel("Texture Mean")
ax.set_zlabel("Perimeter Mean")

ax.scatter(x3[y == 1, 0], x3[y == 1, 1], x3[y == 1, 2],
           color="green", marker="o", s=8, label="Benign")
ax.scatter(x3[y == -1, 0], x3[y == -1, 1], x3[y == -1, 2],
           color="red", marker="x", s=8, label="Malignant")
plt.legend()
plt.show()

x = df[["radius_mean", "texture_mean"]].values

plt.figure(figsize=(4, 4))
plt.scatter(x[y == 1, 0], x[y == 1, 1],
            color="green", marker="o", label="Benign")
plt.scatter(x[y == -1, 0], x[y == -1, 1],
            color="red", marker="x", label="Malignant")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.show()

x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0, stratify=y
)

classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train, n_iter=10)

print("Accuracy:", accuracy_score(classifier.predict(x_test), y_test) * 100)

plt.figure(figsize=(4, 4))
plt.plot(range(1, len(classifier.misclassified_samples) + 1),
         classifier.misclassified_samples, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.title("Perceptron Learning")
plt.show()

```
# OUTPUT:
<img width="833" height="755" alt="image" src="https://github.com/user-attachments/assets/6d766693-a204-4ad9-a2c5-91550deb8cb7" />

<img width="837" height="809" alt="image" src="https://github.com/user-attachments/assets/6d7a6f38-a22d-4f52-ac5f-b326526f43a3" />

<img width="835" height="604" alt="image" src="https://github.com/user-attachments/assets/31d32747-97c8-439a-aed4-3a578fb349e3" />

<img width="810" height="447" alt="image" src="https://github.com/user-attachments/assets/cf5cc0ec-07a6-4f6d-b8e6-1b8b19ec1444" />

<img width="871" height="533" alt="image" src="https://github.com/user-attachments/assets/2b8b9a79-1dfe-476c-a7bc-3749a0bacabc" />


# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
