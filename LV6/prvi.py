import numpy as np
from sklearn.datasets import fetch_openml
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

imgNum = 50
plt.figure(1)
plt.imshow(np.reshape(X[imgNum,:], (28,28)), cmap = 'gray')
print("Slika sadrži broj: ", y[imgNum])
plt.show()


X = X / 255.
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


mlp_mnist = MLPClassifier (hidden_layer_sizes =(100,100,), 
                            activation = 'relu', 
                            max_iter=100, alpha=1e-3, 
                            momentum=0.9, 
                            solver='sgd', 
                            verbose =10, 
                            random_state =1, 
                            batch_size=64,
                            learning_rate_init=0.1)
mlp_mnist.fit(X_train, y_train)



print("Training set score: %f" %mlp_mnist.score(X_train, y_train))
print("Test set score: %f" %mlp_mnist.score(X_test, y_test))

y_test_pred = mlp_mnist.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_test_pred)


filename = "NN_model.sav"
joblib.dump(mlp_mnist, filename)
