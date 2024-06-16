from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import color
import matplotlib.image as mpimg
import numpy as np
import joblib

filename = 'test.png'

img = mpimg.imread(filename)
img = color.rgb2gray(img[:,:,0:3])
img = resize(img, (28, 28))

plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

img = np.reshape(img, (1, 28*28))

img = img.astype('float32')

filename = "NN_model.sav"
mlp_mnist = joblib.load(filename)

label = mlp_mnist.predict(img)


print("------------------------")
print("Slika sadrzi znamenku: ", label)
