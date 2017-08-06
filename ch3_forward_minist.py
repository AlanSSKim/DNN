# coding: utf-8
# Load minist image set
#%%
import sys, os
import numpy as np 
import pickle

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def predict(network, x):
    W1, W2, W3 = network['W1'], netowork['W2'], netowork['W3']
    b1, b2, b3 = netowork['b1'], netowork['b2'], netowork['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print ("Accuracy:" + str(float(accuracy_cnt) / len(x)))

#print (x_train.shape)
#print (t_train.shape)
#print (x_test.shape)
#print (t_test.shape)
#
#img = x_train[0]
#label = t_train[0]
#print(label)
#
#print(img.shape)
#img = img.reshape(28,28)
#print(img.shape)
#
#img_show(img)