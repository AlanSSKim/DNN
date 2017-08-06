import numpy as np
import matplotlib.pyplot as plt

#x = np.arange(0,6,0.1)
#y1 = np.sin(x)
#y2 = np.cos(x)
#
#plt.plot(x, y1, label="sin")
#plt.plot(x, y2, linestyle="--", label="cos")
#plt.xlabel("x")
#plt.ylabel("y")
#plt.title('sin & cos')
#plt.legend()
#plt.show()
#%%
#def AND(x1, x2):
#    x = np.array([x1, x2])
#    w = np.array([0.5, 0.5])
#    b = -0.7
#    tmp = np.sum(w*x) + b
#    if tmp <= 0:
#        return 0
#    else:
#        return 1
#
#def NAND(x1, x2):
#    x = np.array([x1, x2])
#    w = np.array([-0.5, -0.5])
#    b = 0.7
#    tmp = np.sum(w*x) + b
#    if tmp <= 0:
#        return 0
#    else:
#        return 1
#
#def OR(x1, x2):
#    x = np.array([x1, x2])
#    w = np.array([0.5, 0.5])
#    b = -0.2
#    tmp = np.sum(w*x) + b
#    if tmp <= 0:
#        return 0
#    else:
#        return 1

#def XOR(x1, x2):
#    s1 = NAND(x1, x2)
#    s2 = OR(x1, x2)
#    y = AND(s1, s2)
#    return y

#%%
#print NAND(1,0)
#print NAND(0,1)
#print NAND(1,1)
#print OR(1,0)
#print OR(0,1)
#print OR(1,1)
#print "------------------------"
#print XOR(0,0)
#print XOR(1,0)
#print XOR(0,1)
#print XOR(1,1)

#%%
# Replace varable type from BOOL to INT
#import numpy as np
#x = np.array([-1.0, 1.0, 2.0])
#print x
#y = x > 0
#print y
#y = y.astype(np.int)
#print y

##%%
#import numpy as np
#import matplotlib.pylab as plt
#
#def step_frunction(x):
#    return np.array( x > 0, dtype=np.int)
#
#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))
#
#x = np.arange(-5.0, 5.0, 0.1)
#
#y = step_frunction(x)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()
#
#y = sigmoid(x)
#print y
#print type(y)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()

