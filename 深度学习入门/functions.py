import numpy as np

def identity_function(x):
    return x

def step_function(x):
    return np.where(x>0,1,0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad

def softmax(x):
    x = x-np.max(x, axis=-1, keepdims=True)
    '''如果不加 keepdims=True，结果会降维。例如：
        输入 x 是形状 (3, 4) 的二维数组，np.max(x, axis=-1) 的结果是形状 (3,)。
        加上 keepdims=True 后，结果的形状是 (3, 1)，方便后续与原数组 x 进行广播运算。'''
    return np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)
    
def sum_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t): #交叉熵误差
    if y.ndim ==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
        # 对于交叉熵误差t存放是标签，取值只有1和0
    if(t.size==y.size):
        t = t.argmax(axis=1)
        # axis=1按行来求，求每一行的最大值
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # 加误差是防止log(0)
    # 这里计算可以只取标签对应的位置，因为其他的都是0
def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y,t)
        