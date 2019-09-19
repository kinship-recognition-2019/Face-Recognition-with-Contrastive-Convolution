import tensorflow as tf
from linear import Linear

class Regressor():
    def __init__(self, n):
        self.n = n
        self.linear = Linear("linear", n, 1)

    def forward(self, x):
        # bs, c = x.shape[0], x.shape[1]
        # print(x) # 64*350 pytorch 64*686
        # return x
        x = self.linear.forward(x)
        # return x
        # print(x) # 64*1
        x = tf.nn.sigmoid(x)
        # print(x) # 64*1
        return x
