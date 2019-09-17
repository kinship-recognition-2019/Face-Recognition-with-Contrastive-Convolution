import tensorflow as tf
from linear import Linear

class Regressor():
    def __init__(self, n):
        self.n = n
        self.linear = Linear("linear", n, 1)

    def forward(self, x):
        # bs, c = x.shape[0], x.shape[1]
        # print(x) # 64*350 pytorch 64*686
        x = self.linear.forward(x)
        # print(x) # 64*1
        x = tf.sigmoid(x)
        # print(x) # 64*1
        return x
