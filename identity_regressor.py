import tensorflow as tf
from linear import Linear


class IdentityRegressor():
    def __init__(self, n, classes):
        self.fc1 = Linear("fc1", n, 256)
        self.fc2 = Linear("fc2", 256, classes)

    def forward(self, x):
        # bs, m, n = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        # print("x", x)
        m, n = int(x.shape[1]), int(x.shape[2])

        # print(x) # 64*14*4608
        x = tf.reshape(x, (-1, n*m))
        # print(x) # 64*64512
        x = tf.nn.relu(self.fc1.forward(x))
        # print(x) # 64*256
        x = self.fc2.forward(x)
        # return x
        x = tf.nn.softmax(x)
        # print(x) # 64*10575
        return x
