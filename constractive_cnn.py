import cnn_functions as cf


class ConstractiveFourLayers():
    def __init__(self, input_op):
        self.input_op = input_op

    def forward(self):
        conv1 = cf.conv_op(self.input_op, name="conv1", kh=3, kw=3, n_out=64, dh=1, dw=1)
        pool1 = cf.mpool_op(conv1, name="pool1", kh=3, kw=3, dh=2, dw=2)
        conv2 = cf.conv_op(pool1, name="conv2", kh=3, kw=3, n_out=128, dh=1, dw=1)
        pool2 = cf.mpool_op(conv2, name="pool2", kh=3, kw=3, dh=2, dw=2)
        conv3 = cf.conv_op(pool2, name="conv3", kh=3, kw=3, n_out=256, dh=1, dw=1)
        pool3 = cf.mpool_op(conv3, name="pool3", kh=3, kw=3, dh=2, dw=2)
        conv4 = cf.conv_op(pool3, name="conv4", kh=3, kw=3, n_out=512, dh=1, dw=1)
        pool4 = cf.mpool_op(conv4, name="pool4", kh=3, kw=3, dh=2, dw=2)

        return pool4