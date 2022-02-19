from tensorflow.keras.layers import Layer
class SubpixelConv2d(Layer):
# class SubpixelConv2d():
    """It is a 2D sub-pixel up-sampling layer, usually be used
    for Super-Resolution applications, see `SRGAN <https://github.com/tensorlayer/srgan/>`__ for example.

    Parameters
    ------------
    scale : int
        The up-scaling ratio, a wrong setting will lead to dimension size error.
    n_out_channel : int or None
        The number of output channels.
        - If None, automatically set n_out_channel == the number of input channels / (scale x scale).
        - The number of input channels == (scale x scale) x The number of output channels.
    act : activation function
        The activation function of this layer.
    in_channels : int
        The number of in channels.
    name : str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> # examples here just want to tell you how to set the n_out_channel.
    >>> net = tl.layers.Input([2, 16, 16, 4], name='input1')
    >>> subpixelconv2d = tl.layers.SubpixelConv2d(scale=2, n_out_channel=1, name='subpixel_conv2d1')(net)
    >>> print(subpixelconv2d)
    >>> output shape : (2, 32, 32, 1)

    >>> net = tl.layers.Input([2, 16, 16, 4*10], name='input2')
    >>> subpixelconv2d = tl.layers.SubpixelConv2d(scale=2, n_out_channel=10, name='subpixel_conv2d2')(net)
    >>> print(subpixelconv2d)
    >>> output shape : (2, 32, 32, 10)

    >>> net = tl.layers.Input([2, 16, 16, 25*10], name='input3')
    >>> subpixelconv2d = tl.layers.SubpixelConv2d(scale=5, n_out_channel=10, name='subpixel_conv2d3')(net)
    >>> print(subpixelconv2d)
    >>> output shape : (2, 80, 80, 10)

    References
    ------------
    - `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/pdf/1609.05158.pdf>`__

    """
    # Code borrowed from: https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers/convolution/super_resolution.html
    # github/Tetrachrome/subpixel  https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py
    def __init__(
        self,
        scale=2,
        n_out_channels=None,
        act=None,
        in_channels=None,
        name=None  # 'subpixel_conv2d'
    ):
        super().__init__(name, act=act)
        self.scale = scale
        self.n_out_channels = n_out_channels
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build(None)
            self._built = True
        logging.info(
            "SubpixelConv2d  %s: scale: %d act: %s" %
            (self.name, scale, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={in_channels}, out_channels={n_out_channels}')
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):

        if inputs_shape is not None:
            self.in_channels = inputs_shape[-1]

        if self.in_channels / (self.scale**2) % 1 != 0:
            raise Exception(
                "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"
            )
        self.n_out_channels = int(self.in_channels / (self.scale**2))

    def forward(self, inputs):
        outputs = self._PS(X=inputs, r=self.scale, n_out_channels=self.n_out_channels)
        if self.act is not None:
            outputs = self.act(outputs)
        return outputs

    def _PS(self, X, r, n_out_channels):

        _err_log = "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"

        if n_out_channels >= 1:
            if int(X.get_shape()[-1]) != (r**2) * n_out_channels:
                raise Exception(_err_log)

            X = tf.compat.v1.depth_to_space(input=X, block_size=r)
        else:
            raise RuntimeError(_err_log)

        return X