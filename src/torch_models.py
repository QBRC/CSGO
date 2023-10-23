import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes, scale_factor=1, resize_mode='bilinear',
                 encoder={}, decoder={}, out_channels=None, return_logits=False, **kwargs):
        """ Abstract UNet class. 
            num_classes: num_classes in output layer.
            scale_factor: resize to original size.
            encoder: nn.Module or dictionary. (call UNetfeatures(**encoder)).
            decoder: nn.Module or dictionary. (call self.default_decoder(**decoder)).
            out_channels: call encoder.feature_channels() if None
        """
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.resize_mode = resize_mode
        
        ## encoder
        if isinstance(encoder, dict):
            self.encoder = self.get_encoder(**encoder)
        else:
            self.encoder = encoder
        assert isinstance(self.encoder, nn.Module)
        
        ## out_channels
        self.out_channels = out_channels or self.encoder.out_channels  # self.encoder.feature_channels()
        
        ## decoder
        if isinstance(decoder, dict):
            self.decoder = self.get_decoder(**decoder)
        else:
            self.decoder = decoder
        assert isinstance(self.decoder, nn.Module)

        ## final classification and resize layer
        classifier = [
            nn.Conv2d(self.out_channels[0], num_classes, kernel_size=1),
        ]
        
        if not return_logits:
            classifier.append((nn.Softmax2d() if num_classes > 1 else nn.Sigmoid()))
        
        ## pytorch interpolate == tf interpolate != keras.Upsample/tf.js.Upsample.
        ## Will see differences on resize in keras and pytorch.
        if scale_factor is not None and scale_factor != 1:
            classifier = [
                nn.Upsample(scale_factor=scale_factor, mode=self.resize_mode)
            ] + classifier
        
        self.classifier = nn.Sequential(*classifier)
    
    def get_encoder(self, **kwargs):
        return UNetFeatures(**kwargs)
    
    def get_decoder(self, **kwargs):
        up = kwargs.setdefault('up', 'ResizeConv2d')
        norm_layer = kwargs.setdefault('norm_layer', 'batch')
        kernel_size = kwargs.setdefault('kernel_size', 3)
        
        decoder = nn.ModuleList()
        for in_c, out_c in zip(self.out_channels[-1:0:-1], self.out_channels[-2::-1]):
            if up == 'ResizeConv2d':
                up_layer = ResizeConv2d(in_c, out_c, stride=2, mode='bilinear',
                                        conv=(Conv2d, {'kernel_size': kernel_size, 'padding': 'default'}))
            elif up == 'SubPixelConv2d':
                up_layer = SubPixelConv2d(in_c, out_c, stride=2, 
                                          conv=(Conv2d, {'kernel_size': kernel_size, 'padding': 'default'}))
            elif up == 'ConvTranspose2d':
                ## fix the padding issue
                # padding = (kernel_size-1)//2
                up_layer = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=2)
            
            conv_layer = nn.Sequential(
                Conv2dBNReLU(out_c*2, out_c, 3, norm_layer=norm_layer), 
                Conv2dBNReLU(out_c, out_c, 3, norm_layer=norm_layer)
            )
            modules = nn.ModuleDict([('up', up_layer), ('conv', conv_layer)])
            decoder.append(modules)
        
        return decoder
    
    def forward(self, x):
        features = self.encoder(x)
        x = features.pop()
        for layers in self.decoder:
            up = layers['up'](x)
            up = torch.cat([up, features.pop()], dim=1)
            x = layers['conv'](up)
        
        return self.classifier(x)

class UNetFeatures(nn.Module):
    def __init__(self, in_channels=3, n_channels=32, n_downsampling=4, pool='maxpool', norm_layer='batch'):
        super(UNetFeatures, self).__init__()
        assert pool in ['maxpool', 'stride'], "{} is not a supported down sampling strategy".format(pool)
        self.in_channels = 3
        self.n_channels = 32
        self.n_downsmapling = 4
        norm_layer, bias = get_norm_layer_and_bias(norm_layer)
        
        ## build feature layers
        self.out_channels = [n_channels]
        self.conv = [
            Conv2dBNReLU(in_channels, n_channels, kernel_size=3, norm_layer=norm_layer),
            Conv2dBNReLU(n_channels, n_channels, kernel_size=3, norm_layer=norm_layer),
        ]
        
        for i in range(n_downsampling):
            in_channels, out_channels = self.out_channels[-1], 2**(i+1) * n_channels
            if pool == 'maxpool':
                self.conv.append(nn.MaxPool2d(2))
                self.conv.append(Conv2dBNReLU(in_channels, out_channels, kernel_size=3, stride=1, norm_layer=norm_layer))
            elif pool == 'stride':
                self.conv.append(Conv2dBNReLU(in_channels, out_channels, kernel_size=3, stride=2, norm_layer=norm_layer))
            self.conv.append(Conv2dBNReLU(out_channels, out_channels, kernel_size=3, norm_layer=norm_layer))
            # self.conv.append(nn.MaxPool2d(2))
            self.out_channels.append(out_channels)
        
        self.conv = nn.Sequential(*self.conv)
        if pool == 'maxpool':
            self.return_layers = [3*i+2 for i in range(n_downsampling+1)]
        elif pool == 'stride':
            self.return_layers = [2*i+2 for i in range(n_downsampling+1)]
    
    def feature_channels(self, idx=None):
        if isinstance(idx, int):
            return self.out_channels[idx]
        if idx is None:
            idx = range(len(self.out_channels))
        return [self.out_channels[_] for _ in idx]
    
    def forward(self, x):
        res = []
        for s, t in zip([0] + self.return_layers, self.return_layers):
            for _ in range(s, t):
                x = self.conv[_](x)
            res.append(x)
        
        return res

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        if padding == 'default':
            padding = tuple((k-1)//2 for k in [kernel_size, kernel_size])
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

class ResizeConv2d(nn.Sequential):
    """ Upsampling layer with resize convolution. """
    def __init__(self, in_channels, out_channels, stride=2, 
                 conv=(Conv2d, {'kernel_size': 3, 'padding': 'default'}), mode='bilinear'):
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        super(ResizeConv2d, self).__init__(
            nn.Upsample(scale_factor=stride, mode=mode),            
            conv[0](in_channels, out_channels, **conv[1]),
        )

class SubPixelConv2d(nn.Sequential):
    """ Upsampling layer with better modelling power. 
        Sub-pixel convolution usually gives better result than resize convolution.
        Use incr init (and weight_norm) to avoid checkboard artifact. 
        https://arxiv.org/pdf/1707.02937.pdf
        May combine with (https://arxiv.org/pdf/1806.02658.pdf):
            nn.Sequential(
                nn.LeakyReLU(inplace=True),
                nn.ReplicationPad2d((1,0,1,0)),
                nn.AvgPool2d(2, stride=1),
            )
        to generate non-checkboard artifact image.
    """
    def __init__(self, in_channels, out_channels, stride=2, 
                 conv=(Conv2d, {'kernel_size': 3, 'padding': 'default'})):
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # torch.nn.utils.weight_norm()
        super(SubPixelConv2d, self).__init__(
            conv[0](in_channels, out_channels * stride ** 2, **conv[1]),
            nn.PixelShuffle(stride),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.icnr_(self[0].weight)
    
    def icnr_(self, x):
        """ ICNR init of conv weight. """
        ni, nf, h, w = x.shape
        ni2 = int(ni / (self.stride**2))
        k = nn.init.kaiming_normal_(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, self.stride**2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        x.data.copy_(k)



class ConvBNReLU(nn.Sequential):
    def __init__(self, conv, norm_layer=None, activation=None, dropout_rate=0.0):
        ## get norm layer:
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        
        layers = [conv]
        if norm_layer is not None:
            layers.append(norm_layer(conv.out_channels))
        if activation is not None:
            layers.append(activation)
        if dropout_rate:
            layers.append(nn.Dropout2d(dropout_rate))
        
        super(ConvBNReLU, self).__init__(*layers)


class Conv2dBNReLU(ConvBNReLU):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding='default', dilation=1, groups=1, bias=None, 
                 norm_layer='batch', activation=nn.ReLU(inplace=True), 
                 dropout_rate=0.0, 
                ):
        """ Create a Conv2d->BN->ReLU layer. 
            norm_layer: batch, instance, None
            activation: a nn layer.
            padding: 
                'default' (default): torch standard symmetric padding with (kernel_size - 1) // 2.
                int: symmetric padding to pass to nn.Conv2d(padding=padding)
                "same": tf padding="same", asymmetric for even kernel (l_0, r_1), etc)
                "valid": tf padding="valid", same as padding=0
        """
        ## get norm layer:
        norm_layer, bias = get_norm_layer_and_bias(norm_layer, bias)
        ## use Conv2d (extended nn.Conv2d) to support padding options
        if padding == 'default':
            padding = tuple((k-1)//2 for k in [kernel_size, kernel_size])
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                         padding, dilation, groups, bias=bias)
        super(Conv2dBNReLU, self).__init__(conv, norm_layer, activation, dropout_rate)


def get_norm_layer_and_bias(norm_layer='batch', use_bias=None):
    """ Return a normalization layer and set up use_bias for convoluation layers.
    
    Parameters:
        norm_layer: (str) -- the name of the normalization layer: [batch, instance]
                    None -- no batch norm
                    other module: nn.BatchNorm2d, nn.InstanceNorm2d

    For BatchNorm: use learnable affine parameters. (affine=True)
                   track running statistics (mean/stddev). (track_running_stats=True)
                   do not use bias in previous convolution layer. (use_bias=False)
    For InstanceNorm: do not use learnable affine parameters. (affine=False)
                      do not track running statistics. (track_running_stats=False)
                      use bias in previous convolution layer. (use_bias=True)
    Test commands:
        get_norm_layer_and_bias('batch', None) -> affine=True, track_running_stats=True, False
        get_norm_layer_and_bias('batch', True) -> affine=True, track_running_stats=True, True
        get_norm_layer_and_bias('instance', None) -> affine=False, track_running_stats=False, True
        get_norm_layer_and_bias('instance', False) -> affine=False, track_running_stats=False, False
        get_norm_layer_and_bias(None, None) -> None, True
        get_norm_layer_and_bias(None, False) -> None, False
        get_norm_layer_and_bias(nn.BatchNorm2d, None) -> BatchNorm2d, False
        get_norm_layer_and_bias(nn.BatchNorm2d, True) -> BatchNorm2d, True
        get_norm_layer_and_bias(nn.InstanceNorm2d, None) -> InstanceNorm2d, True
        get_norm_layer_and_bias(nn.InstanceNorm2d, False) -> InstanceNorm2d, False
    """
    if isinstance(norm_layer, str):
        if norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('normalization layer {} is not found'.format(norm_layer))
    
    if use_bias is None:
        use_bias = norm_layer == nn.InstanceNorm2d
    
    return norm_layer, use_bias