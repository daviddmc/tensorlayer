#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from .layers import *
from .activation import lrelu

def unet(x, is_train = True, reuse = False, 
         num_channel_out = 1, num_downsampling = 3,
         use_bn = True, use_res = True, use_concat = True,
         method_down = 'mean', method_up = 'upsample',
         block_type = {'down':'dense', 'center':'dense', 'up':'dense'},
         # conv block
         conv_depth = 2, conv_channel_first = 16,
         # dense block
         dense_depth = 3, dense_growth_rate = 12,
         # res block
         res_num = 1, res_channel_first = 16,
         act_out = tf.tanh, act = tf.nn.relu, Name = 'unet'):
         
    """U-Net for image denoising and super resolution. A multi-scale encoder-decoder network with symmetric concatenate connection.
    The input images and output images must have the same size. Therefore, when this network is used in super resolution the input
    image must be upsampled first.
    
    Parameters
    ----------
    x : input tf tensor
    is_train : only bn layers will be affected by this flag
    reuse : reuse variables
    num_channel_out : number of output channels
    num_channel_first : number of channels of the first convolution layer
    num_conv_per_pooling : number of convolution layers betweent two pooling layers
    num_poolings : number of pooling layers
    use_bn : use batch normalization
    use_dc : use deconvolution to upsample the image, otherwise use image resize
    act : activation of the output layer usually sigmoid for image output and tanh for residual output"""
    
    if block_type == 'dense':
        block_type = {'down':'dense', 'center':'dense', 'up':'dense'}
    elif block_type == 'res':
        block_type = {'down':'res', 'center':'res', 'up':'res'}
    elif block_type == 'conv':
        block_type = {'down':'conv', 'center':'conv', 'up':'conv'}
    else:
        raise Exception('block type error')


    down = lambda x, oc, bac, name: DownSampling2D(x, scale = 2, out_channel = oc, method = method_down, 
                                              act = act, bn = use_bn, is_train = is_train, 
                                              BAC = bac, name = name) 
    
    up = lambda x, oc, bac, name: UpSampling2D(x, scale = 2, out_channel = oc, method = method_up, 
                                          act = act, bn = use_bn, is_train = is_train, 
                                          BAC = bac, name = name)
    
    bac = {'dense':True, 'conv':False, 'res':False}

    block = {}
    out_channel = {}
    for key in ['center','down','up']:
        if block_type[key] == 'dense':
            block[key] = lambda x, i : dense_block(x, dense_depth, dense_growth_rate, act,
                                                   use_bn, is_train, 'dense_block{}'.format(i))
            out_channel[key] = lambda x, i: x.outputs.get_shape().as_list()[-1] // 2
        elif block_type[key] == 'res':
            block[key] = lambda x, i : residual_block(x, res_num, 2**(abs(i)-1) * res_channel_first, act,
                                                     use_bn, is_train, 'res_block{}'.format(i))
            #out_channel[key] = lambda x, i: 2**(abs(i)-1) * res_channel_first
            out_channel[key] = lambda x, i: None
        elif block_type[key] == 'conv':
            block[key] = lambda x, i : conv_block(x, conv_depth, 2**(abs(i)-1) * conv_channel_first, act,
                                                  use_bn, is_train, 'conv_block{}'.format(i))
            out_channel[key] = lambda x, i: None
            #out_channel[key] = lambda x, i: 2**(abs(i)-1) * conv_channel_first
    
    #if block_type['up'] != 'dense':
    #    out_channel['up'] = lambda x, i: None

    with tf.variable_scope(Name, reuse=reuse):
        set_name_reuse(reuse)
        
        # input
        inputs = InputLayer(x, name = 'input')
        encoder = inputs
        # encode
        encoders = []
        for i in xrange(1, num_downsampling+1):
            encoder = block['down'](encoder, i)
            encoders.append(encoder)
            encoder = down(encoder, out_channel['down'](encoder, i+1), bac[block_type['down']], 'down{}'.format(i))
            
        # center connection
        decoder = block['center'](encoder, (num_downsampling+1))
         
        # decode
        for i in xrange(num_downsampling, 0, -1):
            decoder = up(decoder, out_channel['up'](encoder, -i), bac[block_type['up']], 'up{}'.format(i))
            decoder = ConcatLayer([decoder, encoders[i-1]], 3, name = 'concat{}'.format(i))
            decoder = block['up'](decoder, -i)
         
        # output layer
        outputs = Conv2d(decoder, num_channel_out, (1, 1), act=act_out, name='output')
        
        # residual learnning
        if use_res:
            outputs = ResLayer(outputs, inputs, name = 'res_output')
                  
    return outputs

def AutoContextNet(x, is_train = True, reuse = False, use_bn = False,
                  num_stage = 3, num_conv_per_stage = 4):
    """Auto-context network for low-dose PET image reconstruction.
    
    reference
    ---------
    Xiang, L., Qiao, Y., Nie, D., An, L., Wang, Q., & Shen, D. (2017). 
    Deep Auto-context Convolutional Neural Networks for Standard-Dose PET Image Estimation from Low-Dose PET/MRI. Neurocomputing.
    
    parameters
    ----------
    """
    
    gamma_init = tf.random_normal_initializer(1., 0.02)
    if use_bn:
        bn = lambda x, name : BatchNormLayer(x, act = tf.nn.relu, is_train = is_train, gamma_init = gamma_init, name = name)
        conv_act = None
    else:
        bn = lambda x, name : x
        conv_act = tf.nn.relu
    
    with tf.variable_scope("AC", reuse=reuse):
        set_name_reuse(reuse)
        
        size_x = x.get_shape().as_list()[1]
        size_y = x.get_shape().as_list()[2]
        
        outputs = []
        
        for stage in range(num_stage):
            
            if stage == 0:
                input1 = InputLayer(x, name = 'input')
                conv = input1
            else:
                size_x -= num_conv_per_stage * 2
                size_y -= num_conv_per_stage * 2
                crop1   = CropLayer(input1, [size_x, size_y], name = 'crop_{0}'.format(stage))
                conv  = ConcatLayer([conv, crop1], 3, name='concat_{0}'.format(stage))
            for i in range(num_conv_per_stage - 1):
                conv = Conv2d(conv, 64, (3,3), (1,1), conv_act, 'VALID', name = 'conv{0}_{1}'.format(stage+1, i+1))
                conv = bn(conv, name = 'bn{0}_{1}'.format(stage+1, i+1))
            conv = Conv2d(conv, 1, (3,3), (1,1), tf.sigmoid, 'VALID', name = 'conv{0}_{1}'.format(stage+1, num_conv_per_stage))
            outputs.append(conv)
                
    return outputs
    
def GAN(z, x, G, D, G_arg = {}, D_arg_fake = {}, D_arg_real = {},
       is_train = True, reuse = False):
    
    G_arg.update({'is_train':is_train, 'reuse':reuse})
    D_arg_fake.update({'is_train':is_train, 'reuse':reuse})
    D_arg_real.update({'is_train':is_train, 'reuse':True})
    
    with tf.variable_scope("GAN", reuse=reuse):
        set_name_reuse(reuse)
    
        G_out = G(z, **G_arg)
        D_out_fake = D(G_out.outputs, **D_arg_fake)
        D_out_real = D(x, **D_arg_real)

    return G_out, D_out_fake, D_out_real
    
def FCNN_discriminator(x, structure = [8, 16, 32, 64],
                       is_train = True, reuse = False, use_bn = False):
    
    gamma_init = tf.random_normal_initializer(1., 0.02)
    if use_bn:
        bn = lambda x, name : BatchNormLayer(x, act = tf.nn.relu, is_train = is_train, gamma_init = gamma_init, name = name)
        conv_act = None
    else:
        bn = lambda x, name : x
        conv_act = tf.nn.relu
    
    with tf.variable_scope("FCNN_D", reuse=reuse):
        set_name_reuse(reuse)
        
        inputs = InputLayer(x, name = 'input')
        conv = inputs
        
        for layer in range(len(structure)):
            num_channel = structure[layer]
            conv = Conv2d(conv, num_channel, (3, 3), (2, 2), act=conv_act, name='conv{0}'.format(layer+1))
            conv = bn(conv, name = 'bn{0}'.format(layer+1))

    # Finalization a la "all convolutional net"
        conv = Conv2d(conv, num_channel, (3, 3), (1, 1), act=conv_act, name='conv{0}'.format(len(structure)+1))
        conv = bn(conv, name = 'bn{0}'.format(len(structure)+1))
        conv = Conv2d(conv, num_channel, (1, 1), (1, 1), act=conv_act, name='conv{0}'.format(len(structure)+2))
        conv = bn(conv, name = 'bn{0}'.format(len(structure)+2))
    
    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
        conv = Conv2d(conv, 1, (1, 1), (1, 1), act=None, name='conv{0}'.format(len(structure)+3))
        conv = ReduceMeanLayer(conv, name = 'output')
    return conv

def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 32#64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (3, 3), (2, 2), act=lrelu,
                padding='SAME', W_init=w_init, name='h0/c')

        net_h1 = Conv2d(net_h0, df_dim*2, (3, 3), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h1/bn')
        net_h2 = Conv2d(net_h1, df_dim*4, (3, 3), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h2/bn')
        net_h3 = Conv2d(net_h2, df_dim*8, (3, 3), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h3/bn')
        net_h4 = Conv2d(net_h3, df_dim*16, (3, 3), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h4/bn')
        net_h5 = Conv2d(net_h4, df_dim*32, (3, 3), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h5/bn')
        net_h6 = Conv2d(net_h5, df_dim*16, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='h6/bn')
        net_h7 = Conv2d(net_h6, df_dim*8, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train,
                gamma_init=gamma_init, name='h7/bn')
        net = Conv2d(net_h7, df_dim*2, (1, 1), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='res/bn')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train,
                gamma_init=gamma_init, name='res/bn2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train,
                gamma_init=gamma_init, name='res/bn3')
        net_h8 = ElementwiseLayer(layer=[net_h7, net],
                combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity,
                W_init = w_init, name='ho/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
        
    return net_ho, logits

def unet_old(x, is_train = True, reuse = False, 
         num_channel_out = 1, num_channel_first = 32, 
         num_conv_per_pooling = 2, num_poolings = 3, 
         use_bn = False, use_dc = False, use_res = True,use_concat = True, den_con = False, in_res = False,
         use_selu = False,
         act = tf.tanh, Name = 'unet'):
    """U-Net for image denoising and super resolution. A multi-scale encoder-decoder network with symmetric concatenate connection.
    The input images and output images must have the same size. Therefore, when this network is used in super resolution the input
    image must be upsampled first.
    
    Parameters
    ----------
    x : input tf tensor
    is_train : only bn layers will be affected by this flag
    reuse : reuse variables
    num_channel_out : number of output channels
    num_channel_first : number of channels of the first convolution layer
    num_conv_per_pooling : number of convolution layers betweent two pooling layers
    num_poolings : number of pooling layers
    use_bn : use batch normalization
    use_dc : use deconvolution to upsample the image, otherwise use image resize
    act : activation of the output layer usually sigmoid for image output and tanh for residual output"""
    
    # Batch Normalization
    gamma_init = tf.random_normal_initializer(1., 0.02)
    if use_bn:
        bn = lambda x, name : BatchNormLayer(x, act = lambda y: lrelu(y, 0.2), is_train = is_train, gamma_init = gamma_init, name = name)
        conv_act = None
    else:
        bn = lambda x, name : x
        if use_selu:
            conv_act = tl.activation.selu
        else:
            conv_act = tf.nn.relu
    
    # deconv or upsampling
    if use_dc:
        up = lambda x, out_channel, name : myDeConv2d(x, out_channel, (3, 3), (2, 2), name = name)
    else:
        up = lambda x, out_channel, name : UpSampling2dLayer(x, (2, 2), name = name)
    
    with tf.variable_scope(Name, reuse=reuse):
        set_name_reuse(reuse)
        
        # input
        inputs = InputLayer(x, name = 'input')
        conv1 = inputs
        for i in range(num_conv_per_pooling):
            conv1 = Conv2d(conv1, num_channel_first, (3, 3), act=conv_act, name='conv{0}_{1}'.format(1, i+1))
            conv1 = bn(conv1, name = 'bn{0}_{1}'.format(1, i))
        pool1 = MaxPool2d(conv1, (2, 2), name = 'pool{0}'.format(1))
        # encode
        convs = [inputs, conv1]
        pools = [inputs, pool1]
        list_num_features = [x.get_shape().as_list()[-1], num_channel_first]
        for i in range(1, num_poolings):
            conv_encoder = pools[-1]
            num_channel = num_channel_first*(2**(i-1))
            for j in range(num_conv_per_pooling):
                conv_encoder = Conv2d(conv_encoder, num_channel, (3, 3), act=conv_act, name='conv{0}_{1}'.format(i+1, j+1))
                conv_encoder = bn(conv_encoder, name = 'bn{0}_{1}'.format(i+1, j+1))
            if in_res:
                shortcut = Conv2d(pools[-1], num_channel, (1, 1), act=conv_act, name='shortcut{}'.format(i+1))
                conv_encoder = ElementwiseLayer([conv_encoder, shortcut], tf.add, name='residual_in{}'.format(i+1))
                #paddings = [[0,0],[0,0],[0,0],[0,conv_encoder.outputs.get_shape().as_list()[-1] - pools[-1].outputs.get_shape().as_list()[-1]]]
                #conv_encoder = ElementwiseLayer([conv_encoder,
                #                                 PadLayer(pools[-1], paddings = paddings, name = 'slice_en{}'.format(i+1))], 
                #                                tf.add, name='residual_en{}'.format(i+1))
            pool_encoder = MaxPool2d(conv_encoder, (2, 2), name = 'pool{0}'.format(i+1))
            pools.append(pool_encoder)
            convs.append(conv_encoder)
            list_num_features.append(num_channel)

        # center connection
        conv_center = Conv2d(pools[-1], list_num_features[-1] * 2, (3, 3), act=tf.nn.relu, name = 'conv_center')
        if in_res:
            shortcut = Conv2d(pools[-1], list_num_features[-1] * 2, (1, 1), act=conv_act, name='shortcut_center')
            conv_center = ElementwiseLayer([conv_center, shortcut], tf.add, name='residual_cen')
            #paddings = [[0,0],[0,0],[0,0],[0,conv_center.outputs.get_shape().as_list()[-1] - pools[-1].outputs.get_shape().as_list()[-1]]]
            #conv_center = ElementwiseLayer([conv_center,
            #                                PadLayer(pools[-1], paddings = paddings, name = 'slice_cen')],
            #                               tf.add, name='residual_cen')
        conv_decoders = [conv_center]

        # decode
        for i in xrange(1, num_poolings+1):
        #fro i in range(num_poolings, 0):
            up_decoder = up(conv_decoders[-1], list_num_features[-i], name = 'up_{0}'.format(num_poolings+1-i))
            if use_concat:
                tmp_list = [convs[-i]]
                if den_con:
                    for j in xrange(i + 1, num_poolings + 1):
                        tmp_list.append(MaxPool2d(convs[-j], (2*(j-i), 2*(j-i)), 
                                                  name = 'den_pool{}_{}'.format(num_poolings+1-i, num_poolings+1-j)))
                up_decoder = ConcatLayer([up_decoder] + tmp_list, 3, name='concat_{0}'.format(num_poolings+1-i))
            conv_decoder = up_decoder
            for j in xrange(num_conv_per_pooling):
                conv_decoder = Conv2d(conv_decoder, list_num_features[-i], (3, 3), act=conv_act, name='uconv{0}_{1}'.format(num_poolings+1-i, j+1))
                conv_decoder = bn(conv_decoder, name = 'ubn{0}_{1}'.format(num_poolings+1-i, j+1))
            if in_res:
                conv_decoder = ElementwiseLayer([conv_decoder, convs[-i]], tf.add, name='residual_de{}'.format(num_poolings+1-i))
            conv_decoders.append(conv_decoder)

        # output layer
        conv_decoder = conv_decoders[-1]
        conv_output = Conv2d(conv_decoder, num_channel_out, (1, 1), act=act, name='output')
        
        if use_res:
            if conv_output.outputs.get_shape().as_list()[-1] == x.get_shape().as_list()[-1]:
                x_ = x
            elif conv_output.outputs.get_shape().as_list()[-1] < x.get_shape().as_list()[-1]:
                x_ = x[:,:,:,:conv_output.outputs.get_shape().as_list()[-1]]
            else:
                raise Exception(".")
            conv_output = ElementwiseLayer([conv_output, InputLayer(x_, name = 'slice')], tf.add, name='residual')
                
    return conv_output


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model
    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else: # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool4')                               # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool5')                               # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
    return network, conv

def patchGAN_d(x, is_train = True, reuse = False, ndf=64, n_layers=3, Name = 'patchGAN_d'):

    with tf.variable_scope(Name, reuse=reuse):
        set_name_reuse(reuse)
        inputs = InputLayer(x, name = 'input')
        gamma_init = tf.random_normal_initializer(1., 0.02)
        conv = inputs
        for i in range(n_layers):
            conv = Conv2d(conv, min(2**i,8) * ndf, (4, 4), (2, 2), name='conv{0}'.format(i+1))
            conv = BatchNormLayer(conv, act = lambda y: lrelu(y, 0.2), is_train = is_train, gamma_init = gamma_init, name = 'bn{}'.format(i+1))
        conv = Conv2d(conv, min(2**n_layers,8) * ndf, (4, 4), (1, 1), name='conv{0}'.format(n_layers+1))
        conv = BatchNormLayer(conv, act = lambda y: lrelu(y, 0.2), is_train = is_train, gamma_init = gamma_init, name = 'bn{}'.format(n_layers+1))
        outputs = Conv2d(conv, 1, (4, 4), (1, 1), name='output')

    return outputs

def refinenet(x, is_train = True, reuse = False, 
         num_channel_out = 1, num_downsampling = 3, use_res = True, res_channel_first = 16,
         act_out = tf.tanh, Name = 'refinenet'):

    def refine_block(xs, num_channel, name = 'refine_block'):

        def chained_res_pool(x, num_channel, pool_stride = 5, conv_stride = 3, num_stage = 2, name = 'CRP'):
            with tf.variable_scope(name):
                y = ActivationLayer(x, act = tf.nn.relu, name = 'relu')
                outputs = y
                for i in range(num_stage):
                    y = MeanPool2d(y, filter_size=(pool_stride, pool_stride), strides=(1, 1), padding='SAME', name='maxpool{}'.format(i+1))
                    y = Conv2d(y, num_channel, (conv_stride, conv_stride), act=tf.identity, name='conv{}'.format(i+1))
                    outputs = ElementwiseLayer(layer=[outputs, y], combine_fn=tf.add, name='add{}'.format(i+1))
            return outputs

        def multiresolution_fusion(xs, num_channel, name='MRF'):
            with tf.variable_scope(name):
                for i, x in enumerate(xs):
                    y = Conv2d(x, num_channel, (3, 3), act=tf.identity, name='conv{}'.format(i+1))
                    if i == 0:
                        outputs = y
                    else:
                        y = UpSampling2dLayer(y, (2**i, 2**i), name = 'upsample{}'.format(i+1))
                        outputs = ElementwiseLayer(layer=[outputs, y], combine_fn=tf.add, name='add{}'.format(i+1))
            return outputs

        with tf.variable_scope(name):
            ys = [residual_block(x, 1, num_channel, is_train = is_train, BAC = True, name = 'res_block{}'.format(i+1)) for i, x in enumerate(xs)]
            y = multiresolution_fusion(ys, num_channel)
            y = chained_res_pool(y, num_channel)
            y = residual_block(y, 1, num_channel, is_train = is_train, BAC = True, name = 'output')
        return y
            

    with tf.variable_scope(Name, reuse=reuse):
        set_name_reuse(reuse)
        
        # input
        inputs = InputLayer(x, name = 'input')
        encoder = inputs
        # encode
        encoders = []
        encoder = residual_block(encoder, 1, res_channel_first, is_train = is_train, BAC = True, name = 'res_block{}'.format(1))
        encoders.append(encoder)
        for i in xrange(1, num_downsampling+1):
            encoder = MeanPool2d(encoder, (2, 2), name = 'down{}'.format(i))
            encoder = residual_block(encoder, 1, 2**i * res_channel_first, is_train = is_train, BAC = True, name = 'res_block{}'.format(i+1))
            encoders.append(encoder)
        
        # decode (refine)
        for i in xrange(num_downsampling + 1, 0, -1):
            if i == num_downsampling + 1:
                decoder = refine_block([encoders[i-1]], 2**(i-1) * res_channel_first, name = 'refine_block{}'.format(i))
            else:
                decoder = refine_block([encoders[i-1], decoder], 2**(i-1) * res_channel_first, name = 'refine_block{}'.format(i))
         
        # output layer
        outputs = Conv2d(decoder, num_channel_out, (1, 1), act=act_out, name='output')
        
        # residual learnning
        if use_res:
            outputs = ResLayer(outputs, inputs, name = 'res_output')
                  
    return outputs