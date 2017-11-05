"""
Inception-Resnet-v2, suitable for images with around 299 x 299

Implemented the following paper:
Szegedy C, Ioffe S, Vanhoucke V. Inception-v4, inception-resnet and the impact of residual connections on learning[J]. arXiv preprint arXiv:1602.07261, 2016.
Jie Hu, Li Shen, Gang Sun. "Squeeze-and-Excitation Networks" https://arxiv.org/pdf/1709.01507v1.pdf

This modification version is based on Inception-Resnet-v2 original but change to 224 x 224 size of input data.
Modified by Lin Xiong, Oct-31, 2017 for 224 x 224
Added Squeeze-and-Excitation block by Lin Xiong Oct-31, 2017
Thanks to Cher Keng Heng
"""

import mxnet as mx


def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix='', withRelu=True, withBn=False, bn_mom=0.9, workspace=256):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                              name='%s%s_conv2d' % (name, suffix), workspace=workspace)
    if withBn:
        conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%s%s_bn' % (name, suffix))
    if withRelu:
        conv = mx.sym.Activation(data=conv, act_type='relu', name='%s%s_relu' % (name, suffix))
    return conv


# Input Shape is 299*299*3 (old) 
# Input Shape is 224*224*3 (new)

def inception_resnet_stem(name, data, 
                          num_1_1=32, num_1_2=32, num_1_3=64, 
                          num_2_1=96, 
                          num_3_1=64, num_3_2=96, 
                          num_4_1=64, num_4_2=64, num_4_3=64, num_4_4=96, 
                          num_5_1=192, 
                          bn_mom=0.9):
    stem_3x3 = Conv(data=data, num_filter=num_1_1, kernel=(3, 3), stride=(2, 2), name=('%s_conv' % name), bn_mom=bn_mom, workspace=256)
    stem_3x3 = Conv(data=stem_3x3, num_filter=num_1_2, kernel=(3, 3), name=('%s_stem' % name), suffix='_conv', bn_mom=bn_mom, workspace=256)
    stem_3x3 = Conv(data=stem_3x3, num_filter=num_1_3, kernel=(3, 3), pad=(1, 1), name=('%s_stem' % name),
                    suffix='_conv_1', bn_mom=bn_mom, workspace=256)

    pool1 = mx.sym.Pooling(data=stem_3x3, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max',
                           name=('%s_%s_pool1' % ('max', name)))
    stem_1_3x3 = Conv(data=stem_3x3, num_filter=num_2_1, kernel=(3, 3), stride=(2, 2), name=('%s_stem_1' % name),
                      suffix='_conv_1', bn_mom=bn_mom, workspace=256)

    concat1 = mx.sym.Concat(*[pool1, stem_1_3x3], name=('%s_concat_1' % name))

    stem_1_1x1 = Conv(data=concat1, num_filter=num_3_1, name=('%s_stem_1' % name), suffix='_conv_2', bn_mom=bn_mom, workspace=256)
    stem_1_3x3 = Conv(data=stem_1_1x1, num_filter=num_3_2, kernel=(3, 3), name=('%s_stem_1' % name), suffix='_conv_3', bn_mom=bn_mom, workspace=256)

    stem_2_1x1 = Conv(data=concat1, num_filter=num_4_1, name=('%s_stem_2' % name), suffix='_conv_1', bn_mom=bn_mom, workspace=256)
    stem_2_7x1 = Conv(data=stem_2_1x1, num_filter=num_4_2, kernel=(7, 1), pad=(3, 0), name=('%s_stem_2' % name),
                      suffix='_conv_2', bn_mom=bn_mom, workspace=256)
    stem_2_1x7 = Conv(data=stem_2_7x1, num_filter=num_4_3, kernel=(1, 7), pad=(0, 3), name=('%s_stem_2' % name),
                      suffix='_conv_3', bn_mom=bn_mom, workspace=256)
    stem_2_3x3 = Conv(data=stem_2_1x7, num_filter=num_4_4, kernel=(3, 3), name=('%s_stem_2' % name), suffix='_conv_4', bn_mom=bn_mom, workspace=256)

    concat2 = mx.sym.Concat(*[stem_1_3x3, stem_2_3x3], name=('%s_concat_2' % name))

    pool2 = mx.sym.Pooling(data=concat2, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max',
                           name=('%s_%s_pool2' % ('max', name)))
    stem_3_3x3 = Conv(data=concat2, num_filter=num_5_1, kernel=(3, 3), stride=(2, 2), name=('%s_stem_3' % name),
                      suffix='_conv_1', withRelu=False, bn_mom=bn_mom, workspace=256)

    concat3 = mx.sym.Concat(*[pool2, stem_3_3x3], name=('%s_concat_3' % name))
    bn1 = mx.sym.BatchNorm(data=concat3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=('%s_bn1' % name))
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=('%s_relu1' % name))

    return act1
# Output Shape is 25*25*384

# Input Shape is 25*25*384
def Inception_Resnet_A(name, data,
                       num_1_1=32,
                       num_2_1=32, num_2_2=32,
                       num_3_1=32, num_3_2=48, num_3_3=64,
                       cat=384,
                       scaleResidual=True,
                       bn_mom=0.9):
    init = data

    a1 = Conv(data=data, num_filter=num_1_1, name=('%s_a_1' % name), suffix='_conv', bn_mom=bn_mom, workspace=256)

    a2 = Conv(data=data, num_filter=num_2_1, name=('%s_a_2' % name), suffix='_conv_1', bn_mom=bn_mom, workspace=256)
    a2 = Conv(data=a2, num_filter=num_2_2, kernel=(3, 3), pad=(1, 1), name=('%s_a_2' % name), suffix='_conv_2', bn_mom=bn_mom, workspace=256)

    a3 = Conv(data=data, num_filter=num_3_1, name=('%s_a_3' % name), suffix='_conv_1', bn_mom=bn_mom, workspace=256)
    a3 = Conv(data=a3, num_filter=num_3_2, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_2', bn_mom=bn_mom, workspace=256)
    a3 = Conv(data=a3, num_filter=num_3_3, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_3', bn_mom=bn_mom, workspace=256)

    m = mx.sym.Concat(*[a1, a2, a3], name=('%s_a_concat1' % name))

    conv = Conv(data=m, num_filter=cat, name=('%s_a_linear_conv' % name), withRelu=False, bn_mom=bn_mom, workspace=256)
    if scaleResidual:
        conv = conv.__mul__(0.1)  # for new version > 0.11.0
#         conv *= 0.1     # for old version < 0.11.0

    out = init + conv
    bn = mx.sym.BatchNorm(data=out, fix_gamma=False, eps=2e-5, name=('%s_a_bn1' % name))
    act = mx.sym.Activation(data=bn, act_type='relu', name=('%s_a_relu1' % name))

    return act
# Output Shape is 25*25*384

# Input Shape is 12*12*1152
def Inception_Resnet_B(name, data,
                       num_1_1=192,
                       num_2_1=128, num_2_2=160, num_2_3=192,
                       cat=1152,
                       scaleResidual=True,
                       bn_mom=0.9):
    init = data

    b1 = Conv(data=data, num_filter=num_1_1, name=('%s_b_1' % name), suffix='_conv', bn_mom=bn_mom, workspace=256)

    b2 = Conv(data=data, num_filter=num_2_1, name=('%s_b_2' % name), suffix='_conv_1', bn_mom=bn_mom, workspace=256)
    b2 = Conv(data=b2, num_filter=num_2_2, kernel=(1, 7), pad=(0, 3), name=('%s_b_2' % name), suffix='_conv_2', bn_mom=bn_mom, workspace=256)
    b2 = Conv(data=b2, num_filter=num_2_3, kernel=(7, 1), pad=(3, 0), name=('%s_b_2' % name), suffix='_conv_3', bn_mom=bn_mom, workspace=256)

    m = mx.sym.Concat(*[b1, b2], name=('%s_b_concat1' % name))

    conv = Conv(data=m, num_filter=cat, name=('%s_b_linear_conv' % name), withRelu=False, bn_mom=bn_mom, workspace=256)
    if scaleResidual:
        conv = conv.__mul__(0.1)  # for new version > 0.11.0
#         conv *= 0.1     # for old version < 0.11.0

    out = init + conv
    bn = mx.sym.BatchNorm(data=out, fix_gamma=False, eps=2e-5, name=('%s_b_bn1' % name))
    act = mx.sym.Activation(data=bn, act_type='relu', name=('%s_b_relu1' % name))

    return act
# Output Shape is 12*12*1152

# Input Shape is 5*5*2144
def Inception_Resnet_C(name, data,
                       num_1_1=192,
                       num_2_1=192, num_2_2=224, num_2_3=256,
                       cat=2144,
                       scaleResidual=True,
                       bn_mom=0.9):
    init = data

    c1 = Conv(data=data, num_filter=num_1_1, name=('%s_c_1' % name), suffix='_conv', bn_mom=bn_mom, workspace=256)

    c2 = Conv(data=data, num_filter=num_2_1, name=('%s_c_2' % name), suffix='_conv_1', bn_mom=bn_mom, workspace=256)
    c2 = Conv(data=c2, num_filter=num_2_2, kernel=(1, 3), pad=(0, 1), name=('%s_c_2' % name), suffix='_conv_2', bn_mom=bn_mom, workspace=256)
    c2 = Conv(data=c2, num_filter=num_2_3, kernel=(3, 1), pad=(1, 0), name=('%s_c_2' % name), suffix='_conv_3', bn_mom=bn_mom, workspace=256)

    m = mx.sym.Concat(*[c1, c2], name=('%s_c_concat1' % name))

    conv = Conv(data=m, num_filter=cat, name=('%s_c_linear_conv' % name), withRelu=False, bn_mom=bn_mom,workspace=256)
    if scaleResidual:
        conv = conv.__mul__(0.1)  # for new version > 0.11.0
#         conv *= 0.1     # for old version < 0.11.0

    out = init + conv
    bn = mx.sym.BatchNorm(data=out, fix_gamma=False, eps=2e-5, name=('%s_c_bn1' % name))
    act = mx.sym.Activation(data=bn, act_type='relu', name=('%s_c_relu1' % name))

    return act
# Output Shape is 5*5*2144

# Input Shape is 25*25*384
def ReductionA(name, data,
               num_2_1=384,
               num_3_1=256, num_3_2=256, num_3_3=384,
               bn_mom=0.9):
    ra1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    ra2 = Conv(data=data, num_filter=num_2_1, kernel=(3, 3), stride=(2, 2), name=('%s_ra_2' % name), suffix='_conv',
               withRelu=False, bn_mom=bn_mom, workspace=256)

    ra3 = Conv(data=data, num_filter=num_3_1, name=('%s_ra_3' % name), suffix='_conv_1', bn_mom=bn_mom, workspace=256)
    ra3 = Conv(data=ra3, num_filter=num_3_2, kernel=(3, 3), pad=(1, 1), name=('%s_ra_3' % name), suffix='_conv_2', bn_mom=bn_mom, workspace=256)
    ra3 = Conv(data=ra3, num_filter=num_3_3, kernel=(3, 3), stride=(2, 2), name=('%s_ra_3' % name), suffix='_conv_3',
               withRelu=False, bn_mom=bn_mom, workspace=256)

    m = mx.sym.Concat(*[ra1, ra2, ra3], name=('%s_ra_concat1' % name))
    m = mx.sym.BatchNorm(data=m, fix_gamma=False, eps=2e-5, name=('%s_ra_bn1' % name))
    m = mx.sym.Activation(data=m, act_type='relu', name=('%s_ra_relu1' % name))

    return m
# Output Shape is 12*12*1152

# Input Shape is 12*12*1152
def ReductionB(name, data, 
               num_2_1=256, num_2_2=384,
               num_3_1=256, num_3_2=288,
               num_4_1=256, num_4_2=288, num_4_3=320,
               bn_mom=0.9):
    rb1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    rb2 = Conv(data=data, num_filter=num_2_1, name=('%s_rb_2' % name), suffix='_conv_1', bn_mom=bn_mom, workspace=256)
    rb2 = Conv(data=rb2, num_filter=num_2_2, kernel=(3, 3), stride=(2, 2), name=('%s_rb_2' % name), suffix='_conv_2', 
               withRelu=False, bn_mom=bn_mom, workspace=256)

    rb3 = Conv(data=data, num_filter=num_3_1, name=('%s_rb_3' % name), suffix='_conv_1', bn_mom=bn_mom, workspace=256)
    rb3 = Conv(data=rb3, num_filter=num_3_2, kernel=(3, 3), stride=(2, 2), name=('%s_rb_3' % name), suffix='_conv_2', 
               withRelu=False, bn_mom=bn_mom, workspace=256)

    rb4 = Conv(data=data, num_filter=num_4_1, name=('%s_rb_4' % name), suffix='_conv_1', bn_mom=bn_mom, workspace=256)
    rb4 = Conv(data=rb4, num_filter=num_4_2, kernel=(3, 3), pad=(1, 1), name=('%s_rb_4' % name), suffix='_conv_2', bn_mom=bn_mom, workspace=256)
    rb4 = Conv(data=rb4, num_filter=num_4_3, kernel=(3, 3), stride=(2, 2), name=('%s_rb_4' % name), suffix='_conv_3',
               withRelu=False, bn_mom=bn_mom, workspace=256)
    
    m = mx.sym.Concat(*[rb1, rb2, rb3, rb4], name=('%s_rb_concat1' % name))
    m = mx.sym.BatchNorm(data=m, fix_gamma=False, eps=2e-5, name=('%s_rb_bn1' % name))
    m = mx.sym.Activation(data=m, act_type='relu', name=('%s_rb_relu1' % name))

    return m
# Output Shape is 5*5*2144

def squeeze_excitation_block(name, data, num_filter, ratio):
    squeeze = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_squeeze')
    squeeze = mx.symbol.Flatten(data=squeeze, name=name + '_flatten')
    excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter*ratio), name=name + '_excitation1')
    excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
    excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
    excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
    scale = mx.symbol.broadcast_mul(data, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))

    return scale


def circle_in5a(name, data, scale, ratio,
                num_1_1=32,
                num_2_1=32, num_2_2=32,
                num_3_1=32, num_3_2=48, num_3_3=64,
                cat=384,
                bn_mom=0.9,
                round=5):
    in5a = data
    for i in xrange(round):
        in5a = Inception_Resnet_A(name + ('_%d' % i), in5a,
                                  num_1_1,
                                  num_2_1, num_2_2,
                                  num_3_1, num_3_2, num_3_3,
                                  cat,
                                  scale,
                                  bn_mom)
        _, out_shapes, _ = in5a.infer_shape(data=(1, 3, 224, 224))
        # import pdb
        # pdb.set_trace()
        num_filter = int(out_shapes[0][1])
        in5a = squeeze_excitation_block(name + ('_%d' % i), in5a, num_filter, ratio)


    return in5a

def circle_in10b(name, data, scale, ratio,
                 num_1_1=192,
                 num_2_1=128, num_2_2=160, num_2_3=192,
                 cat=1152,
                 bn_mom=0.9,
                 round=10):
    in10b = data
    for i in xrange(round):
        in10b = Inception_Resnet_B(name + ('_%d' % i), in10b,
                                   num_1_1,
                                   num_2_1, num_2_2, num_2_3,
                                   cat,
                                   scale,
                                   bn_mom)
        _, out_shapes, _, = in10b.infer_shape(data=(1, 3, 224, 224))
        # import pdb
        # pdb.set_trace()
        num_filter = int(out_shapes[0][1])
        in10b = squeeze_excitation_block(name + ('_%d' % i), in10b, num_filter, ratio)
    return in10b

def circle_in5c(name, data, scale, ratio,
                num_1_1=192,
                num_2_1=192, num_2_2=224, num_2_3=256,
                cat=2144,
                bn_mom=0.9,
                round=5):
    in5c = data
    for i in xrange(round):
        in5c = Inception_Resnet_C(name + ('_%d' % i), in5c,
                                  num_1_1,
                                  num_2_1, num_2_2, num_2_3,
                                  cat,
                                  scale,
                                  bn_mom)
        _, out_shapes, _, = in5c.infer_shape(data=(1, 3, 224, 224))
        # import pdb
        # pdb.set_trace()
        num_filter = int(out_shapes[0][1])
        in5c = squeeze_excitation_block(name + ('_%d' % i), in5c, num_filter, ratio)
    return in5c

# create SE inception-resnet-v2
def get_symbol(ratio, num_classes=1000, scale=True):
    # input shape 229*229*3 (old)
    # input shape 224*224*3 (new)
    data = mx.symbol.Variable(name="data")
    bn_mom = 0.9

    # import pdb
    # pdb.set_trace()

    # stage stem
    (num_1_1, num_1_2, num_1_3) = (32, 32, 64)
    num_2_1 = 96
    (num_3_1, num_3_2) = (64, 96)
    (num_4_1, num_4_2, num_4_3, num_4_4) = (64, 64, 64, 96)
    num_5_1 = 192

    in_stem = inception_resnet_stem('stem_stage', data,
                                     num_1_1, num_1_2, num_1_3,
                                     num_2_1,
                                     num_3_1, num_3_2,
                                     num_4_1, num_4_2, num_4_3, num_4_4,
                                     num_5_1,
                                     bn_mom)

    # stage 5 x Inception-Resnet-A
    num_1_1 = 32
    (num_2_1, num_2_2) = (32, 32)
    (num_3_1, num_3_2, num_3_3) = (32, 48, 64)
    cat = 384

    in5a = circle_in5a('in5a', in_stem, scale, ratio,
                       num_1_1,
                       num_2_1, num_2_2,
                       num_3_1, num_3_2, num_3_3,
                       cat,
                       bn_mom,
                       5)

    # stage ReductionA
    num_1_1 = 384
    (num_2_1, num_2_2, num_2_3) = (256, 256, 384)

    re3a = ReductionA('re3a', in5a,
                      num_1_1,
                      num_2_1, num_2_2, num_2_3,
                      bn_mom)

    # stage 10 x Inception-Resnet-B
    num_1_1 = 192
    (num_2_1, num_2_2, num_2_3) = (128, 160, 192)
    cat = 1152

    in10b = circle_in10b('in10b', re3a, scale, ratio,
                         num_1_1,
                         num_2_1, num_2_2, num_2_3,
                         cat,
                         bn_mom,
                         10)

    # stage ReductionB
    (num_1_1, num_1_2) = (256, 384)
    (num_2_1, num_2_2) = (256, 288)
    (num_3_1, num_3_2, num_3_3) = (256, 288, 320)

    re4b = ReductionB('re4b', in10b,
                       num_1_1, num_1_2,
                       num_2_1, num_2_2,
                       num_3_1, num_3_2, num_3_3,
                       bn_mom)

    # stage 5 x Inception-Resnet-C
    num_1_1 = 192
    (num_2_1, num_2_2, num_2_3) = (192, 224, 256)
    cat = 2144

    in5c = circle_in5c('in5c', re4b, scale, ratio,
                        num_1_1,
                        num_2_1, num_2_2, num_2_3,
                        cat,
                        bn_mom,
                        5)

    # stage Average Pooling
    pool = mx.sym.Pooling(data=in5c, global_pool=True, kernel=(5, 5), stride=(1, 1), pad=(0, 0), pool_type="avg", name="global_pool")
    
    # stage Dropout
    dropout = mx.sym.Dropout(data=pool, p=0.2)       #original
    # dropout =  mx.sym.Dropout(data=pool, p=0.8)
    flatten = mx.sym.Flatten(data=dropout)

    # output
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax


# if __name__ == '__main__':
#     net = get_symbol(1000, scale=True)
#     shape = {'softmax_label': (32, 1000), 'data': (32, 3, 299, 299)}
#     mx.viz.plot_network(net, title='inception-resnet-v2', format='png', shape=shape).render('inception-resnet-v2')
