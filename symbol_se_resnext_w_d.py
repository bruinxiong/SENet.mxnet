'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py 
Original author Wei Wu

Implemented the following paper:
Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He. "Aggregated Residual Transformations for Deep Neural Network" CVPR 2017 https://arxiv.org/pdf/1611.05431v2.pdf
Jie Hu, Li Shen, Gang Sun. "Squeeze-and-Excitation Networks" https://arxiv.org/pdf/1709.01507v1.pdf

This modification version is based on ResNet v1
This modificaiton version adds dropout layer followed by last pooling layer.
Modified by Lin Xiong Feb-11, 2017
Updated by Lin Xiong Jul-21, 2017
Added Squeeze-and-Excitation block by Lin Xiong Sep-13, 2017
'''
import mxnet as mx

def residual_unit(data, num_filter, ratio, stride, dim_match, name, num_group, bottle_neck=True,  bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNext Unit symbol for building ResNext
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    bottle_neck : Boolen
        Whether or not to adopt bottle_neck trick as did in ResNet
    num_group : int
        Number of convolution groupes
    bn_mom : float
        Momentum of batch normalization
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')        
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.5), num_group=num_group, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        squeeze = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_squeeze')
        squeeze = mx.symbol.Flatten(data=squeeze, name=name + '_flatten')
        excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter*ratio), name=name + '_excitation1')
        excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
        excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
        excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
        bn3 = mx.symbol.broadcast_mul(bn3, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise =  bn3 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')

        squeeze = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_squeeze')
        squeeze = mx.symbol.Flatten(data=squeeze, name=name + '_flatten')
        excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter*ratio), name=name + '_excitation1')
        excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
        excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
        excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
        bn2 = mx.symbol.broadcast_mul(bn2, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise = bn2 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
		
def resnext(units, num_stage, filter_list, ratio_list, num_class, num_group, data_type, drop_out, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNeXt symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    num_groupes: int
		Number of convolution groups
    drop_out : float
        Probability of an element to be zeroed. Default = 0.0
    data_type : str
        Dataset type, only cifar10, imagenet and vggface supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    elif data_type == 'vggface':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    elif data_type == 'msface':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
         raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], ratio_list[2], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), num_group=num_group, bottle_neck=bottle_neck,  
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], ratio_list[2], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 num_group=num_group, bottle_neck=bottle_neck, bn_mom=bn_mom, workspace=workspace, memonger=memonger)        
    pool1 = mx.symbol.Pooling(data=body, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    drop1= mx.symbol.Dropout(data=flat, p=drop_out, name='dp1')
    fc1 = mx.symbol.FullyConnected(data=drop1, num_hidden=num_class, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    