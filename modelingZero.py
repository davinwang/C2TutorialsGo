﻿from caffe2.python import core, model_helper, brew, utils, workspace
from caffe2.proto import caffe2_pb2

def AddInput(model, batch_size, db, db_type):
    # Data is stored in INT8 while label is stored in INT32 and reward is stored in FLOAT
    # This will save disk storage
    data_int8, label_int32, reward_float = model.TensorProtosDBInput(
        [], ['data_int8', 'label_int32', 'reward_float'], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast data to float
    data = model.Cast(data_int8, 'data', to=core.DataType.FLOAT)
    label = model.Cast(label_int32, 'label', to=core.DataType.INT32)
    reward = model.Cast(reward_float, 'reward', to=core.DataType.FLOAT)
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    label = model.StopGradient(label, label)
    reward = model.StopGradient(reward, reward)
    return data, label, reward

def AddResNetModel(model, data, num_blocks=19, filters=256, dim_in=17, is_test=True):
    """
        params
            data: Data Input in shape of NCHW.
            num_blocks: Number of Residual Tower. Each block contains 2 convolution. Default 19
            filters: Number of output Channel of NCHW. Default 256
            dim_in: Number of input Channel of NCHW. Default 17. i.e.
        returns
            predict: unscaled prediction, need Softmax to translate to probabilities
            value: scaled value [-1,1]
    """
    # Layer 1: 17 x 19 x 19 -pad-> 17 x 21 x 21 -conv-> 256 x 19 x 19
    pad1 = model.PadImage(data, 'pad1', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
    conv1 = brew.conv(model, pad1, 'conv1', dim_in=dim_in, dim_out=filters, kernel=3)
    norm1 = brew.spatial_bn(model, conv1, 'norm1', filters, is_test=is_test)
    res_in = brew.relu(model, norm1, 'relu1')
    # Blocks: 256 x 19 x 19 -conv-> -normalize-> -relu-> -conv-> -normalize-> +INPUT -relu-> 256 x 19 x 19
    def AddResBlock(model, input, i, filters, scope='res'):
        pad1 = model.PadImage(input, '{}/{}/pad1'.format(scope,i), pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv1 = brew.conv(model, pad1, '{}/{}/conv1'.format(scope,i), dim_in=filters, dim_out=filters, kernel=3)
        norm1 = brew.spatial_bn(model, conv1, '{}/{}/norm1'.format(scope,i), filters, is_test=is_test)
        relu1 = brew.relu(model, norm1, '{}/{}/relu1'.format(scope,i))
        pad2 = model.PadImage(relu1, '{}/{}/pad2'.format(scope,i), pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv2 = brew.conv(model, pad2, '{}/{}/conv2'.format(scope,i), dim_in=filters, dim_out=filters, kernel=3)
        norm2 = brew.spatial_bn(model, conv2, '{}/{}/norm2'.format(scope,i), filters, is_test=is_test)
        res = model.Add([norm2, input], '{}/{}/res'.format(scope,i))
        output = brew.relu(model, res, '{}/{}/relu2'.format(scope,i))
        return output
    for i in range(num_blocks):
        res_out = AddResBlock(model, res_in, i, filters)
        res_in = res_out
    # Policy Head: 256 x 19 x 19 -conv-> 2 x 19 x 19 -normalize-> -relu-> -FC-> 362
    ph_conv1 = brew.conv(model, res_out, 'ph/conv1', dim_in=filters, dim_out=2, kernel=1)
    ph_norm1 = brew.spatial_bn(model, ph_conv1, 'ph/norm1', 2, is_test=is_test)
    ph_relu1 = brew.relu(model, ph_norm1, 'ph/relu1')
    ph_fc = brew.fc(model, ph_relu1, 'ph/fc', dim_in=2*19*19, dim_out=362)
    predict = model.Flatten(ph_fc, 'predict')
    # Value Head: 256 x 19 x 19 -conv-> 1 x 19 x 19 -> -normalize-> -relu-> -FC-> 256 x 19 x19 -relu-> -FC-> 1(scalar) -tanh->
    vh_conv1 = brew.conv(model, res_out, 'vh/conv1', dim_in=filters, dim_out=1, kernel=1)
    vh_norm1 = brew.spatial_bn(model, vh_conv1, 'vh/norm1', 1, is_test=is_test)
    vh_relu1 = brew.relu(model, vh_norm1, 'vh/relu1')
    vh_fc1 = brew.fc(model, vh_relu1, 'vh/fc1', dim_in=1*19*19, dim_out=filters*19*19)
    vh_relu2 = brew.relu(model, vh_fc1, 'vh/relu2')
    vh_fc2 = brew.fc(model, vh_relu2, 'vh/fc2', dim_in=filters*19*19, dim_out=1)
    vh_tanh = brew.tanh(model, vh_fc2, 'vh/tanh')
    value = model.FlattenToVec(vh_tanh, 'value')
    return predict, value

def AddSoftmax(model, predict):
    softmax = brew.softmax(model, predict, 'softmax')
    return softmax

def AddAccuracy(model, softmax, label, log=True):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    if log:
        model.Print('accuracy', [], to_file=1)
    return accuracy

def AddOneHot(model, label):
    """Decode onehot at modelling, not on the fly
    """
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
        BOARD_SIZE = model.param_init_net.ConstantFill([], 'BOARD_SIZE', shape=[1,], value=362) # constant
    label_int64 = model.Cast(label, 'label_int64', to=core.DataType.INT64)
    onehot = model.OneHot([label_int64, BOARD_SIZE], 'onehot') # shape=(mini_batch,362)
    onehot = model.StopGradient(onehot, onehot)
    return onehot

def AddTrainingOperators(model, predict, expect, value, reward, 
                         base_lr=-0.1, policy='fixed', stepsize=200000, gamma=0.1, log=True):
    """Adds training operators to the model.
        params
            predict: Predicted move by Policy Model, unscaled, in shape (N,362)
            label: Labelled move in shape (N,)
            expect: Expected distribution by MCTS, in shape (N,362)
            value: Predicted value by Value Model, scalar value in (-1,1)
            reward: Labelled value, scalar value in {-1,1}
            base_lr: Base Learning Rate. Policy is always fixed
            log: Whether to log the loss and accuracy in file, default True
    """
    _, xent = model.SoftmaxWithLoss([predict, expect], ['_', 'xent'], label_prob=1)
    #loss1 = model.AveragedLoss(xent, 'loss1')
    msqrl2 = model.AveragedLoss(model.SquaredL2Distance([value, reward], None), 'msqrl2')
    loss = model.Add([xent, msqrl2], 'loss')
    # use the average loss we just computed to add gradient operators to the model
    #model.AddGradientOperators([xent, loss2])
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(ITER, "LR", base_lr=base_lr, policy=policy, stepsize=stepsize, gamma=gamma)
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
    if log:
        model.Print('loss', [], to_file=1)
        model.Print('xent', [], to_file=1)
        model.Print('msqrl2', [], to_file=1)

def LoadParams(load_from):
    init_def = caffe2_pb2.NetDef()
    with open(load_from, 'r') as f:
        init_def.ParseFromString(f.read())
        #init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def.SerializeToString())

def SaveParams(model, save_to) :
    init_net = caffe2_pb2.NetDef()
    for param in model.params:
        blob = workspace.FetchBlob(param)
        shape = blob.shape
        op = core.CreateOperator("GivenTensorFill",
                                 [],
                                 [param],
                                 arg=[utils.MakeArgument("shape", shape),utils.MakeArgument("values", blob)])
        init_net.op.extend([op])
    with open(save_to, 'wb') as f:
        f.write(init_net.SerializeToString())
