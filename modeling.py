﻿from caffe2.python import core, model_helper, brew, utils

def AddInput(model, batch_size, db, db_type):
    # Data is stored in INT8 while label is stored in UINT16
    # This will save disk storage
    data_int8, label_uint16 = model.TensorProtosDBInput(
        [], ['data_int8', 'label_uint16'], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast data to float
    data = model.Cast(data_int8, 'data', to=core.DataType.FLOAT)
    # cast to int
    label_int32 = model.Cast(label_uint16, 'label_int32', to=core.DataType.INT32)
    label = model.FlattenToVec(label_int32, 'label')
    # encode onehot
    BOARD_SIZE = model.param_init_net.ConstantFill([], 'BOARD_SIZE', shape=[1,], value=361) # constant
    label_int64 = model.Cast(label, 'label_int64', to=core.DataType.INT64)
    onehot = model.OneHot([label_int64, BOARD_SIZE], 'onehot') # shape=(mini_batch,361)
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    label = model.StopGradient(label, label)
    onehot = model.StopGradient(onehot, onehot)
    return data, label, onehot

def AddConvModel(model, data, conv_level=13, filters=192, dim_in=48):
    # Layer 1: 48 x 19 x 19 -pad-> 48 x 23 x 23 -conv-> 192 x 19 x 19
    pad1 = model.PadImage(data, 'pad1', pad_t=2, pad_l=2, pad_b=2, pad_r=2, mode="constant", value=0.)
    conv1 = brew.conv(model, pad1, 'conv1', dim_in=dim_in, dim_out=filters, kernel=5)
    input = brew.relu(model, conv1, 'relu1')
    # Layer 2-12: 192 x 19 x 19 -pad-> 192 x 21 x 21 -conv-> 192 x 19 x 19
    def AddConvLevel(model, input, i, filters):
        pad = model.PadImage(input, 'pad{}'.format(i), pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv = brew.conv(model, pad, 'conv{}'.format(i), dim_in=filters, dim_out=filters, kernel=3)
        relu = brew.relu(model, conv, 'relu{}'.format(i))
        return relu
    for i in range(2, conv_level):
        output = AddConvLevel(model, input, i, filters)
        input = output
    # Layer 13: 192 x 19 x 19 -conv-> 1 x 19 x 19 -softmax-> 361
    conv13 = brew.conv(model, output, 'conv13', dim_in=filters, dim_out=1, kernel=1)
    softmax = brew.softmax(model, conv13, 'softmax')
    predict = model.Flatten(softmax, 'predict')
    return predict

def AddAccuracy(model, predict, label, log=True):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [predict, label], "accuracy")
    if log:
        model.Print('accuracy', [], to_file=1)
    return accuracy

def AddTrainingOperators(model, predict, expect, base_lr=-0.003, log=True):
    """Adds training operators to the model.
        predict: Predicted distribution by Policy Model
        expect: Expected distribution by MCTS, or transformed from Policy Model
        base_lr: Base Learning Rate. Always fixed
    """
    xent = model.SigmoidCrossEntropyWithLogits([predict, expect], 'xent')
    #xent = model.SquaredL2Distance([predict, expect], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(ITER, "LR", base_lr=base_lr, policy="fixed") # when policy=fixed, stepsize and gamma are ignored
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
