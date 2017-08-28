from caffe2.python import core, model_helper, brew, utils

def AddConvModel(model, data, conv_level=13, filters=192):
    # Layer 1: 48 x 19 x 19 -pad-> 48 x 23 x 23 -conv-> 192 x 19 x 19
    pad1 = model.PadImage(data, 'pad1', pad_t=2, pad_l=2, pad_b=2, pad_r=2, mode="constant", value=0.)
    conv1 = brew.conv(model, pad1, 'conv1', dim_in=48, dim_out=filters, kernel=5)
    relu1 = brew.relu(model, conv1, 'relu1')
    # Layer 2-12: 192 x 19 x 19 -pad-> 192 x 21 x 21 -conv-> 192 x 19 x 19
    if conv_level > 2:
        pad2 = model.PadImage(relu1, 'pad2', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv2 = brew.conv(model, pad2, 'conv2', dim_in=filters, dim_out=filters, kernel=3)
        relu2 = brew.relu(model, conv2, 'relu2')
        relu = relu2
    #
    if conv_level > 3:
        pad3 = model.PadImage(relu2, 'pad3', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv3 = brew.conv(model, pad3, 'conv3', dim_in=filters, dim_out=filters, kernel=3)
        relu3 = brew.relu(model, conv3, 'relu3')
        relu = relu3
    #
    if conv_level > 4:
        pad4 = model.PadImage(relu3, 'pad4', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv4 = brew.conv(model, pad4, 'conv4', dim_in=filters, dim_out=filters, kernel=3)
        relu4 = brew.relu(model, conv4, 'relu4')
        relu = relu4
    #
    if conv_level > 5:
        pad5 = model.PadImage(relu4, 'pad5', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv5 = brew.conv(model, pad5, 'conv5', dim_in=filters, dim_out=filters, kernel=3)
        relu5 = brew.relu(model, conv5, 'relu5')
        relu = relu5
    #
    if conv_level > 6:
        pad6 = model.PadImage(relu5, 'pad6', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv6 = brew.conv(model, pad6, 'conv6', dim_in=filters, dim_out=filters, kernel=3)
        relu6 = brew.relu(model, conv6, 'relu6')
        relu = relu6
    #
    if conv_level > 7:
        pad7 = model.PadImage(relu6, 'pad7', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv7 = brew.conv(model, pad7, 'conv7', dim_in=filters, dim_out=filters, kernel=3)
        relu7 = brew.relu(model, conv7, 'relu7')
        relu = relu7
    #
    if conv_level > 8:
        pad8 = model.PadImage(relu7, 'pad8', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv8 = brew.conv(model, pad8, 'conv8', dim_in=filters, dim_out=filters, kernel=3)
        relu8 = brew.relu(model, conv8, 'relu8')
        relu = relu8
    #
    if conv_level > 9:
        pad9 = model.PadImage(relu8, 'pad9', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv9 = brew.conv(model, pad9, 'conv9', dim_in=filters, dim_out=filters, kernel=3)
        relu9 = brew.relu(model, conv9, 'relu9')
        relu = relu9
    #
    if conv_level > 10:
        pad10 = model.PadImage(relu9, 'pad10', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv10 = brew.conv(model, pad10, 'conv10', dim_in=filters, dim_out=filters, kernel=3)
        relu10 = brew.relu(model, conv10, 'relu10')
        relu = relu10
    #
    if conv_level > 11:
        pad11 = model.PadImage(relu10, 'pad11', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv11 = brew.conv(model, pad11, 'conv11', dim_in=filters, dim_out=filters, kernel=3)
        relu11 = brew.relu(model, conv11, 'relu11')
        relu = relu11
    #
    if conv_level > 12:
        pad12 = model.PadImage(relu11, 'pad12', pad_t=1, pad_l=1, pad_b=1, pad_r=1, mode="constant", value=0.)
        conv12 = brew.conv(model, pad12, 'conv12', dim_in=filters, dim_out=filters, kernel=3)
        relu12 = brew.relu(model, conv12, 'relu12')
        relu = relu12
    # Layer 13: 192 x 19 x 19 -conv-> 1 x 19 x 19 -softmax-> 361
    conv13 = brew.conv(model, relu, 'conv13', dim_in=filters, dim_out=1, kernel=1)
    ## todo: bias layer?
    softmax = brew.softmax(model, conv13, 'softmax')
    predict = model.Flatten(softmax, 'predict')
    return predict

def AddAccuracy(model, predict, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [predict, label], "accuracy")
    return accuracy
	
def AddTrainingOperators(model, softmax, label, base_lr=-0.003):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    # AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=base_lr, policy="fixed", stepsize=1, gamma=0.999 ) # when policy=fixed, stepsize and gamma are ignored
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
		
def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.
    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """    
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)