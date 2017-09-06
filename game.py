

def AddPreprocess(model, data, predict):
	board_size = model.ConstantFill([], 'board_size', shape=[1,], value=361)
	onehot = model.OneHot(predict, board_size)
	layer0, layer1, layer2, ... = model.segment(data, .....)
	###
	layer0n = layer1 # this turn player = last turn opponent
	layer1n = model.Add(layer0, onehot)
	layer2n = model.Minus(layer2, onehot) # and need to calculate taken
	layer3n = layer3 # all ONE
	###
	layer4n = onehot # 1 turns since last move
	layer5n = layer4
	layer6n = layer5
	layer7n = layer6
	layer8n = layer7
	layer9n = layer8
	layer10n = layer9
	layer11n = model.Add(layer10, layer11)
	data = model.concatenate(layer0n, layer1n, layer2n, ...layer11n)
	return data

def Unify(model, predict)
	symm0, symm1, ... symm7 = model.Segment(predict)
	symm0r = symm0
	symm1r = model.Flip(symm1, axes(3))
	symm2r = model.Flip(symm2, axes=(2))
	symm3r = model.Flip(symm3, axes=(2,3))
	symm4r = model.Transpose(symm4, axes=(0,1,3,2))
	symm5r = model.Flip(symm5, axes=(3))
	symm6r = model.Flip(symm6, axes=(2))
	symm7r = model.Flip(symm7, axes=(2,3))
	unify_predict = model.avg(symm0r, symm1r, ... symm7r)
	return unify_predict
	
def Symmetric(model, unify_predict)
	''' shape(predict) = [N,361] '''
	symm0 = model.Reshape(unify_predict, '_' Nx1x19x19)
	symm1 = model.Flip(symm0, axes=(3))
	symm2 = model.Flip(symm0, axes=(2))
	symm3 = model.Flip(symm0, axes=(2,3))
	symm4 = model.Transpose(symm0, axes=(0,1,3,2))
	symm5 = model.Flip(symm4, axes=(3))
	symm6 = model.Flip(symm4, axes=(2))
	symm7 = model.Flip(symm4, axes=(2,3))
	# shape(symm_predict) = [N*8,1,19,19]
	symm_predict = model.concatenate(symm0, ... symm7)
	return symm_predict


workspace.Switch(black)
workspace.RunNet(black_predict_model.net) # 16*8
# 'predict' generated, 
workspace.Switch(white)
workspace.RunNet(white_predict_model.net) # 16*8