# coding=UTF-8
import numpy as np
PASS = -1
EMPTY = 2
WHITE = 3
BLACK = 4

'''
      | Feature         | # of planes | Description
      |-----------------|-------------|-------------------------
      | Stone colour    | 3           | Player stone / opponent stone / empty
      | Ones            | 1           | A constant plane filled with 1
      | Turns since     | 8           | How many turns since a move was played
      | Liberties       | 8           | Number of liberties (empty adjacent points)
      | Capture size    | 8           | How many opponent stones would be captured
      | Self-atari size | 8           | How many of own stones would be captured
      | Liberties after move | 8      | Number of liberties after this move is played
      | Ladder capture  | 1           | Whether a move at this point is a successful ladder capture
      | Ladder escape   | 1           | Whether a move at this point is a successful ladder escape
      | Sensibleness    | 1           | Whether a move is legal and does not fill its own eyes
      | Zeros           | 1           | A constant plane filled with 0
      | Player color    | 1           | Whether current player is black (only for Value Network)
'''
DEFAULT_FEATURES = [
    "board", "ones", "turns_since", "liberties", "capture_size",
    "self_atari_size", "liberties_after", "ladder_capture", "ladder_escape",
    "sensibleness", "zeros"]

def UpdateGroups(model, stones, liberties, captures, data, label, player):
    '''
      stones: shape (mini_batch, 361, 361) type=BOOL
      | Features        | # of planes |
      |-----------------|-------------|
      | Stones          | 1           |
      | Liberties       | 1           |
      | Captures        | 1           |

      data: shape (mini_batch,48,19,19)
      label: type blob, shape (mini_batch,)
      player: BLACK or WHITE
    '''
    black = model.ConstantFill([], 'BLACK', shape=[1,], value=BLACK)
    white = model.ConstantFill([], 'WHITE', shape=[1,], value=WHITE)

    # group of current position = self + intersect(union of neighbor groups, self stones)
    g_left = momodel.Slice([stones, [label-1,0], [label-1,-1]], 'g_left') # if label-1<0 should be null
    g_right = momodel.Slice([stones, [label-1,0], [label-1,-1]], 'g_right') # if label+1>360 should be null
    g_up = model.Slice([stones, [label-19,0], [label-19,-1]], 'g_up') # if label-19<0 should be null
    g_down = model.Slice([stones, [label+19,0], [label+19,-1]], 'g_down') # if label+19>360 should be null
    # union of neighbor groups
    g_self = model.Or([g_left, g_right], 'g_self')
    g_self = model.Or([g_self, g_up], 'g_self')
    g_self = model.Or([g_self, g_down], 'g_self')
    # board0 contains all player stones
    board0 = model.Slice([data, [N,0,0,0], [N,0,19,19]], 'board0')
    g_self = model.And([g_self, board0], 'g_self')
    # onehot to get self
    onehot = model.Cast(onehot, 'onehotb', to=BOOL)
    # group of current position
    g_self = model.Or([g_self, onehotb], 'g_self')

    # liberties of current group = SUM(liberties of neighbor) - 4 + 2 * SUM(liberties of self)
    # board2 contains all empty
    board2 = model.Slice([data, [N,2,0,0],[N,2,19,19]], 'board2')
    # liberties of self can be counted from board2
    l_self = model.Add(board2[label-1], board2[label+1], board2[label-19], board2[label+19], 'l_self')
    # liberties of neighbor can be counted from liberties
    l_neighbor = model.Add(liberties[label-1], liberties[label+1], liberties[label-19], liberties[label+19], 'l_neighbor')
    l_self = 2 * l_self + l_neighbor - 4
    
    # liberties of neighbor opponent = 
    model.Substract(neighbor,1) # only if neighbor is independant group
    
    # Captures of current move = 
    c_self = None
    
    # all stones in current group will update
    indices = model.LengthsRangeFill([361,], 'indices') # [0,1,...360]
    indices = model.BooleanMask([indices, g_self], 'indices')
    #
    stones = model.ScatterAssign([stones, indices, slice], g_self) # update inplace
    liberties = model.ScatterAssign([liberties, indices, slice], l_self) 
    captures = model.ScatterAssign([captures, indices, slice], c_self)

    return stones, liberties, captures

def UpdateLiberties(model, groups_after, data, label, player, batch_size=64):
    '''
      groups_after: shape (mini_batch, 19x19, 19x19) type=BOOL
    '''
    neighbors = np.zeros((19,19,21,21), dtype=np.bool) # constant represents neighbors, including borders
    for i in range(19):
        for j in range(19):
            neighbors[i, j, i, j+1] = True   # ◌ ◌ ● ◌ ◌
            neighbors[i, j, i+1, j] = True   # ◌ ● ◌ ● ◌
            neighbors[i, j, i+1, j+2] = True #
            neighbors[i, j, i+2, j+1] = True # ◌ ◌ ● ◌ ◌
    # remove borders (19,19,21,21) => (19,19,19,19)
    neighbors = np.delete(neighbors, [0,20], axis=2)
    neighbors = np.delete(neighbors, [0,20], axis=3)
    NEIGHBORS = model.GivenTensorBoolFill([], 'neighbors', shape=[batch_size,361,361], values=neighbors) # 
    #
    INDICES = model.LengthsRangeFill([361]*batch_size, 'indices') # N*[0,1,...360]
    
    current_group = model.BooleanMask([INDICES, groups_after[label]], 'current_group') # (N,361)
    group_neighbors = model.Or(NEIGHBORS[current_group], 'group_neighbors' ,axis=1) # (N,?)
    empties = model.Slice([data, [N,2,0,0],[N,2,19,19]], 'empties') # all empties on board[2]
    liberties_pos = model.And([group_neighbors, empties], 'liberties_pos') # (N,361)
    liberties_count = model.countTrue(liberties_pos, 'liberties_count', axis=1) # (N,)
    liberties_after = groups
    return liberties_after

def UpdateGameStatus(model, data, predict):
    ''' UpdateGameStatus
        It does not consider symmetric, all games are treated independantly.
        Input: data with shape (N, C, H, W)
               predict with shape (N, C, H, W)
        Output: data with shape (N, C, H, W)
    '''
    BOARD_SIZE = model.ConstantFill([], 'board_size', shape=[1,], value=361) # constant
    SPLIT_SIZE = model.GivenTensorIntFill([], 'split_size', shape=[15,], values=np.array([1,1,1,1,6,1,1,8,8,8,8,1,1,1,1])) # constant
    
    board0, board1, board2, ones3, \
    turns_since4to9, turns_since10, turns_since11, liberties12to19, \
    capture_size20to27, self_atari_size28to35, liberties_after36to43, \
    ladder_capture44, ladder_escape45, sensibleness46, zeros47 = model.Split([data, SPLIT_SIZE], \
                                                       ['board0', 'board1', 'board2','ones3', \
                                                        'turns_since4to9', 'turns_since10', 'turns_since11', 'liberties12to19', \
                                                        'capture_size20to27', 'self_atari_size28to35', 'liberties_after36to43', \
                                                        'ladder_capture44', 'ladder_escape45', 'sensibleness46', 'zeros47'], \
                                                       axis=1)
    
    _topk, topk_indices = model.TopK(predict, ['_topk', 'topk_indices'], k=1) #shape=(mini_batch,1)
    label = model.FlattenToVec([topk_indices], ['label']) # shape=(mini_batch,)
    
    onehot2d = model.OneHot([label, BOARD_SIZE], 'onehot2d') # shape=(mini_batch,361)
    onehot, _shape = model.Reshape(['onehot2d'], ['onehot', '_shape'], shape=(0,1,19,19)) #shape=(mini_batch,1,19,19)
    
    ## board
    # player of this turn = opponent of last turn
    board0n = board1
    # opponent of this turn = player of last turn
    board1n = model.Add([board0, onehot], 'board1n')
    # empty
    board2n = model.Sub([board2, onehot], 'board2n')
    ## ones
    ones3n = ones3 # all ONE
    ## turns since --- age the stones
    # for new move set age = 0
    turns_since4n = onehot
    # for age in [1..6] set age += 1
    turns_since5to10n = turns_since4to9
    # for age >= 7 set age = 8
    turns_since11n = model.Add([ turns_since10,  turns_since11], ' turns_since11n')
    # liberties = liberties after move of last move
    liberties12to19n = liberties_after36to43
    # TBD: 
    capture_size20to27n = capture_size20to27
    # TBD:
    self_atari_size28to35n = self_atari_size28to35
    # TBD: liberties after move
    liberties_after36to43n = liberties_after36to43
      # after this move, this stone (not group) has N vacant neighbor (N=0..3)
      # for neighbor opponent group, minus 1 liberties
      # if opponent group reaches 0 liberties, remove the stones
      # for neighbor self group, plus N-1 liberties
    # TBD: 
    ladder_capture44n = ladder_capture44
    ladder_escape45n = ladder_escape45
    sensibleness46n = board2n
    ## zeros
    zeros47n = zeros47
    ###
    data, _dim = model.Concat([board0n, board1n, board2n, ones3n, \
                               turns_since4n, turns_since5to10n, turns_since11n, liberties12to19n, \
                               capture_size20to27n, self_atari_size28to35n, liberties_after36to43n, \
                               ladder_capture44n, ladder_escape45n, sensibleness46n, zeros47n], \
                              ['data','_dim'], axis=1)
    return data

#def InitGame(model, mini_batch=64):
#  ZERO = np.zeros((mini_batch,1,19,19), dtype=np.float32)
#  ONE = np.ones((mini_batch,1,19,19), dtype=np.float32)
#  init_data = np.concatenate((ZERO,ZERO,ONE,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,ONE), axis=1)
#  workspace.FeedBlob("data", init_data)
#  
#  model = model_helper.ModelHelper(name="model", arg_scope={"order": "NCHW"}, init_params=True)
#  AddConvModel(model, "data", dim_in=13)
#  AddGamePlay(model, "data", "predict", mini_batch=mini_batch)
#  
#  workspace.RunNetOnce(model.param_init_net)
#  workspace.CreateNet(model.net, overwrite=True)
#  workspace.RunNet(model.net)
#  
#  init_move = np.reshape(workspace.FetchBlob('predict')[0], (-1)) # shape=(361,)
#  top_choice = np.argsort(-init_move)[0:mini_batch] # the top K step
#  
#  for i in range(mini_batch):
#      x = top_choice[i]/19
#      y = top_choice[i]%19
#      init_data[i,1,x,y] = 1 # opponent plus (x,y)
#      init_data[i,2,x,y] = 0 # empty minus (x,y)
#      init_data[i,4,x,y] = 1 # last 1 step plus (x,y)
#      init_data[i,12] = -1
#  
#  workspace.FeedBlob("data", init_data)
#  return data

#def Symmetric(model, predict):
#  ''' Symmetric is optional
#      Input: predict with shape (N*8, C, H, W)
#      Output: symm_predict with shape (N*8, C, H, W)
#  '''
#  # Unify
#  symm0, symm1, symm2, symm3, \
#    symm4, symm5, symm6, symm7 = model.Split([predict], ['symm0', 'symm1', 'symm2', 'symm3',
#                                                         'symm4', 'symm5', 'symm6', 'symm7'], axis=0)
#  symm0u = symm0
#  symm1u = model.Flip(symm1, axes(3))
#  symm2u = model.Flip(symm2, axes=(2))
#  symm3u = model.Flip(symm3, axes=(2,3))
#  symm4u = model.Transpose(symm4, axes=(0,1,3,2))
#  symm5u = model.Flip(symm5, axes=(3))
#  symm6u = model.Flip(symm6, axes=(2))
#  symm7u = model.Flip(symm7, axes=(2,3))
#  # Average
#  unify_predict = model.avg(symm0r, symm1r, ... symm7r)
#  # Diversify
#  symm0d = model.Reshape(unify_predict, Nx1x19x19)
#  symm1d = model.Flip(symm0d, axes=(3))
#  symm2d = model.Flip(symm0d, axes=(2))
#  symm3d = model.Flip(symm0d, axes=(2,3))
#  symm4d = model.Transpose(symm0d, axes=(0,1,3,2))
#  symm5d = model.Flip(symm4d, axes=(3))
#  symm6d = model.Flip(symm4d, axes=(2))
#  symm7d = model.Flip(symm4d, axes=(2,3))
#  # shape(symm_predict) = [N*8,C,H,W]
#  symm_predict = model.concatenate(symm0, ... symm7)
#  return symm_predict

