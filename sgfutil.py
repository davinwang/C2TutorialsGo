import sgf, os
from go import GameState, BLACK, WHITE, EMPTY
from datetime import datetime

# BOARD_POSITION contains SGF symbol which represents each row (or column) of the board
# It can be used to convert between 0,1,2,3... and a,b,c,d...
# Symbol [tt] or [] represents PASS in SGF, therefore is omitted
BOARD_POSITION = 'abcdefghijklmnopqrs'

def GetWinner(game_state):
    winner = game_state.get_winner()
    if winner == BLACK:
        return 'B+'
    elif winner == WHITE:
        return 'W+'
    else:
        return 'T'

def WriteBackSGF(winner, history, filename, PB=None, PW=None, Komi='7.5'):
    parser = sgf.Parser()
    collection = sgf.Collection(parser)
    # game properties
    parser.start_gametree()
    parser.start_node()
    parser.start_property('FF') # SGF format version
    parser.add_prop_value('4')
    parser.end_property()
    parser.start_property('SZ') # Board Size = 19
    parser.add_prop_value('19')
    parser.end_property()
    parser.start_property('KM') # default Komi = 7.5
    parser.add_prop_value(str(Komi))
    parser.end_property()
    parser.start_property('PB') # Black Player = Supervised Learning / Reinforced Learning
    parser.add_prop_value('RL-{}'.format(PB))
    parser.end_property()
    parser.start_property('PW') # White Player = Supervised Learning / Reinforced Learning
    parser.add_prop_value('SL-{}'.format(PW))
    parser.end_property()
    parser.start_property('DT') # Game Date
    parser.add_prop_value(datetime.now().strftime("%Y-%m-%d"))
    parser.end_property()
    parser.start_property('RE') # Result = B+, W+, T
    parser.add_prop_value(str(winner))
    parser.end_property()
    parser.end_node()
    # start of game
    for step in history:
        parser.start_node()
        parser.start_property(step[0]) # or W
        parser.add_prop_value(BOARD_POSITION[step[1]/19]+BOARD_POSITION[step[1]%19])
        parser.end_property()
        parser.end_node()
    # end of game
    parser.end_gametree()
    # record the game in SGF
    with open(os.path.join('{}.sgf'.format(filename)), "w") as f:
        collection.output(f)
