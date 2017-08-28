import sgf
from go import GameState, BLACK, WHITE, EMPTY
from datetime import datetime

BOARD_POSITION = 'abcdefghijklmnopqrs'

def WriteBackSGF(game_state, history, i):
    parser = sgf.Parser()
    collection = sgf.Collection(parser)
    parser.start_gametree()
    parser.start_node()
    parser.start_property('FF') # SGF format version
    parser.add_prop_value('4')
    parser.end_property()
    parser.start_property('SZ') # Board Size = 19
    parser.add_prop_value('19')
    parser.end_property()
    parser.start_property('KM') # Komi = 7.5
    parser.add_prop_value('7.5')
    parser.end_property()
    parser.start_property('PB') # Black Player = Supervised Learning / Reinforced Learning
    parser.add_prop_value('RL-{}')
    parser.end_property()
    parser.start_property('PW') # White Player = Supervised Learning / Reinforced Learning
    parser.add_prop_value('SL-{}')
    parser.end_property()
    parser.start_property('DT') # Game Date
    parser.add_prop_value(datetime.now().strftime("%Y-%m-%d"))
    parser.end_property()
    parser.start_property('RE') # Result = B+, W+, T
    winner = game_state.get_winner()
    if winner == BLACK:
        parser.add_prop_value('B+')
        winner = 'B+'
    elif winner == WHITE:
        parser.add_prop_value('W+')
        winner = 'W+'
    else:
        parser.add_prop_value('T')
        winner = 'T'
    parser.end_property()
    parser.end_node()
    
    for step in history:
        parser.start_node()
        parser.start_property(step[0]) # or W
        parser.add_prop_value(BOARD_POSITION[step[1]]+BOARD_POSITION[step[2]])
        parser.end_property()
        parser.end_node()
    
    parser.end_gametree()
    
    # record the game in SGF
    with open(os.path.join(os.path.expanduser('~'), 'python', 'tutorial_files','selfplay',
                           '({}_{}_{})vs({}_{}_{})_{}_{}_{}.sgf'.format(
                               BLACK_CONV_LEVEL, BLACK_FILTERS, BLACK_PRE_TRAINED_ITERS,
                               WHITE_CONV_LEVEL, WHITE_FILTERS, WHITE_PRE_TRAINED_ITERS,
                               winner,
                               i,
                               datetime.now().strftime("%Y-%m-%d"))), "w") as f:
        collection.output(f)