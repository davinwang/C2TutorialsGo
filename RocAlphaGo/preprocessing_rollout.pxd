import ast
import time
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.stdlib cimport malloc, free
from go cimport GameState
from go_data cimport _BLACK, _EMPTY, _STONE, _LIBERTY, _CAPTURE, _FREE, _PASS, _HASHVALUE, Group, Locations_List, locations_list_destroy, locations_list_new

# type of tensor created
# char works but float might be needed later
ctypedef char tensor_type

# type defining cdef function
ctypedef int (*preprocess_method)(Preprocess, GameState, tensor_type[ :, ::1 ], int)


cdef class Preprocess:

    ############################################################################
    #   variables declarations                                                 #
    #                                                                          #
    ############################################################################

    # all feature processors
    # TODO find correct type so an array can be used
    cdef preprocess_method *processors

    # list with all features used currently
    # TODO find correct type so an array can be used
    cdef list  feature_list

    # output tensor size
    cdef int   output_dim

    # board size
    cdef char  size
    cdef short board_size

    # pattern dictionaries
    cdef dict  pattern_nakade
    cdef dict  pattern_response_12d
    cdef dict  pattern_non_response_3x3

    # pattern dictionary sizes
    cdef int   pattern_nakade_size
    cdef int   pattern_response_12d_size
    cdef int   pattern_non_response_3x3_size

    ############################################################################
    #   Tensor generating functions                                            #
    #                                                                          #
    ############################################################################

    cdef int get_board(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       A feature encoding WHITE BLACK and EMPTY on separate planes.
       plane 0 always refers to the current player stones
       plane 1 to the opponent stones
       plane 2 to empty locations
    """

    cdef int get_turns_since(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       A feature encoding the age of the stone at each location up to 'maximum'

       Note:
       - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
       - EMPTY locations are all-zero features
    """

    cdef int get_liberties(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       A feature encoding the number of liberties of the group connected to the stone at
       each location

       Note:
       - there is no zero-liberties plane; the 0th plane indicates groups in atari
       - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
       - EMPTY locations are all-zero features
    """

    cdef int get_ladder_capture(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       A feature wrapping GameState.is_ladder_capture().
       check if an opponent group can be captured in a ladder
    """

    cdef int get_ladder_escape(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       A feature wrapping GameState.is_ladder_escape().
       check if player_current group can escape ladder
    """

    cdef int get_sensibleness(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
    """

    cdef int get_legal(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       Zero at all illegal moves, one at all legal moves. Unlike sensibleness, no eye check is done
       not used??
    """

    cdef int zeros(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       Plane filled with zeros
    """

    cdef int ones(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       Plane filled with ones
    """

    cdef int colour(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       Value net feature, plane with ones if active_player is black else zeros
    """

    cdef int ko(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       Single plane encoding ko location
    """

    cdef int get_response(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       single feature plane encoding whether this location matches any of the response
       patterns, for now it only checks the 12d response patterns as we do not use the
       3x3 response patterns.
    """

    cdef int get_save_atari(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       A feature wrapping GameState.is_ladder_escape().
       check if player_current group can escape atari for at least one turn
    """

    cdef int get_neighbor(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       encode last move neighbor positions in two planes:
       - horizontal & vertical / direct neighbor
       - diagonal neighbor
    """

    cdef int get_nakade(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       A nakade pattern is a 12d pattern on a location a stone was captured before
       it is unclear if a max size of the captured group has to be considered and
       how recent the capture event should have been

       the 12d pattern can be encoded without stone colour and liberty count
       unclear if a border location should be considered a stone or liberty

       pattern lookup value is being set instead of 1
    """

    cdef int get_nakade_offset(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       A nakade pattern is a 12d pattern on a location a stone was captured before
       it is unclear if a max size of the captured group has to be considered and
       how recent the capture event should have been

       the 12d pattern can be encoded without stone colour and liberty count
       unclear if a border location should be considered a stone or liberty

       #pattern_id is offset
    """

    cdef int get_response_12d(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       Set 12d hash pattern for 12d shape around last move
       pattern lookup value is being set instead of 1
    """

    cdef int get_response_12d_offset(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       Set 12d hash pattern for 12d shape around last move where
       #pattern_id is offset
    """

    cdef int get_non_response_3x3(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       Set 3x3 hash pattern for every legal location where
       pattern lookup value is being set instead of 1
    """

    cdef int get_non_response_3x3_offset(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet)
    """
       Set 3x3 hash pattern for every legal location where
       #pattern_id is offset
    """

    ############################################################################
    #   public cdef function                                                   #
    #                                                                          #
    ############################################################################

    cdef np.ndarray[ tensor_type, ndim=4 ] generate_tensor(self, GameState state)
    """
       Convert a GameState to a Theano-compatible tensor
    """
