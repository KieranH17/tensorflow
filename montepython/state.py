import chess
from sys import maxsize
from collections import defaultdict


class State:
    def __init__(self, old_state_and_action=None):
        """Given an old state and action in a tuple, this will create the resulting state; otherwise,
        it will create the beginning board state."""
        if not old_state_and_action:
            self.board = chess.Board()

            self.legal_actions = [move.uci() for move in self.board.legal_moves]

            self.my_pieces = get_piece_planes(self.board, self.board.turn)
            self.opp_pieces = get_piece_planes(self.board, self.board.turn)

            self.rep_count = defaultdict(int)
            self.rep_count[self.board.epd()] += 1

            self.turn = self.board.turn
            self.full_mc = self.board.fullmove_number
            self.my_castling = get_bit_castles(self.board, self.turn)
            self.opp_castling = get_bit_castles(self.board, not self.turn)
            self.np_count = 0

        else:
            old_state = old_state_and_action[0]
            move = chess.Move.from_uci(old_state_and_action[1])
            move_start_ind = move.from_square
            move_end_ind = move.to_square
            piece_type_taken = old_state.board.piece_type_at(move_end_ind)
            piece_type_moved = old_state.board.piece_type_at(move_start_ind)

            self.my_pieces = []
            for i in range(len(old_state.opp_pieces)):
                self.my_pieces.append(old_state.opp_pieces[i][::-1])

            self.opp_pieces = []
            for i in range(len(old_state.my_pieces)):
                self.opp_pieces.append(old_state.my_pieces[i][::-1])

            update_piece_planes(self, old_state, move)

            self.board = old_state.board.copy()
            self.board.push(move)

            self.legal_actions = [move.uci() for move in self.board.legal_moves]

            self.rep_count = old_state.rep_count.copy()
            self.rep_count[self.board.epd()] += 1

            self.turn = self.board.turn
            self.full_mc = self.board.fullmove_number

            self.my_castling = get_bit_castles(self.board, self.turn)
            self.opp_castling = get_bit_castles(self.board, not self.turn)

            if piece_type_moved == chess.PAWN or piece_type_taken:
                self.np_count = 0
            else:
                self.np_count = old_state.np_count + 1 / 2

    def is_terminal(self):
        if self.np_count > 49.5 or\
                self.rep_count[self.board.epd()] > 2 or\
                self.board.is_game_over():
            return True
        return False


class StateHistoryKeeper:
    def __init__(self):
        self.curr_state = State()
        self.state_history_list = [self.curr_state]

    def update_state_history(self, action):
        self.curr_state = State((self.curr_state, action))
        self.state_history_list.append(self.curr_state)

    def get_state_history(self):
        return self.state_history_list


def get_state_after_opening(opening_actions):
    """Returns the state after an opening_actions
    (a list of actions from the beginning of the game)."""
    curr_state = State()
    for action in opening_actions:
        curr_state = State((curr_state, action))
    return curr_state


# called after board perspective switches
def update_piece_planes(new_state, old_state, move):
    start_ind = move.from_square
    end_ind = move.to_square
    piece = old_state.board.piece_at(start_ind)
    if move.promotion:
        new_piece_type = move.promotion
    else:
        new_piece_type = piece.piece_type

    new_state.opp_pieces[new_piece_type - 1][63 - start_ind] = 0
    new_state.opp_pieces[new_piece_type - 1][63 - end_ind] = 1
    for i in range(NUM_PIECE_TYPES):
        new_state.my_pieces[i][63 - end_ind] = 0


def get_piece_planes(board, turn):
    pawn_list = [0] * 64
    knight_list = [0] * 64
    bishop_list = [0] * 64
    rook_list = [0] * 64
    queen_list = [0] * 64
    king_list = [0] * 64
    player_piece_lists = [pawn_list, knight_list, bishop_list, rook_list, queen_list, king_list]

    for i in range(NUM_PIECE_TYPES):
        piece_indices = board.pieces(i+1, turn)
        for j in piece_indices:
            player_piece_lists[i][j] = 1

    return player_piece_lists


def get_bit_castles(board, turn):
    castles = [0] * 2
    if board.has_kingside_castling_rights(turn):
        castles[0] = 1
    if board.has_queenside_castling_rights(turn):
        castles[1] = 1
    return castles


NUM_PIECE_TYPES = 6
MAX_VALUE = maxsize
MIN_VALUE = - MAX_VALUE