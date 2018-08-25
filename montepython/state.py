import chess
from sys import maxsize
from collections import defaultdict
import numpy as np


class State:
    def __init__(self, old_state_and_action=None):
        """Given an old state and action in a tuple, this will create the resulting state; otherwise,
        it will create the beginning board state."""
        if not old_state_and_action:
            self.board = chess.Board()

            self.my_pieces = {chess.WHITE: get_piece_planes(self.board, chess.WHITE),
                              chess.BLACK: get_piece_planes(self.board, chess.BLACK)}

            self.legal_actions = [move.uci() for move in self.board.legal_moves]

            self.rep_count = defaultdict(int)
            self.rep_count[self.board.epd()] += 1

            self.turn = self.board.turn
            self.full_mc = self.board.fullmove_number

            self.my_king_castling = self.board.has_kingside_castling_rights(self.turn)
            self.my_queen_castling = self.board.has_queenside_castling_rights(self.turn)
            self.opp_king_castling = self.board.has_kingside_castling_rights(not self.turn)
            self.opp_queen_castling = self.board.has_queenside_castling_rights(not self.turn)

            self.np_count = 0

        else:
            old_state = old_state_and_action[0]
            move = chess.Move.from_uci(old_state_and_action[1])
            move_start_ind = move.from_square
            move_end_ind = move.to_square
            piece_type_taken = old_state.board.piece_type_at(move_end_ind)
            piece_type_moved = old_state.board.piece_type_at(move_start_ind)

            self.board = old_state.board.copy()
            self.board.push(move)

            self.my_pieces = get_updated_pieces(old_state, move)

            self.legal_actions = [move.uci() for move in self.board.legal_moves]

            self.rep_count = old_state.rep_count.copy()
            self.rep_count[self.board.epd()] += 1

            self.turn = self.board.turn
            self.full_mc = self.board.fullmove_number

            self.my_king_castling = self.board.has_kingside_castling_rights(self.turn)
            self.my_queen_castling = self.board.has_queenside_castling_rights(self.turn)
            self.opp_king_castling = self.board.has_kingside_castling_rights(not self.turn)
            self.opp_queen_castling = self.board.has_queenside_castling_rights(not self.turn)

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

    def get_input_layers(self):

        white_piece_vectors = self.my_pieces[chess.WHITE]
        black_piece_vectors = self.my_pieces[chess.BLACK]

        my_king_castle_vector = [[int(self.my_king_castling)] * 8] * 8
        my_queen_castle_vector = [[int(self.my_queen_castling)] * 8] * 8
        opp_king_castle_vector = [[int(self.opp_king_castling)] * 8] * 8
        opp_queen_castle_vector = [[int(self.opp_queen_castling)] * 8] * 8
        rep_count_vector = [[self.rep_count[self.board.epd()]] * 8] * 8
        np_count_vector = [[self.np_count] * 8] * 8
        turn_vector = [[int(self.turn)] * 8] * 8
        full_mc_vector = [[self.full_mc] * 8] * 8

        input_vector = []

        input_vector.extend(np.asarray(white_piece_vectors).astype(np.float16))
        input_vector.extend(np.asarray(black_piece_vectors).astype(np.float16))

        input_vector.append(np.asarray(my_king_castle_vector).astype(np.float16))
        input_vector.append(np.asarray(my_queen_castle_vector).astype(np.float16))
        input_vector.append(np.asarray(opp_king_castle_vector).astype(np.float16))
        input_vector.append(np.asarray(opp_queen_castle_vector).astype(np.float16))
        input_vector.append(np.asarray(rep_count_vector).astype(np.float16))
        input_vector.append(np.asarray(np_count_vector).astype(np.float16))
        input_vector.append(np.asarray(turn_vector).astype(np.float16))
        input_vector.append(np.asarray(full_mc_vector).astype(np.float16))

        return input_vector


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


def get_piece_planes(board, color):
    pawn_list = get_8x8_zeros()
    knight_list = get_8x8_zeros()
    bishop_list = get_8x8_zeros()
    rook_list = get_8x8_zeros()
    queen_list = get_8x8_zeros()
    king_list = get_8x8_zeros()
    player_piece_lists = [pawn_list, knight_list, bishop_list, rook_list, queen_list, king_list]

    for i in range(NUM_PIECE_TYPES):
        # tells which index (on board tiles labeled 0-63) a piece 1
        piece_indices = board.pieces(i+1, color)
        for j in piece_indices:
            x, y = x_y_from_ind(j)
            player_piece_lists[i][y][x] = 1

    return player_piece_lists


def get_8x8_zeros():
    lst = []
    for _ in range(8):
        lst.append([0] * 8)
    return lst


def get_updated_pieces(old_state, move):
    my_pieces = {chess.WHITE: [], chess.BLACK: []}

    move_start_ind = move.from_square
    x_start, y_start = x_y_from_ind(move_start_ind)
    move_end_ind = move.to_square
    x_end, y_end = x_y_from_ind(move_end_ind)

    piece_type_taken = old_state.board.piece_type_at(move_end_ind)
    piece_type_moved = old_state.board.piece_type_at(move_start_ind)

    # FIX_ ME
    # Doesn't work for castles or piece promotions
    for i in range(NUM_PIECE_TYPES):
        if i + 1 == piece_type_moved:
            my_pieces[old_state.turn].append(old_state.my_pieces[old_state.turn][i][:])
            my_pieces[old_state.turn][i][y_start] = \
                my_pieces[old_state.turn][i][y_start][:x_start] \
                + [0] + my_pieces[old_state.turn][i][y_start][x_start+1:]
            my_pieces[old_state.turn][i][y_end] = \
                my_pieces[old_state.turn][i][y_end][:x_end] + [1] + my_pieces[old_state.turn][i][y_end][x_end+1:]
        else:
            my_pieces[old_state.turn].append(old_state.my_pieces[old_state.turn][i])

        if i + 1 == piece_type_taken:
            my_pieces[not old_state.turn].append(old_state.my_pieces[not old_state.turn][i][:])
            my_pieces[not old_state.turn][i][y_end] = \
                my_pieces[not old_state.turn][i][y_end][:x_end] \
                + [0] + my_pieces[not old_state.turn][i][y_end][x_end + 1:]
        else:
            my_pieces[not old_state.turn].append(old_state.my_pieces[not old_state.turn][i])

    return my_pieces


def x_y_from_ind(i):
    return i % 8, i // 8


def piece_plane_from_64_to_8x8(piece_plane):
    my_piece_vectors = []
    for i in range(len(piece_plane)):
        my_piece_vectors.append([piece_plane[i][j:j + 8] for j in range(0, len(piece_plane[i]), 8)])

    return my_piece_vectors


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