import chess.pgn
import chess
import numpy as np
import pandas as pd


def pgn_file_to_games_and_results(pgn_file):
    games_and_results = []

    for offset, headers in chess.pgn.scan_headers(pgn_file):
            pgn_file.seek(offset)
            games_and_results.append((chess.pgn.read_game(pgn_file), headers["Result"]))

    return games_and_results


# appends to the given lists (feature_dict & labels) the decisions made by player_name
# (decisions consist of 1. features: board state (list of FEN fields) and 2. labels:
# move (a UCI string) player_name made) ... does nothing if not a player_name game
def game_to_train_data(game, feature_dict, labels, player_name="Carlsen", pseudo="DrDrunkenstein"):
    board = game.board()
    if player_name in game.headers["White"] or pseudo in game.headers["White"]:
        color = chess.WHITE
    elif player_name in game.headers["Black"] or pseudo in game.headers["Black"]:
        color = chess.BLACK
    else:
        return

    for move in game.main_line():
        if color == board.turn:

            features = board_to_features(board)
            feature_dict.get("board_array").append(features[0])
            feature_dict.get("turn").append(features[1])
            feature_dict.get("castling").append(features[2])
            feature_dict.get("en_pass").append(features[3])
            feature_dict.get("legal_moves").append(features[4])

            move_uci = move.uci()
            labels.append(move_uci)

        board.push(move)


# if csv_path, saves a csv file version of the pgn
def pgn_to_train_data(pgn_file, player_name="Carlsen", pseudo="DrDrunkenstein", csv_path=None):
    games = pgn_file_to_games(pgn_file, player_name, pseudo)

    feature_dict = {"board_array": [], "turn": [], "castling": [], "en_pass": [], "legal_moves": []}
    labels = []
    for game in games:
        game_to_train_data(game, feature_dict, labels, player_name, pseudo)

    if csv_path:
        features_to_csv(feature_dict, labels, csv_path)

    return feature_dict, labels


# writes features/labels to csv file named csv_path (non-destructive)
def features_to_csv(feature_dict, labels, csv_path):
    dict_length = len(labels)
    if dict_length > 25000:
        num_splits = dict_length // 25000 + 1
        for i in range(num_splits):
            start_ind = i * 25000
            end_ind = (i + 1) * 25000
            if i == num_splits - 1:
                end_ind = dict_length - 1
            file_name = open(csv_path + i, "w+")
            feature_dict_new = {"board_array": feature_dict["board_array"][start_ind:end_ind],
                                "turn": feature_dict["turn"][start_ind:end_ind],
                                "castling": feature_dict["castling"][start_ind:end_ind],
                                "en_pass": feature_dict["en_pass"][start_ind:end_ind],
                                "legal_moves": feature_dict["legal_moves"][start_ind:end_ind],
                                "labels": labels[start_ind:end_ind]}
            (pd.DataFrame(feature_dict_new)).to_csv(file_name)
    else:
        feature_dict_new = feature_dict.copy()
        feature_dict_new.update({"labels": labels})
        (pd.DataFrame(feature_dict_new)).to_csv(csv_path)


# Takes ONLY the string of the piece positions from the fen
# returns a 64 elem array representation of the board
def fen_pos_to_64array(fen_pos_string):
    piece_dict = {"p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,
                  "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6}

    board_array = []
    for ch in fen_pos_string:
        if ch == "/":
            continue
        try:
            num_spaces = int(ch)
        except ValueError:
            num_spaces = 0
        if num_spaces:
            for i in range(num_spaces):
                board_array.append(0)
        else:
            board_array.append(piece_dict.get(ch))
    return board_array


# converts a string uci into a 2D array with the 64 based indices of [start, end]
def str_to_64ind(four_ch_str):
    start_pos = coord_to_64ind(str_to_coord(four_ch_str[:2]))
    end_pos = coord_to_64ind(str_to_coord(four_ch_str[2:4]))
    return [start_pos, end_pos]


# could use the chess.square method instead
def coord_to_64ind(tuple_coord):
    return int(tuple_coord[0] + tuple_coord[1] * 8)


def str_to_coord(two_char):
    return ord(two_char[0]) - 97, int(two_char[1]) - 1


def features_from_fen(fen_string):
    feature_dict = {"board_array": [], "turn": [], "castling": [], "en_pass": [], "legal_moves": []}
    board = chess.Board(fen=fen_string)
    features = board_to_features(board)
    feature_dict.get("board_array").append(features[0])
    feature_dict.get("turn").append(features[1])
    feature_dict.get("castling").append(features[2])
    feature_dict.get("en_pass").append(features[3])
    feature_dict.get("legal_moves").append(features[4])

    return feature_dict


all_possible_moves = []
for square1 in chess.SQUARE_NAMES:
    for square2 in chess.SQUARE_NAMES:
        all_possible_moves.append(square1+square2)
        square1_int = coord_to_64ind(str_to_coord(square1))
        square2_int = coord_to_64ind(str_to_coord(square2))
        if (chess.square_rank(square1_int) == 1 and chess.square_rank(square2_int) == 0
                and abs(chess.square_file(square1_int) - chess.square_file(square2_int)) <= 1) \
                or (chess.square_rank(square1_int) == 6 and chess.square_rank(square2_int) == 7
                    and abs(chess.square_file(square1_int) - chess.square_file(square2_int)) <= 1):
            for ch in ["n", "b", "r", "q"]:
                all_possible_moves.append(square1+square2+ch)


def board_to_features(board):
    castling_right_array = np.array([board.has_kingside_castling_rights(chess.WHITE),
                                     board.has_queenside_castling_rights(chess.WHITE),
                                     board.has_kingside_castling_rights(chess.BLACK),
                                     board.has_queenside_castling_rights(chess.BLACK)])
    fen_fields = [board.board_fen(), board.turn, castling_right_array,
                  board.ep_square, board.halfmove_clock, board.fullmove_number,
                  board.legal_moves]

    board_state = fen_fields
    board_array = fen_pos_to_64array(board_state[0])
    turn = 1 * board_state[1]
    castling_int_array = 1 * board_state[2]
    ep_int = board_state[3]
    if not ep_int:
        ep_int = 0

    legal_moves = []
    for move in board_state[6]:
        legal_moves.append([move.from_square, move.to_square])
    legal_moves.sort()
    for _ in range(150-len(legal_moves)):
        legal_moves.append([0, 0])

    return [board_array, turn, castling_int_array.tolist(), ep_int, legal_moves]
