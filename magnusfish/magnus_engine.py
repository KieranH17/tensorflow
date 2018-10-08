import tensorflow as tf
import chess
import chess.uci
import pgn_to_data
import heapq


class Model:
    def __init__(self, training_pgns=None, save_dir=None, batch_size=100, step_size=1000, hidden_units=64,
                 player_name="Carlsen", pseudo="DrDrunkenstein", csv_out=None, csv_in=None):
        self.training_pgns = training_pgns
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.step_size = step_size
        self.hidden_units = hidden_units
        self.player_name = player_name
        self.pseudo = pseudo
        self.csv_out = csv_out
        self.csv_in = csv_in

        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_cols,
            hidden_units=hidden_units,
            model_dir=self.save_dir,
            n_classes=len(all_possible_moves),
            label_vocabulary=all_possible_moves,
            optimizer=tf.train.AdamOptimizer()
        )
        self.train_features = {}
        self.train_labels = []
        if self.training_pgns:
            self.prep_pgn()

    def prep_pgn(self):
        for pgn_file in self.training_pgns:
            features_to_add, labels_to_add = \
                pgn_to_data.pgn_to_train_data(pgn_file, player_name=self.player_name,
                                              pseudo=self.pseudo, csv_path=self.csv_out)

            if not self.train_features:
                self.train_features.update(features_to_add)
            else:
                for key in self.train_features.keys():
                    self.train_features.get(key).extend(features_to_add.get(key))
            self.train_labels.extend(labels_to_add)

    def train(self):
        self.classifier.train(
            input_fn=lambda: train_input_fn(
                self.train_features, self.train_labels, self.batch_size),
            steps=self.step_size
        )

    def eval(self, test_pgn):
        test_data = pgn_to_data.pgn_to_train_data(test_pgn, player_name=self.player_name, pseudo=self.pseudo)
        test_features, test_labels = test_data

        eval_result = self.classifier.evaluate(
            input_fn=lambda: eval_input_fn(test_features, test_labels, self.batch_size)
        )
        return eval_result

    def predict_move(self, board_fen):
        features = pgn_to_data.features_from_fen(board_fen)
        predictions = self.classifier.predict(
            input_fn=lambda: eval_input_fn(features, None, self.batch_size))

        prediction_dict = next(predictions)
        class_id = prediction_dict["class_ids"][0]
        probability = prediction_dict["probabilities"][class_id]

        return all_possible_moves[class_id], probability

    def select_move(self, board_fen, time):
        features = pgn_to_data.features_from_fen(board_fen)
        predictions = self.classifier.predict(
            input_fn=lambda: eval_input_fn(features, None, self.batch_size))

        prediction_dict = next(predictions)

        move_number = chess.Board(fen=board_fen).fullmove_number
        if move_number < 4:
            engines = []
            stockfish_eng = chess.uci.popen_engine("chess_engines/stockfish-9-mac/Mac/stockfish-9-64")
            stockfish_eng.uci()
            engines.append(stockfish_eng)
            return mag_move(board_fen, prediction_dict, time, engines, n=6, k=7, force_diff=100)
        elif move_number < 10:
            engines = []
            stockfish_eng = chess.uci.popen_engine("chess_engines/stockfish-9-mac/Mac/stockfish-9-64")
            stockfish_eng.uci()
            engines.append(stockfish_eng)
            return mag_move(board_fen, prediction_dict, time, engines, n=5, force_diff=60)
        else:
            engines = []
            mamba_eng = chess.uci.popen_engine("chess_engines/BlackMamba/BlackMamba_1_4")
            mamba_eng.uci()
            engines.append(mamba_eng)
            return mag_move(board_fen, prediction_dict, time, engines, n=4)


def mag_move(board_fen, prediction_dict, time, engines, n=5, k=5, use_n_engines=False, force_diff=50):

    if use_n_engines:
        engine_suggestions = n_engines(board_fen, time)
    else:
        engine_suggestions = get_engine_top_n(board_fen, n, time, engines, k=k)

    suggested_moves_dict = engine_suggestions[0]
    print(suggested_moves_dict)

    best_move_uci = engine_pick_uci = engine_suggestions[1]
    engine_pick_score = suggested_moves_dict.get(engine_suggestions[1])

    max_mag_score = prediction_dict["probabilities"][all_possible_moves.index(engine_pick_uci)]
    for move_uci in suggested_moves_dict.keys():
        mag_score = prediction_dict["probabilities"][all_possible_moves.index(move_uci)]

        if mag_score > max_mag_score:
            max_mag_score = mag_score
            best_move_uci = move_uci

    engine_score = suggested_moves_dict.get(best_move_uci)
    if (engine_score - engine_pick_score) > force_diff and \
            (engine_score > -500 or (engine_score - engine_pick_score) > 5 * force_diff):
            best_move_uci = engine_pick_uci
            probability = "forced to be 100"
    else:
        probability = max_mag_score * 100

    return best_move_uci, probability


def n_engines(board_fen, time):

    board = chess.Board(board_fen)
    engines = []

    stockfish_eng = chess.uci.popen_engine("chess_engines/stockfish-9-mac/Mac/stockfish-9-64")
    stockfish_eng.uci()
    engines.append(stockfish_eng)

    komodo_eng = chess.uci.popen_engine("chess_engines/komodo-9_9dd577 2/OSX/komodo-9.02-64-osx")
    komodo_eng.uci()
    engines.append(komodo_eng)

    mamba_eng = chess.uci.popen_engine("chess_engines/BlackMamba/BlackMamba_1_4")
    mamba_eng.uci()
    engines.append(mamba_eng)

    best_moves = {}
    best_move_uci = ""
    best_move_score = MAX_SCORE

    for engine in engines:
        info_handler = chess.uci.InfoHandler()
        engine.info_handlers.append(info_handler)
        engine.position(board)
        move_uci = engine.go(movetime=(time / len(engines)))[0].uci()

        cp_score = info_handler.info["score"][1].cp

        mate_score = None
        if info_handler.info["score"][1].mate:
            mate_num = info_handler.info["score"][1].mate
            mate_score = int(MAX_SCORE - abs(mate_num))
            if mate_num < 0:
                mate_score = - mate_score

        if mate_score:
            board_eval = -int(mate_score)
        else:
            board_eval = -int(cp_score)

        if board_eval < best_move_score:
            best_move_score = board_eval
            best_move_uci = move_uci

        print(move_uci)
        best_moves[move_uci] = board_eval

    return best_moves, best_move_uci


feature_cols = [tf.feature_column.numeric_column(key="board_array", shape=64),
                tf.feature_column.numeric_column(key="turn", dtype=tf.int32),
                tf.feature_column.numeric_column(key="castling", shape=4),
                tf.feature_column.numeric_column(key="en_pass", dtype=tf.int32),
                tf.feature_column.numeric_column(key="legal_moves", shape=[150, 2])]

all_possible_moves = pgn_to_data.all_possible_moves


def train_input_fn(feature_dict, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((feature_dict, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(feature_dict, labels, batch_size):
    if not labels:
        inputs = feature_dict
    else:
        inputs = (feature_dict, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset


# n (number of desired suggestions) should be less than or equal to k,
# k heavily considered moves, all moves considered for frac of time
def get_engine_top_n(board_fen, n, time, engines, k=5, frac=3/4):
    board = chess.Board(board_fen)

    all_move_dict = {}
    moves = list(board.legal_moves)
    num_moves = len(moves)
    time_per_all_moves = frac * (time / num_moves)
    for move in moves:
        board_copy = chess.Board(board_fen)
        board_copy.push(move)
        all_move_dict[move.uci()] = poll_the_engines(engines, board_copy, time_per_all_moves)

    k = min(k, len(all_move_dict))
    best_k_ucis = heapq.nsmallest(k, all_move_dict, key=all_move_dict.get)

    best_k_dict = {}
    time_per_k_moves = (1 - frac) * (time / k)
    for uci in best_k_ucis:
        board_copy = chess.Board(board_fen)
        board_copy.push(chess.Move.from_uci(uci))
        best_k_dict[uci] = poll_the_engines(engines, board_copy, time_per_k_moves)

    n = min(n, len(best_k_dict))
    best_n_ucis = heapq.nsmallest(n, best_k_dict, key=best_k_dict.get)
    engine_pick_uci = best_n_ucis[0]

    best_n_dict = {}
    for uci in best_n_ucis:
        best_n_dict[uci] = best_k_dict.get(uci)

    return best_n_dict, engine_pick_uci


def poll_the_engines(engines, board, time):

    board_eval = 0
    for engine in engines:
        info_handler = chess.uci.InfoHandler()
        engine.info_handlers.append(info_handler)
        engine.position(board)
        engine.go(movetime=time)

        cp_score = info_handler.info["score"][1].cp

        mate_score = None
        if info_handler.info["score"][1].mate:
            mate_num = info_handler.info["score"][1].mate
            mate_score = int(MAX_SCORE - abs(mate_num))
            if mate_num < 0:
                mate_score = - mate_score

        if mate_score:
            board_eval += int(mate_score)
        else:
            board_eval += int(cp_score)

    return int(board_eval / len(engines))


MAX_SCORE = 100000
