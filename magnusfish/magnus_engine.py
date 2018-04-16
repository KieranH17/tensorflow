import tensorflow as tf
import chess
import pgn_to_data
from random import randint


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

    def select_move(self, board_fen):
        board = chess.Board(fen=board_fen)
        features = pgn_to_data.features_from_fen(board_fen)
        predictions = self.classifier.predict(
            input_fn=lambda: eval_input_fn(features, None, self.batch_size))

        prediction_dict = next(predictions)
        max_score = 0
        legal_moves = [move for move in board.legal_moves]
        best_move = legal_moves[randint(0, len(legal_moves)-1)]
        for move in board.legal_moves:
            move_uci = move.uci()
            score = prediction_dict["probabilities"][all_possible_moves.index(move_uci)]
            if score > max_score:
                max_score = score
                best_move = move_uci
        probability = max_score*100
        return best_move, probability


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
