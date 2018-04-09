import tensorflow as tf
import chess
import pgn_to_data


class MagnusModel:
    def __init__(self, training_pgns, save_dir, batch_size, step_size, hidden_units):
        self.training_pgns = training_pgns
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.step_size = step_size
        self.hidden_units = hidden_units
        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_cols,
            hidden_units=hidden_units,
            model_dir=self.save_dir,
            n_classes=4272,
            label_vocabulary=all_possible_moves,
        )
        self.train_features = {}
        self.train_labels = []
        for pgn_file in self.training_pgns:
            features_to_add, labels_to_add = pgn_to_data.pgn_to_train_data(pgn_file)
            if not self.train_features:
                self.train_features.update(features_to_add)
            else:
                for key in self.train_features.keys():
                    self.train_features.get(key).extend(features_to_add.get(key))
            self.train_labels.extend(labels_to_add)

    def train(self):
        for label in self.train_labels:
            if label not in all_possible_moves:
                print(label)
        self.classifier.train(
            input_fn=lambda: train_input_fn(
                self.train_features, self.train_labels, self.batch_size),
            steps=self.step_size
        )

    def eval(self, test_pgn):
        test_data = pgn_to_data.pgn_to_train_data(test_pgn)
        test_features, test_labels = test_data
        for label in test_labels:
            if label not in all_possible_moves or not label:
                print(label)
        eval_result = self.classifier.evaluate(
            input_fn=lambda: eval_input_fn(test_features, test_labels, self.batch_size)
        )
        return eval_result

    def predict_move(self, board_fen):
        features = pgn_to_data.features_from_fen(board_fen)
        predictions = self.classifier.predict(
            input_fn=lambda: eval_input_fn(features, None, self.batch_size)
        )
        for prediction_dict, labels in zip(predictions, all_possible_moves):
            class_id = prediction_dict["class_ids"][0]
            probability = prediction_dict["probabilities"][class_id]

        return all_possible_moves[class_id], probability


class StockfishModel:
    def __init__(self, training_pgns, save_dir, batch_size, step_size, hidden_units):
        self.training_pgns = training_pgns
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.step_size = step_size
        self.hidden_units = hidden_units
        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_cols,
            hidden_units=hidden_units,
            model_dir=self.save_dir,
            n_classes=4272,
            label_vocabulary=all_possible_moves,
        )
        self.train_features = {}
        self.train_labels = []
        for pgn_file in self.training_pgns:
            features_to_add, labels_to_add = pgn_to_data.pgn_to_train_data(pgn_file)
            if not self.train_features:
                self.train_features.update(features_to_add)
            else:
                for key in self.train_features.keys():
                    self.train_features.get(key).extend(features_to_add.get(key))
            self.train_labels.extend(labels_to_add)

    def train(self):
        for label in self.train_labels:
            if label not in all_possible_moves:
                print(label)
        self.classifier.train(
            input_fn=lambda: train_input_fn(
                self.train_features, self.train_labels, self.batch_size),
            steps=self.step_size
        )

    def eval(self, test_pgn):
        test_data = pgn_to_data.pgn_to_train_data(test_pgn)
        test_features, test_labels = test_data
        for label in test_labels:
            if label not in all_possible_moves or not label:
                print(label)
        eval_result = self.classifier.evaluate(
            input_fn=lambda: eval_input_fn(test_features, test_labels, self.batch_size)
        )
        return eval_result

    def predict_move(self, board_fen):
        features = pgn_to_data.features_from_fen(board_fen)
        predictions = self.classifier.predict(
            input_fn=lambda: eval_input_fn(features, None, self.batch_size)
        )
        for prediction_dict, labels in zip(predictions, all_possible_moves):
            class_id = prediction_dict["class_ids"][0]
            probability = prediction_dict["probabilities"][class_id]

        return all_possible_moves[class_id], probability


feature_cols = [tf.feature_column.numeric_column(key="board_array", shape=64),
                tf.feature_column.numeric_column(key="turn", dtype=tf.int32),
                tf.feature_column.numeric_column(key="castling", shape=4),
                tf.feature_column.numeric_column(key="en_pass", dtype=tf.int32),
                tf.feature_column.numeric_column(key="legal_moves", shape=4272)]

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

    return dataset.make_one_shot_iterator().get_next()
