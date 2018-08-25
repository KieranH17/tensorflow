
import tensorflow as tf
import montepython_cnn
import train_game
import numpy as np
import os
import sys
import pgn_to_data
import play_game


def main():
    board_result_classifier = tf.estimator.Estimator(
        model_fn=montepython_cnn.mp_ccn_fn,
        model_dir="models/montepython_boardeval_two"
    )

    if sys.argv[1] == "play":
        play_game.main(board_result_classifier)
        return

    # setup logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50
    )

    train_dir_name = "pgn_folder/small_test"
    train_pgns = []
    for file_name in os.listdir(train_dir_name):
        if file_name == ".DS_Store":
            continue
        train_pgns.append(open(train_dir_name + "/" + file_name))

    games_and_results = []
    for pgn in train_pgns:
        games_and_results.extend(pgn_to_data.pgn_file_to_games_and_results(pgn))

    states, outcomes = [], []
    for game, result in games_and_results:
        states_to_add, outcomes_to_add = train_game.run_train_game(
            board_classifier=board_result_classifier,
            import_game_and_result=(game, result)
        )
        states.extend(states_to_add)
        outcomes.extend(outcomes_to_add)
    all_features = [state.get_input_layers() for state in states]

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=np.asarray(all_features),
        y=np.asarray(outcomes),
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    board_result_classifier.train(
        input_fn=train_input_fn,
        steps=6000,
        hooks=[logging_hook]
    )

    #
    #
    # evaluate on fun sample states
    # eval_states = []
    # eval_outcomes = []
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x=np.asarray(eval_states),
    #     y=np.asarray(eval_outcomes),
    #     num_epochs=1,
    #     shuffle=False)
    # eval_results = board_result_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)

    #
    #
    # predict on more fun sample states
    # pred_states = []
    # pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x=np.asarray(pred_states),
    #     num_epochs=1,
    #     shuffle=False)
    # predict_results = board_result_classifier.predict(input_fn=pred_input_fn)
    # print(predict_results)


if __name__ == "__main__":
    main()
