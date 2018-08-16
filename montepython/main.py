import magnus_engine
import os
import sys


def main():
    model_name = sys.argv[1]
    train_dir_name = sys.argv[2]
    test_dir_name = sys.argv[3]

    train_pgns = []
    for file_name in os.listdir(train_dir_name):
        if file_name == ".DS_Store":
            continue
        train_pgns.append(open(train_dir_name+"/"+file_name))

    test_pgns = []
    for file_name in os.listdir(test_dir_name):
        if file_name == ".DS_Store":
            continue
        test_pgns.append(open(test_dir_name+"/"+file_name))

    engine_model = magnus_engine.Model(training_pgns=train_pgns,
                                       save_dir="models/" + model_name,
                                       batch_size=100,
                                       step_size=100000,
                                       hidden_units=[64, 8, 8, 4])
    engine_model.train()

    print(engine_model.predict_move("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"))
    print(engine_model.predict_move("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"))
    print(engine_model.predict_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))

    for pgn in test_pgns:
        print(pgn)
        print(engine_model.eval(pgn))


if __name__ == "__main__":
    main()
