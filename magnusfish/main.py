import tensorflow as tf
import magnus_engine
import pgn_to_data
import os


def main():
    train_pgns = []
    for file_name in os.listdir("chess.com_training"):
        if file_name == ".DS_Store":
            continue
        train_pgns.append(open("chess.com_training/" + file_name))
    for file_name in os.listdir("training_dir"):
        if file_name == ".DS_Store":
            continue
        train_pgns.append(open("training_dir/" + file_name))

    test_pgns = []
    for file_name in os.listdir("testing_dir"):
        if file_name == ".DS_Store":
            continue
        test_pgns.append(open("testing_dir/" + file_name))

    london_magnus = magnus_engine.MagnusModel(train_pgns, "models/magnus_legal", 300, 2000, [512, 128, 64])

    fen = input()
    while(fen != "q"):
        print(print(london_magnus.predict_move(fen)))
        fen = input()

    print(london_magnus.predict_move("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"))
    print(london_magnus.predict_move("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"))
    print(london_magnus.predict_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))

    for pgn in test_pgns:
        print(pgn)
        print(london_magnus.eval(pgn))

main()
