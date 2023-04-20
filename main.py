import os

from src import inference, model_fitting

if __name__ == "__main__":

    train_path = input("Input path/file for train file >")
    test_path = input("Input path/file for test file >")
    output_file_path = input("Input path/file for prediction file >")

    if not os.path.exists("models/lama.pkl"):
        model_fitting.train_model(train_path)
    inference.predict(test_path, output_file_path)
