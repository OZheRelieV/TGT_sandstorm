import os

import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError

from .data_preparation import prepare


def predict(
    file_path: str,
    output_filename: str,
) -> None:
    """
    Making prediction
    :file_path: path where file is located
    :output_filename: path/file name where file with prediction
                      will be located
    :returns: None
    """
    if not os.path.exists("data/test_processed.csv"):
        prepare(
            file_path,
            "data/test_processed",
            is_train=False
        )
    test = pd.read_csv("data/test_processed.csv")

    if not os.path.exists("models/lama.pkl"):
        raise NotFittedError(
            "Model isn't fitted. Call modell_fitting.py before using estimator"
        )
    else:
        lama = joblib.load("models/lama.pkl")

        preds = lama.predict(test).data
        pd.DataFrame(
            columns=["label"],
            data=list(map(lambda x: 1 if x > 0.3 else 0, preds.flatten()))
        ).to_csv(f"{output_filename}.csv", index=False)
