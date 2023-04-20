import os

import joblib
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task

from .data_preparation import prepare


def train_model(
        file_path: str
) -> None:
    """
    Model training and saving its weights
    :file_path: path where file is located
    :returns: None
    """
    if not os.path.exists("data/train_processed.csv"):
        prepare(
            file_path,
            "data/train_processed",
            is_train=True
        )
    train = pd.read_csv("data/train_processed.csv")

    lama = TabularUtilizedAutoML(
        task=Task(name="binary", metric="auc"),
        cpu_limit=12,
        timeout=360,
        reader_params={"random_state": 11},
        general_params={"return_all_predictions": False}
    )

    _ = lama.fit_predict(
        train,
        roles={"target": "label"},
        verbose=0,
        log_file="models/lama.log"
    )
    joblib.dump(lama, "models/lama.pkl")
