import gc
import warnings

import librosa
import numpy as np
import pandas as pd
import scipy
from tsfresh import extract_features
from tsfresh.feature_extraction import settings
from tsfresh.utilities.dataframe_functions import impute

warnings.filterwarnings("ignore")


def get_mccs_features(
        file_path: str,
        is_train: bool = True
) -> pd.core.frame.DataFrame:
    """
    Making dataframe with mccs features
    :file_path: path where file is located
    :is_train: flag if input dataframe is train
    :returns: dataframe with mccs features
    """
    df = pd.read_csv(file_path)
    mccs = [
        librosa.feature.mfcc(
            y=df.iloc[i, :300].loc[~df.iloc[i, :300].isna()].values,
            sr=117200,
            n_fft=512,
            n_mfcc=20,
            dct_type=3,
        ).ravel()
        for i in range(df.shape[0])
    ]

    result = pd.DataFrame(mccs)
    result.columns = [f"f{i}" for i in range(len(mccs[0]))]

    if is_train:
        result["label"] = df["label"]
        result["label"] = result["label"].astype(np.uint8)
        return result
    else:
        return result


def get_stat_features(
        file_path: str,
        data: pd.core.frame.DataFrame
) -> None:
    """
    Making statistical features
    :file_path: path where file is located
    :data: data where new features will be placed
    :returns: None
    """
    df = pd.read_csv(file_path)
    cols = list(df)[:300]

    data["len"] = 300 - df.isna().sum(axis=1)
    data["max"] = df[cols].max(axis=1)
    data["min"] = df[cols].min(axis=1)
    data["mean"] = df[cols].mean(axis=1)
    data["median"] = df[cols].median(axis=1)
    data["std"] = df[cols].std(axis=1)
    data["q25"] = df[cols].quantile(0.25, axis=1)
    data["q75"] = df[cols].quantile(0.75, axis=1)
    data["sum"] = df[cols].sum(axis=1)
    data["abs_sum"] = df.abs().sum(axis=1)

    for q in np.arange(0.1, 1.0, 0.1):
        data[f"q_{round(q, 1)}"] = df[cols].quantile(round(q, 1), axis=1)

    data["close1"] = [
        np.isclose(df.loc[i].values[:300], 1.0).sum()
        for i in range(df.shape[0])
    ]

    data["greater0"] = [
        (df.loc[i].values[:300] > 0).sum()
        for i in range(df.shape[0])
    ]

    data["fff"] = [
        pd.Series(
            scipy.signal.find_peaks(
                df.iloc[i, : 300], height=0, threshold=0.01
            )[0]
        ).diff().dropna().quantile(0.5)
        for i in range(len(df))
    ]

    data["fff1"] = (df.iloc[:, :5] > 0.3).sum(axis=1)


def get_tsfresh_features(
        file_path: str,
        data: pd.core.frame.DataFrame
) -> pd.core.frame.DataFrame:
    """
    Autogeneration of new features
    :file_path: path where file is located
    :data: data where new features will be placed
    :returns: dataframe with new features
    """
    features = settings.TimeBasedFCParameters()
    features.update(settings.EfficientFCParameters())
    features.update(settings.ComprehensiveFCParameters())

    init_data = pd.read_csv(file_path)
    data_long = pd.DataFrame(
        {
            0: init_data.iloc[:, :300].values.flatten(),
            1: np.arange(
                            init_data.iloc[:, :300].shape[0]
                        ).repeat(init_data.iloc[:, :300].shape[1])
        }
    )

    del init_data
    gc.collect()

    data_long.dropna(inplace=True)
    assert data_long.isna().sum().sum() == 0, "NAs in data"

    new_features = extract_features(
        data_long,
        column_id=1,
        impute_function=impute,
        default_fc_parameters=features
    )

    new_features["idx"] = new_features.index
    data["idx"] = data.index

    data = data.merge(
        new_features,
        on="idx",
        how="left"
    )
    data.drop("idx", axis=1, inplace=True)

    return data
