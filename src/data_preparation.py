from . import utils


def prepare(
    file_path: str,
    output_filename: str,
    is_train: bool = True
) -> None:
    """
    Data preparation i.e. generation new features
    :file_path: path where file is located
    :output_filename: path/file name where new csv file with
                      generated features will be placed
    :is_train: flag if input dataframe is train
    :returns: None
    """
    df = utils.get_mccs_features(file_path, is_train=is_train)
    utils.get_stat_features(file_path, df)
    df = utils.get_tsfresh_features(file_path, df)
    assert df.shape[1] == (827 - int(not is_train)), "Smth gone wrong"
    df.to_csv(f"{output_filename}.csv", index=False)
