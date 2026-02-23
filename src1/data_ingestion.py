import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_data(source_path: str) -> pd.DataFrame:
    """
    Load dataset from local path or URL
    """
    return pd.read_csv(source_path)


def preprocess_data(df: pd.DataFrame, drop_columns=None, rename_columns=None) -> pd.DataFrame:
    """
    Optional preprocessing steps
    """
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")

    if rename_columns:
        df = df.rename(columns=rename_columns)

    return df


def save_split_data(train_df, test_df, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train saved at: {train_path}")
    print(f"Test saved at: {test_path}")


def main():
    params = load_params("params.yaml")

    data_source = params["data_ingestion"]["data_source"]
    test_size = params["data_ingestion"]["test_size"]
    output_dir = params["data_ingestion"]["output_dir"]

    drop_columns = params["data_ingestion"].get("drop_columns", None)
    rename_columns = params["data_ingestion"].get("rename_columns", None)

    df = load_data(data_source)
    df = preprocess_data(df, drop_columns, rename_columns)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42
    )

    save_split_data(train_df, test_df, output_dir)

    print("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()