from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_processed_data_dir() -> Path:
    return get_project_root() / "data" / "processed"


def load_csv(filename: str) -> pd.DataFrame:
    path = get_processed_data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_csv(path)


def load_dw_data() -> dict:
    """
    Load toàn bộ fact + dimensions từ thư mục data/processed
    """
    fact = load_csv("facts_sales.csv")
    dim_product = load_csv("dim_product.csv")
    dim_date = load_csv("dim_date.csv")
    dim_time = load_csv("dim_time.csv")
    dim_store = load_csv("dim_store.csv")

    return {
        "fact": fact,
        "dim_product": dim_product,
        "dim_date": dim_date,
        "dim_time": dim_time,
        "dim_store": dim_store,
    }


def build_analysis_dataframe() -> pd.DataFrame:
    """
    Join fact và các dimension thành 1 dataframe phân tích duy nhất.
    """
    data = load_dw_data()

    fact = data["fact"].copy()
    dim_product = data["dim_product"].copy()
    dim_date = data["dim_date"].copy()
    dim_time = data["dim_time"].copy()
    dim_store = data["dim_store"].copy()

    if "line_item_amount" in fact.columns and "revenue" not in fact.columns:
        fact["revenue"] = fact["line_item_amount"]

    df = (
        fact.merge(dim_product, on="product_id", how="left")
            .merge(dim_date, on="date_skey", how="left")
            .merge(dim_time, on="time_skey", how="left")
            .merge(dim_store, on="store_id", how="left")
    )

    return df