import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_processed_data_dir() -> Path:
    return get_project_root() / "data" / "processed"


def load_csv(filename: str) -> pd.DataFrame:
    path = get_processed_data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_csv(path)


def get_postgres_uri() -> str:
    """
    Lấy URI kết nối PostgreSQL từ biến môi trường.
    Ưu tiên DATABASE_URL, nếu không có thì tự ghép từ PG* variables.
    """
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE", "postgres")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


def load_dw_data_from_postgres(schema: str | None = None) -> dict:
    """
    Load toàn bộ fact + dimensions từ PostgreSQL.
    Có thể set schema qua biến DW_SCHEMA.
    """
    engine = create_engine(get_postgres_uri())

    table_fact = os.getenv("DW_TABLE_FACT", "facts_sales")
    table_product = os.getenv("DW_TABLE_DIM_PRODUCT", "dim_product")
    table_date = os.getenv("DW_TABLE_DIM_DATE", "dim_date")
    table_time = os.getenv("DW_TABLE_DIM_TIME", "dim_time")
    table_store = os.getenv("DW_TABLE_DIM_STORE", "dim_store")

    schema_name = schema or os.getenv("DW_SCHEMA")

    def read_table(table_name: str) -> pd.DataFrame:
        if schema_name:
            query = f'SELECT * FROM "{schema_name}"."{table_name}"'
        else:
            query = f'SELECT * FROM "{table_name}"'
        return pd.read_sql_query(query, engine)

    return {
        "fact": read_table(table_fact),
        "dim_product": read_table(table_product),
        "dim_date": read_table(table_date),
        "dim_time": read_table(table_time),
        "dim_store": read_table(table_store),
    }


def load_dw_data() -> dict:
    """
    Load toàn bộ fact + dimensions.
    Nguồn dữ liệu được chọn bằng biến DATA_SOURCE:
    - csv (mặc định)
    - postgres
    """
    data_source = os.getenv("DATA_SOURCE", "csv").strip().lower()

    if data_source == "postgres":
        return load_dw_data_from_postgres()

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