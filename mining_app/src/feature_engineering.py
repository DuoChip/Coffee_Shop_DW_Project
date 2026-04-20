import pandas as pd
import numpy as np


DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

MONTH_ORDER = list(range(1, 13))


def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu sau khi join.
    """
    out = df.copy()

    if "transaction_date" in out.columns:
        out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")

    if "transaction_qty" in out.columns:
        out["transaction_qty"] = pd.to_numeric(out["transaction_qty"], errors="coerce").fillna(0)

    if "unit_price" in out.columns:
        out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce").fillna(0)

    if "revenue" not in out.columns:
        out["revenue"] = out["transaction_qty"] * out["unit_price"]
    else:
        out["revenue"] = pd.to_numeric(out["revenue"], errors="coerce").fillna(0)

    if "day_name" in out.columns:
        out["day_name"] = pd.Categorical(out["day_name"], categories=DAY_ORDER, ordered=True)

    if "month_number" in out.columns:
        out["month_number"] = pd.to_numeric(out["month_number"], errors="coerce")
        out["month_number"] = pd.Categorical(out["month_number"], categories=MONTH_ORDER, ordered=True)

    if "hour_number" in out.columns:
        out["hour_number"] = pd.to_numeric(out["hour_number"], errors="coerce")

    if "day_name" in out.columns:
        out["is_weekend"] = out["day_name"].astype(str).isin(["Saturday", "Sunday"]).astype(int)
    else:
        out["is_weekend"] = 0

    if "transaction_date" in out.columns:
        out["year_month"] = out["transaction_date"].dt.to_period("M").astype(str)
        out["date_only"] = out["transaction_date"].dt.date

    return out


def build_store_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo feature cho clustering theo mức store-day.
    Tránh việc cluster trực tiếp chỉ 3 cửa hàng.
    """
    required_cols = [
        "store_id", "store_location", "transaction_date", "transaction_id",
        "transaction_qty", "revenue", "hour_number",
        "product_category", "is_weekend"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột cần thiết cho clustering: {missing}")

    tmp = df.copy()
    tmp["date_only"] = pd.to_datetime(tmp["transaction_date"]).dt.date

    grouped = (
        tmp.groupby(["store_id", "store_location", "date_only"], as_index=False)
           .agg(
               total_revenue=("revenue", "sum"),
               total_qty=("transaction_qty", "sum"),
               transaction_count=("transaction_id", "nunique"),
               avg_hour=("hour_number", "mean"),
               weekend_ratio=("is_weekend", "mean"),
           )
    )

    peak_hour_df = (
        tmp.groupby(["store_id", "store_location", "date_only", "hour_number"], as_index=False)
           .agg(hour_revenue=("revenue", "sum"))
           .sort_values(["store_id", "date_only", "hour_revenue"], ascending=[True, True, False])
           .drop_duplicates(subset=["store_id", "date_only"])
           [["store_id", "date_only", "hour_number"]]
           .rename(columns={"hour_number": "peak_hour"})
    )

    category_rev = (
        tmp.pivot_table(
            index=["store_id", "store_location", "date_only"],
            columns="product_category",
            values="revenue",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    feature_df = grouped.merge(
        peak_hour_df,
        on=["store_id", "date_only"],
        how="left"
    ).merge(
        category_rev,
        on=["store_id", "store_location", "date_only"],
        how="left"
    )

    category_cols = [
        c for c in feature_df.columns
        if c not in {
            "store_id", "store_location", "date_only", "total_revenue",
            "total_qty", "transaction_count", "avg_hour", "weekend_ratio", "peak_hour"
        }
    ]

    for c in category_cols:
        feature_df[f"ratio_{str(c).lower().replace(' ', '_')}"] = np.where(
            feature_df["total_revenue"] > 0,
            feature_df[c] / feature_df["total_revenue"],
            0
        )

    feature_df["avg_order_value"] = np.where(
        feature_df["transaction_count"] > 0,
        feature_df["total_revenue"] / feature_df["transaction_count"],
        0
    )

    keep_cols = [
        "store_id", "store_location", "date_only",
        "total_revenue", "total_qty", "transaction_count",
        "avg_order_value", "avg_hour", "peak_hour", "weekend_ratio"
    ] + [c for c in feature_df.columns if c.startswith("ratio_")]

    return feature_df[keep_cols].copy()
    