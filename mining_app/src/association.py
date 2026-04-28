import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def add_basket_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo basket_id tạm thời vì transaction_id hiện tại chỉ là ID từng dòng.
    Gom các dòng có cùng store_id + date_skey + time_skey thành 1 basket.
    """
    df = df.copy()

    required_cols = ["store_id", "date_skey", "time_skey"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột để tạo basket_id: {missing}")

    df["basket_id"] = (
        df["store_id"].astype(str) + "_" +
        df["date_skey"].astype(str) + "_" +
        df["time_skey"].astype(str)
    )

    return df


def build_basket(df: pd.DataFrame, product_col: str = "product_detail") -> pd.DataFrame:
    """
    Tạo basket theo basket_id x product_detail/product_type/product_category.
    """
    df = add_basket_id(df)

    required_cols = ["basket_id", product_col, "transaction_qty"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột để build basket: {missing}")

    basket = (
        df.groupby(["basket_id", product_col])["transaction_qty"]
          .sum()
          .unstack(fill_value=0)
    )

    basket = (basket > 0).astype(bool)

    return basket


def run_apriori(
    df: pd.DataFrame,
    product_col: str = "product_detail",
    min_support: float = 0.01,
    min_confidence: float = 0.2,
    min_lift: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Trả về:
    - basket
    - frequent itemsets
    - association rules
    """
    basket = build_basket(df, product_col=product_col)

    if basket.empty:
        return basket, pd.DataFrame(), pd.DataFrame()

    max_items_per_basket = basket.sum(axis=1).max()

    if max_items_per_basket < 2:
        return basket, pd.DataFrame(), pd.DataFrame()

    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        return basket, frequent_itemsets, pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        return basket, frequent_itemsets, rules

    rules = rules[rules["lift"] >= min_lift].copy()

    if rules.empty:
        return basket, frequent_itemsets, rules

    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(map(str, sorted(list(x))))
    )

    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(map(str, sorted(list(x))))
    )

    rules["rule"] = rules["antecedents_str"] + " → " + rules["consequents_str"]

    rules = rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False
    ).reset_index(drop=True)

    return basket, frequent_itemsets, rules


def get_top_rules(rules: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if rules.empty:
        return rules

    cols = [
        "rule",
        "support",
        "confidence",
        "lift",
        "antecedents_str",
        "consequents_str"
    ]

    available_cols = [c for c in cols if c in rules.columns]

    return rules[available_cols].head(top_n).copy()