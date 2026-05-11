from pathlib import Path
import sqlite3
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "structured" / "financial_indicators.csv"
DB_PATH = BASE_DIR / "financial_risk.db"


def load_financial_indicators() -> pd.DataFrame:
    """
    Load the cleaned financial indicators dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            "Run src/01_download_real_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    return df


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact summary table for SQL querying.
    """

    summary_data = {
        "metric": [
            "average_vix",
            "average_three_month_treasury_rate",
            "average_ten_year_treasury_rate",
            "average_high_yield_credit_spread",
            "average_sofr",
            "average_funding_pressure_score",
            "number_of_high_risk_days",
            "total_observations",
        ],
        "value": [
            df["vix"].mean(),
            df["three_month_treasury_rate"].mean(),
            df["ten_year_treasury_rate"].mean(),
            df["high_yield_credit_spread"].mean(),
            df["sofr"].mean(),
            df["funding_pressure_score"].mean(),
            df["risk_flag"].sum(),
            len(df),
        ],
    }

    return pd.DataFrame(summary_data)


def create_monthly_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a monthly summary of risk conditions.
    """

    monthly_df = df.copy()
    monthly_df["month"] = monthly_df["date"].dt.to_period("M").astype(str)

    monthly_summary = (
        monthly_df.groupby("month")
        .agg(
            average_vix=("vix", "mean"),
            average_short_rate=("three_month_treasury_rate", "mean"),
            average_credit_spread=("high_yield_credit_spread", "mean"),
            average_funding_pressure=("funding_pressure_score", "mean"),
            high_risk_days=("risk_flag", "sum"),
            observations=("risk_flag", "count"),
        )
        .reset_index()
    )

    monthly_summary["high_risk_share"] = (
        monthly_summary["high_risk_days"] / monthly_summary["observations"]
    )

    return monthly_summary


def write_database(
    indicators_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> None:
    """
    Write project tables into SQLite.
    """

    conn = sqlite3.connect(DB_PATH)

    indicators_df.to_sql(
        "financial_indicators",
        conn,
        if_exists="replace",
        index=False,
    )

    summary_df.to_sql(
        "risk_summary",
        conn,
        if_exists="replace",
        index=False,
    )

    monthly_df.to_sql(
        "monthly_risk_summary",
        conn,
        if_exists="replace",
        index=False,
    )

    conn.close()


def check_database() -> None:
    """
    Print database tables and example queries.
    """

    conn = sqlite3.connect(DB_PATH)

    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn,
    )

    print("\nCreated SQLite database:")
    print(DB_PATH)

    print("\nTables:")
    print(tables)

    print("\nTop 5 highest funding pressure days:")
    query = """
    SELECT
        date,
        vix,
        three_month_treasury_rate,
        ten_year_treasury_rate,
        high_yield_credit_spread,
        sofr,
        funding_pressure_score,
        risk_flag
    FROM financial_indicators
    ORDER BY funding_pressure_score DESC
    LIMIT 5;
    """
    print(pd.read_sql_query(query, conn))

    print("\nMonthly periods with highest high-risk share:")
    query = """
    SELECT
        month,
        ROUND(average_vix, 2) AS average_vix,
        ROUND(average_credit_spread, 2) AS average_credit_spread,
        ROUND(average_funding_pressure, 2) AS average_funding_pressure,
        high_risk_days,
        observations,
        ROUND(high_risk_share, 2) AS high_risk_share
    FROM monthly_risk_summary
    ORDER BY high_risk_share DESC
    LIMIT 5;
    """
    print(pd.read_sql_query(query, conn))

    conn.close()


def main() -> None:
    indicators_df = load_financial_indicators()
    summary_df = create_summary_table(indicators_df)
    monthly_df = create_monthly_risk_table(indicators_df)

    write_database(indicators_df, summary_df, monthly_df)
    check_database()


if __name__ == "__main__":
    main()