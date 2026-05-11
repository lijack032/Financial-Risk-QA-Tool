from pathlib import Path
import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "structured"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2022-01-01"
END_DATE = "2026-01-01"


def download_yahoo_data() -> pd.DataFrame:
    """
    Download market-based indicators from Yahoo Finance.

    ^VIX = CBOE Volatility Index
    XLF = Financial Select Sector SPDR Fund
    TLT = Long-term Treasury bond ETF
    SHY = Short-term Treasury bond ETF
    """

    tickers = {
        "^VIX": "vix",
        "XLF": "financial_sector_etf",
        "TLT": "long_treasury_etf",
        "SHY": "short_treasury_etf",
    }

    frames = []

    for ticker, name in tickers.items():
        print(f"Downloading Yahoo Finance data for {ticker}...")

        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            raise ValueError(f"No data downloaded for {ticker}")
        
        df.columns = df.columns.get_level_values(0)  

        close = df[["Close"]].rename(columns={"Close": name})
        frames.append(close)

    yahoo_df = pd.concat(frames, axis=1)
    yahoo_df.index.name = "date"
    yahoo_df = yahoo_df.reset_index()

    return yahoo_df


def download_fred_series(series_id: str, column_name: str) -> pd.DataFrame:
    """
    Download one FRED series directly from FRED's CSV endpoint.

    This avoids pandas_datareader, which can break in newer Python versions.
    """

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    df = pd.read_csv(url)
    df = df.rename(columns={"observation_date": "date", series_id: column_name})

    df["date"] = pd.to_datetime(df["date"])
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    df = df[(df["date"] >= START_DATE) & (df["date"] < END_DATE)]

    return df


def download_fred_data() -> pd.DataFrame:
    """
    Download macro and funding-related indicators from FRED.

    DGS3MO = 3-month Treasury constant maturity rate
    DGS10 = 10-year Treasury constant maturity rate
    BAMLH0A0HYM2 = ICE BofA US High Yield Option-Adjusted Spread
    SOFR = Secured Overnight Financing Rate
    """

    fred_series = {
        "DGS3MO": "three_month_treasury_rate",
        "DGS10": "ten_year_treasury_rate",
        "BAMLH0A0HYM2": "high_yield_credit_spread",
        "SOFR": "sofr",
    }

    frames = []

    for series_id, name in fred_series.items():
        print(f"Downloading FRED data for {series_id}...")
        series_df = download_fred_series(series_id, name)
        frames.append(series_df)

    fred_df = frames[0]

    for frame in frames[1:]:
        fred_df = pd.merge(fred_df, frame, on="date", how="outer")

    fred_df = fred_df.sort_values("date").reset_index(drop=True)

    return fred_df


def clean_and_merge(yahoo_df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Yahoo and FRED data into one daily dataset.
    Missing values are forward-filled because some series do not update every trading day.
    """

    yahoo_df["date"] = pd.to_datetime(yahoo_df["date"])
    fred_df["date"] = pd.to_datetime(fred_df["date"])

    df = pd.merge(yahoo_df, fred_df, on="date", how="outer")
    df = df.sort_values("date")

    numeric_cols = [col for col in df.columns if col != "date"]
    df[numeric_cols] = df[numeric_cols].ffill()

    df = df.dropna().reset_index(drop=True)

    return df


def create_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create project-level financial risk indicators.

    These are not official RBC or regulatory metrics.
    They are transparent analytical features used to demonstrate
    financial data analysis, SQL querying, and risk interpretation.
    """

    df = df.copy()

    df["yield_spread_10y_3m"] = (
        df["ten_year_treasury_rate"] - df["three_month_treasury_rate"]
    )

    df["vix_20d_change"] = df["vix"].diff(20)
    df["credit_spread_20d_change"] = df["high_yield_credit_spread"].diff(20)
    df["sofr_20d_change"] = df["sofr"].diff(20)

    df["financial_sector_20d_return"] = (
        df["financial_sector_etf"].pct_change(20) * 100
    )

    df["funding_pressure_score"] = (
        0.45 * df["vix"]
        + 4.00 * df["high_yield_credit_spread"]
        + 1.50 * df["sofr"]
        - 1.00 * df["yield_spread_10y_3m"]
    )

    threshold = df["funding_pressure_score"].quantile(0.85)
    df["risk_flag"] = (df["funding_pressure_score"] >= threshold).astype(int)

    df = df.dropna().reset_index(drop=True)

    return df


def main() -> None:
    yahoo_df = download_yahoo_data()
    fred_df = download_fred_data()

    merged_df = clean_and_merge(yahoo_df, fred_df)
    final_df = create_risk_features(merged_df)

    output_path = OUTPUT_DIR / "financial_indicators.csv"
    final_df.to_csv(output_path, index=False)

    print("\nSaved cleaned financial indicators to:")
    print(output_path)

    print("\nDataset preview:")
    print(final_df.head())

    print("\nColumns:")
    print(final_df.columns.tolist())

    print("\nRisk flag distribution:")
    print(final_df["risk_flag"].value_counts())


if __name__ == "__main__":
    main()