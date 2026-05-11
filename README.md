# Financial Risk Q&A Tool

## Overview

This project is a financial data science tool for exploring bank liquidity, funding conditions, market risk, and regulatory requirements. It combines structured market data with unstructured financial documents so that a user can ask a question in plain language and receive an answer supported by both document evidence and data.

The project was designed around a realistic Treasury or risk analytics workflow. In practice, analysts often need to work with long reports, regulatory guidelines, market indicators, and data tables at the same time. This project simulates that kind of workflow on a smaller scale using public data and public documents.

The current version can:

1. Download real market and macro-financial indicators.
2. Store cleaned financial indicators in a SQLite database.
3. Download and process public financial documents.
4. Search document passages using NLP retrieval.
5. Classify financial text into risk categories using Scikit-learn.
6. Route user questions to the appropriate tool.
7. Produce a business-facing summary from document evidence and structured indicators.

The project currently has a local research assistant that does not require an LLM API key. A LangChain version is planned as the next step.

## Project Motivation

Bank Treasury teams need to monitor liquidity, funding, interest rates, credit spreads, and regulatory requirements. A lot of this information is spread across different formats:

- annual reports,
- Pillar 3 disclosures,
- regulatory guidelines,
- market data,
- interest rate data,
- credit spread data,
- liquidity and funding metrics.

Because these materials are long and technical, it can be difficult to quickly connect what a document says with what the data shows. For example, a user might ask:

```text
What regulatory requirements apply to bank liquidity, and which months show elevated funding pressure?
```

This question requires two different types of work:

1. Search financial or regulatory documents for liquidity requirements.
2. Query structured market data to identify periods of elevated funding pressure.

The goal of this project is to connect those two steps in one workflow.

## What the Project Does

At a high level, the project works like this:

```text
User question
    ↓
Question routing
    ↓
Relevant tool selection
    ↓
Document search and/or SQL query and/or risk classification
    ↓
Business-facing summary
```

For example:

```text
Question:
What regulatory requirements apply to bank liquidity, and which months show elevated funding pressure?

Agent plan:
document_search -> sql_query
```

The system first searches public financial documents for liquidity-related regulatory requirements. Then it queries the SQL database to find months with elevated funding pressure. Finally, it combines both outputs into a concise interpretation.

## Data Sources

This project uses public data only.

### Structured Financial Data

Structured financial indicators are collected from Yahoo Finance and FRED.

The current dataset includes:

| Variable | Description |
|---|---|
| `vix` | CBOE Volatility Index, used as a market volatility indicator |
| `financial_sector_etf` | XLF ETF price, used as a broad financial-sector proxy |
| `long_treasury_etf` | TLT ETF price, used as a long-term Treasury bond proxy |
| `short_treasury_etf` | SHY ETF price, used as a short-term Treasury bond proxy |
| `three_month_treasury_rate` | 3-month Treasury constant maturity rate |
| `ten_year_treasury_rate` | 10-year Treasury constant maturity rate |
| `high_yield_credit_spread` | ICE BofA high-yield credit spread |
| `sofr` | Secured Overnight Financing Rate |

The project then creates additional analytical features:

| Feature | Purpose |
|---|---|
| `yield_spread_10y_3m` | Captures the difference between long-term and short-term rates |
| `vix_20d_change` | Measures recent change in market volatility |
| `credit_spread_20d_change` | Measures recent change in credit stress |
| `sofr_20d_change` | Measures recent change in short-term funding rates |
| `financial_sector_20d_return` | Measures recent financial-sector performance |
| `funding_pressure_score` | Project-level composite score for market/funding stress |
| `risk_flag` | Binary flag for high funding pressure periods |

The funding pressure score is a project-level screening feature. It is not an official RBC, OSFI, Basel, or regulatory metric. It is used only to demonstrate how several market indicators can be combined into a simple risk-monitoring signal.

### Unstructured Financial Documents

The project also uses public financial and regulatory documents. The current document set includes:

| Document | Purpose |
|---|---|
| RBC Annual Report | Used for risk disclosures, funding discussion, liquidity risk, credit risk, market risk, and business context |
| RBC Pillar 3 Report | Used for regulatory capital, risk-weighted assets, credit risk, counterparty risk, and market risk disclosures |
| OSFI Liquidity Adequacy Requirements Guideline | Used for Canadian liquidity requirements, including LCR and NSFR |

These documents are downloaded as PDFs, converted into text, split into chunks, and indexed for retrieval.

## Project Architecture

The project has four main layers:

```text
1. Data collection layer
   - downloads market indicators
   - downloads financial documents

2. Data preparation layer
   - cleans structured data
   - creates risk features
   - extracts PDF text
   - chunks documents

3. Modeling and retrieval layer
   - builds a SQL database
   - builds a TF-IDF document retriever
   - trains a Scikit-learn risk classifier

4. Assistant layer
   - routes user questions
   - calls the appropriate tool
   - combines tool outputs
   - creates a business-facing summary
```

## Project Structure

```text
Financial Risk Q&A Tool/
├── data/
│   ├── raw_docs/
│   │   └── downloaded PDF reports
│   ├── processed_docs/
│   │   └── extracted text files
│   └── structured/
│       └── financial_indicators.csv
│
├── artifacts/
│   ├── document_index/
│   │   ├── chunks.json
│   │   ├── tfidf_vectorizer.joblib
│   │   └── tfidf_matrix.joblib
│   └── risk_classifier/
│       ├── risk_training_data.csv
│       └── risk_text_classifier.joblib
│
├── src/
│   ├── 01_download_real_data.py
│   ├── 02_create_database.py
│   ├── 03_download_documents.py
│   ├── 04_build_document_retriever.py
│   ├── 05_search_documents.py
│   ├── 06_train_risk_classifier.py
│   ├── 07_classify_retrieved_chunks.py
│   └── 08_local_agent.py
│
├── financial_risk.db
├── requirements.txt
└── README.md
```
