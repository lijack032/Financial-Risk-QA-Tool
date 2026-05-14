from pathlib import Path
import sqlite3
import pandas as pd
import joblib
from dotenv import load_dotenv
from importlib.util import spec_from_file_location, module_from_spec

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_risk.db"
MODEL_PATH = BASE_DIR / "artifacts" / "risk_classifier" / "risk_text_classifier.joblib"

RETRIEVER_PATH = BASE_DIR / "src" / "document_search.py"
spec = spec_from_file_location("search_documents", RETRIEVER_PATH)
search_documents = module_from_spec(spec)
spec.loader.exec_module(search_documents)

FinancialDocumentRetriever = search_documents.FinancialDocumentRetriever

load_dotenv()

retriever = FinancialDocumentRetriever()
classifier = joblib.load(MODEL_PATH)


@tool
def search_financial_documents(query: str) -> str:
    """
    Search public financial documents, including RBC reports and OSFI liquidity guidelines.

    Use this tool for questions about financial documents, RBC disclosures, OSFI guidance,
    liquidity risk, funding risk, LCR, NSFR, market risk, credit risk, counterparty risk,
    regulatory requirements, and disclosure language.
    """

    results = retriever.search(query, top_k=3)
    formatted_results = []

    for i, result in enumerate(results, start=1):
        predicted_label = classifier.predict([result["text"]])[0]

        formatted_results.append(
            f"Result {i}\n"
            f"Source: {result['source']}\n"
            f"Chunk: {result['chunk_number']}\n"
            f"Retrieval score: {result['score']:.4f}\n"
            f"Predicted risk category: {predicted_label}\n"
            f"Excerpt: {result['text'][:900]}"
        )

    return "\n\n".join(formatted_results)


@tool
def query_financial_indicators(question: str) -> str:
    """
    Query the SQLite database of structured financial indicators.

    Use this tool for questions about VIX, SOFR, Treasury rates, credit spreads,
    funding pressure, high-risk days, monthly risk summaries, averages, recent
    indicators, or trends in structured financial data.
    """

    q = question.lower()

    if not DB_PATH.exists():
        return "Database not found. Run src/02_create_database.py first."

    conn = sqlite3.connect(DB_PATH)

    if "monthly" in q or "month" in q or "months" in q:
        sql = """
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
        LIMIT 10;
        """

    elif "average" in q or "summary" in q:
        sql = """
        SELECT
            metric,
            ROUND(value, 4) AS value
        FROM risk_summary;
        """

    elif "latest" in q or "recent" in q:
        sql = """
        SELECT
            date,
            ROUND(vix, 2) AS vix,
            ROUND(three_month_treasury_rate, 2) AS three_month_treasury_rate,
            ROUND(ten_year_treasury_rate, 2) AS ten_year_treasury_rate,
            ROUND(high_yield_credit_spread, 2) AS high_yield_credit_spread,
            ROUND(sofr, 2) AS sofr,
            ROUND(funding_pressure_score, 2) AS funding_pressure_score,
            risk_flag
        FROM financial_indicators
        ORDER BY date DESC
        LIMIT 10;
        """

    else:
        sql = """
        SELECT
            date,
            ROUND(vix, 2) AS vix,
            ROUND(three_month_treasury_rate, 2) AS three_month_treasury_rate,
            ROUND(ten_year_treasury_rate, 2) AS ten_year_treasury_rate,
            ROUND(high_yield_credit_spread, 2) AS high_yield_credit_spread,
            ROUND(sofr, 2) AS sofr,
            ROUND(funding_pressure_score, 2) AS funding_pressure_score,
            risk_flag
        FROM financial_indicators
        WHERE risk_flag = 1
        ORDER BY funding_pressure_score DESC
        LIMIT 10;
        """

    df = pd.read_sql_query(sql, conn)
    conn.close()

    return df.to_string(index=False)


@tool
def classify_financial_risk_text(text: str) -> str:
    """
    Classify a financial text passage into a structured risk category.

    Categories include liquidity, market, interest_rate, credit, and regulatory.
    """

    if not MODEL_PATH.exists():
        return "Risk classifier not found. Run src/06_train_risk_classifier.py first."

    prediction = classifier.predict([text])[0]
    return f"Predicted risk category: {prediction}"


def build_agent():
    """
    Build the LangChain tool-calling agent.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    tools = [
        search_financial_documents,
        query_financial_indicators,
        classify_financial_risk_text,
    ]

    system_prompt = """
You are a Treasury risk research assistant.

Your job is to answer user questions by choosing the right tools.

You have three tools:

1. search_financial_documents
Use this when the question asks about financial documents, RBC reports, OSFI guidelines,
regulatory requirements, LCR, NSFR, liquidity risk, funding risk, credit risk,
market risk, counterparty risk, or disclosure language.

2. query_financial_indicators
Use this when the question asks about structured data, SQL tables, VIX, SOFR,
Treasury rates, credit spreads, funding pressure, high-risk days, monthly summaries,
averages, or trends.

3. classify_financial_risk_text
Use this when the user asks to classify a passage into a risk category.

Important instructions:
- Use one tool if the question only needs one tool.
- Use multiple tools if the question asks for both document evidence and structured indicators.
- In the final answer, briefly state which tool or tools you used.
- Give a concise business-facing answer.
- When using document evidence, mention the source file name.
- Do not claim the funding pressure score is an official RBC, OSFI, Basel, or regulatory metric.
- Treat the funding pressure score as a project-level screening metric.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


def extract_final_answer(result) -> str:
    """
    Extract a readable final answer from the LangChain create_agent result.
    """

    messages = result.get("messages", [])

    if not messages:
        return str(result)

    final_message = messages[-1]
    content = getattr(final_message, "content", None)

    if content is None and isinstance(final_message, dict):
        content = final_message.get("content")

    return str(content)


def main() -> None:
    agent = build_agent()

    questions = [
        "What does RBC say about liquidity risk and funding?",
        "Show the highest funding pressure days from the SQL table.",
        "What regulatory requirements apply to bank liquidity, and which months show elevated funding pressure?",
        "Classify this risk category: deposit outflows increased and high quality liquid assets declined.",
    ]

    for question in questions:
        print("=" * 110)
        print(f"Question: {question}")
        print("=" * 110)

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        print("\nFinal answer:")
        print(extract_final_answer(result))
        print()


if __name__ == "__main__":
    main()
