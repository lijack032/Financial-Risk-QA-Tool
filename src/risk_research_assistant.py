from pathlib import Path
import sqlite3
import pandas as pd
import joblib
from importlib.util import spec_from_file_location, module_from_spec

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_risk.db"
MODEL_PATH = BASE_DIR / "artifacts" / "risk_classifier" / "risk_text_classifier.joblib"

RETRIEVER_PATH = BASE_DIR / "src" / "05_search_documents.py"
spec = spec_from_file_location("search_documents", RETRIEVER_PATH)
search_documents = module_from_spec(spec)
spec.loader.exec_module(search_documents)

FinancialDocumentRetriever = search_documents.FinancialDocumentRetriever


class LocalAgenticFinancialAnalyst:
    """
    A local agent controller for financial document and risk analysis.

    This demonstrates agentic AI patterns without requiring an external LLM API:
    - planning: decide which tools are needed
    - tool use: call document search, SQL lookup, and classifier
    - memory: store previous questions and answers
    - multi-step reasoning: combine outputs into a business-facing summary
    """

    def __init__(self) -> None:
        self.retriever = FinancialDocumentRetriever()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Risk classifier not found. Run src/06_train_risk_classifier.py first."
            )

        self.classifier = joblib.load(MODEL_PATH)
        self.memory = []

    def plan(self, question: str) -> list[str]:
        """
        Select tools based on the user's question.
        """

        q = question.lower()
        tools = []

        document_keywords = [
            "document",
            "annual report",
            "pillar",
            "osfi",
            "guideline",
            "policy",
            "rbc say",
            "lcr",
            "nsfr",
            "liquidity",
            "funding",
            "market risk",
            "credit risk",
            "counterparty",
            "regulatory",
            "requirement",
            "requirements",
            "basel",
        ]

        sql_keywords = [
            "sql",
            "table",
            "data",
            "indicator",
            "indicators",
            "vix",
            "sofr",
            "spread",
            "rate",
            "funding pressure",
            "high risk days",
            "highest",
            "average",
            "monthly",
            "month",
            "months",
            "trend",
            "2025",
            "2024",
            "2023",
            "elevated",
        ]

        classify_keywords = [
            "classify",
            "category",
            "risk type",
            "risk category",
        ]

        wants_sql = any(keyword in q for keyword in sql_keywords)
        wants_document = any(keyword in q for keyword in document_keywords)

        # If the user explicitly asks for SQL/table/data output, prioritize SQL.
        if wants_sql:
            tools.append("sql_query")

        sql_only_phrases = [
            "from the sql table",
            "from sql",
            "show the highest",
            "highest funding pressure days",
            "monthly table",
        ]

        is_sql_only = any(phrase in q for phrase in sql_only_phrases)
        
        
        if wants_document and not is_sql_only:
            tools.insert(0, "document_search")

        if any(keyword in q for keyword in classify_keywords):
            tools.append("risk_classifier")

        if not tools:
            tools = ["document_search", "sql_query"]

        return tools

    def document_search_tool(self, question: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve relevant financial document chunks and classify each chunk.
        """

        results = self.retriever.search(question, top_k=top_k)

        enriched_results = []

        for result in results:
            predicted_label = self.classifier.predict([result["text"]])[0]

            enriched_results.append(
                {
                    "source": result["source"],
                    "chunk_number": result["chunk_number"],
                    "score": result["score"],
                    "predicted_risk_category": predicted_label,
                    "text": result["text"],
                }
            )

        return enriched_results

    def sql_query_tool(self, question: str) -> pd.DataFrame:
        """
        Query the SQLite financial risk database using simple intent routing.
        """

        q = question.lower()

        if not DB_PATH.exists():
            raise FileNotFoundError(
                "financial_risk.db not found. Run src/02_create_database.py first."
            )

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

        return df

    def risk_classifier_tool(self, question: str) -> str:
        """
        Classify a user-provided text passage into a financial risk category.
        """

        cleaned_text = (
            question.replace("classify", "")
            .replace("Classify", "")
            .replace("risk category", "")
            .replace("risk type", "")
            .strip()
        )

        prediction = self.classifier.predict([cleaned_text])[0]

        return prediction

    def summarize_outputs(
        self,
        question: str,
        plan: list[str],
        document_results: list[dict] | None,
        sql_results: pd.DataFrame | None,
        classification_result: str | None,
    ) -> str:
        """
        Generate a business-facing summary from tool outputs.
        """

        summary_lines = []

        summary_lines.append(f"Question: {question}")
        summary_lines.append(f"Agent plan: {' -> '.join(plan)}")
        summary_lines.append("")

        if document_results:
            top_categories = [
                item["predicted_risk_category"] for item in document_results
            ]

            summary_lines.append("Document evidence:")
            summary_lines.append(
                f"- Retrieved {len(document_results)} relevant document chunks."
            )
            summary_lines.append(
                "- Predicted risk categories from retrieved evidence: "
                f"{', '.join(top_categories)}."
            )

            top = document_results[0]
            excerpt = top["text"][:500].replace("\n", " ")

            summary_lines.append(
                f"- Most relevant source: {top['source']} "
                f"(chunk {top['chunk_number']}, score {top['score']:.3f})."
            )
            summary_lines.append(f"- Key excerpt: {excerpt}...")

        if sql_results is not None:
            summary_lines.append("")
            summary_lines.append("Structured indicator evidence:")
            summary_lines.append(sql_results.to_string(index=False))

            if "funding_pressure_score" in sql_results.columns:
                max_score = sql_results["funding_pressure_score"].max()
                summary_lines.append(
                    f"- The highest funding pressure score shown is {max_score}, "
                    "which suggests a period of elevated market or funding stress "
                    "under the project rule."
                )

            if "high_risk_share" in sql_results.columns:
                top_month = sql_results.iloc[0]["month"]
                top_share = sql_results.iloc[0]["high_risk_share"]
                summary_lines.append(
                    f"- The highest-risk month shown is {top_month}, "
                    f"with a high-risk share of {top_share}."
                )

        if classification_result:
            summary_lines.append("")
            summary_lines.append("Risk classification:")
            summary_lines.append(f"- Predicted risk category: {classification_result}")

        summary_lines.append("")
        summary_lines.append("Business-facing interpretation:")

        if document_results and sql_results is not None:
            top_source = document_results[0]["source"]
            top_category = document_results[0]["predicted_risk_category"]

            interpretation = (
                f"The retrieved document evidence suggests that the question is mainly "
                f"related to {top_category} risk, with the most relevant evidence coming "
                f"from {top_source}. "
            )

            if "high_risk_share" in sql_results.columns:
                top_month = sql_results.iloc[0]["month"]
                top_share = sql_results.iloc[0]["high_risk_share"]
                interpretation += (
                    f"The structured indicators show that {top_month} had the highest "
                    f"share of high-risk observations, with a high-risk share of "
                    f"{top_share}. "
                )

            if "average_funding_pressure" in sql_results.columns:
                top_pressure = sql_results.iloc[0]["average_funding_pressure"]
                interpretation += (
                    f"Its average funding pressure score was {top_pressure}, suggesting "
                    f"that this period should be reviewed more closely under the "
                    f"project's risk-screening rule. "
                )

            if "funding_pressure_score" in sql_results.columns:
                top_date = sql_results.iloc[0]["date"]
                top_score = sql_results.iloc[0]["funding_pressure_score"]
                interpretation += (
                    f"The highest returned daily funding pressure observation occurred "
                    f"on {top_date}, with a score of {top_score}. "
                )

            interpretation += (
                "Together, the document and SQL evidence help connect regulatory or "
                "risk-management language with observable market conditions."
            )

            summary_lines.append(interpretation)

        elif document_results:
            top_source = document_results[0]["source"]
            top_category = document_results[0]["predicted_risk_category"]

            summary_lines.append(
                f"The retrieved evidence is mainly related to {top_category} risk. "
                f"The most relevant source is {top_source}, which provides the strongest "
                f"textual support for the answer."
            )

        elif sql_results is not None:
            if "funding_pressure_score" in sql_results.columns:
                max_score = sql_results["funding_pressure_score"].max()
                summary_lines.append(
                    f"The structured indicators show elevated funding pressure, with the "
                    f"highest funding pressure score in the returned results equal to "
                    f"{max_score}. This suggests a period of elevated market or funding "
                    f"stress under the project rule."
                )

            elif "high_risk_share" in sql_results.columns:
                top_month = sql_results.iloc[0]["month"]
                top_share = sql_results.iloc[0]["high_risk_share"]

                summary_lines.append(
                    f"The monthly SQL results show that {top_month} had the highest "
                    f"share of high-risk observations, with a high-risk share of "
                    f"{top_share}. This can help identify periods that require closer "
                    f"review."
                )

            else:
                summary_lines.append(
                    "The SQL results summarize financial indicators that can support "
                    "risk monitoring and business-facing analysis."
                )

        elif classification_result:
            summary_lines.append(
                f"The provided text was classified as {classification_result} risk, "
                f"which can help convert unstructured financial language into a "
                f"structured risk category."
            )

        return "\n".join(summary_lines)

    def answer(self, question: str) -> str:
        """
        Run the full local agent workflow.
        """

        plan = self.plan(question)

        document_results = None
        sql_results = None
        classification_result = None

        if "document_search" in plan:
            document_results = self.document_search_tool(question)

        if "sql_query" in plan:
            sql_results = self.sql_query_tool(question)

        if "risk_classifier" in plan:
            classification_result = self.risk_classifier_tool(question)

        final_answer = self.summarize_outputs(
            question=question,
            plan=plan,
            document_results=document_results,
            sql_results=sql_results,
            classification_result=classification_result,
        )

        self.memory.append(
            {
                "question": question,
                "plan": plan,
                "answer": final_answer,
            }
        )

        return final_answer


def main() -> None:
    agent = LocalAgenticFinancialAnalyst()

    questions = [
        "What does RBC say about liquidity risk and funding?",
        "Show the highest funding pressure days from the SQL table.",
        "What regulatory requirements apply to bank liquidity, and which months show elevated funding pressure?",
        "Classify this risk category: deposit outflows increased and high quality liquid assets declined.",
    ]

    for question in questions:
        print("=" * 110)
        print(agent.answer(question))
        print("=" * 110)
        print()


if __name__ == "__main__":
    main()
