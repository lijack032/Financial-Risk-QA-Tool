from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts" / "risk_classifier"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "risk_text_classifier.joblib"
TRAINING_DATA_PATH = ARTIFACT_DIR / "risk_training_data.csv"


def build_training_data() -> pd.DataFrame:
    """
    Build a small labeled training dataset for financial risk text classification.

    Labels:
    - liquidity
    - market
    - interest_rate
    - credit
    - regulatory
    """

    examples = [
        # Liquidity risk
        ("The bank monitors liquidity coverage ratio, net stable funding ratio, and high quality liquid assets.", "liquidity"),
        ("Deposit outflows increased and short-term wholesale funding reliance became more concentrated.", "liquidity"),
        ("The institution maintains contingency funding plans to address unexpected cash outflows.", "liquidity"),
        ("Liquidity risk arises when the bank cannot meet obligations without incurring unacceptable losses.", "liquidity"),
        ("Treasury reviewed funding concentration, available liquid assets, and stress liquidity scenarios.", "liquidity"),
        ("The LCR and NSFR are used to assess the adequacy of the liquidity buffer.", "liquidity"),

        # Market risk
        ("Market volatility increased sharply and equity prices declined across major indexes.", "market"),
        ("The trading portfolio is exposed to changes in interest rates, foreign exchange, and credit spreads.", "market"),
        ("Value-at-risk measures potential losses from adverse market movements.", "market"),
        ("Credit spreads widened during the stress period, reflecting weaker market liquidity.", "market"),
        ("The VIX rose as investors repriced risk assets under uncertain market conditions.", "market"),
        ("Market risk reflects the impact of changing prices, spreads, and volatility on portfolio value.", "market"),

        # Interest rate risk
        ("The yield curve steepened as long-term rates increased relative to short-term rates.", "interest_rate"),
        ("Interest rate sensitivity measures how earnings respond to parallel rate shocks.", "interest_rate"),
        ("Deposit beta assumptions affect how quickly funding costs reprice when rates rise.", "interest_rate"),
        ("Duration gaps can create exposure to changes in the level and shape of the yield curve.", "interest_rate"),
        ("Net interest income may decline if liabilities reprice faster than assets.", "interest_rate"),
        ("The bank tested earnings sensitivity under rising and falling interest rate scenarios.", "interest_rate"),

        # Credit risk
        ("Credit risk arises when a borrower or counterparty fails to meet contractual obligations.", "credit"),
        ("The allowance for credit losses increased because of weaker borrower performance.", "credit"),
        ("Loan portfolios are monitored for default probability, exposure at default, and loss severity.", "credit"),
        ("Counterparty exposure increased due to deteriorating credit quality.", "credit"),
        ("Impaired loans and provisions rose during the reporting period.", "credit"),
        ("The bank evaluates collateral, borrower capacity, and concentration risk in the credit portfolio.", "credit"),

        # Regulatory risk
        ("Regulatory capital requirements include common equity tier one and risk-weighted asset measures.", "regulatory"),
        ("OSFI guidelines require banks to maintain liquidity metrics above supervisory minimums.", "regulatory"),
        ("Basel III introduced liquidity standards including LCR and NSFR.", "regulatory"),
        ("Pillar 3 disclosures provide information about capital adequacy and risk-weighted assets.", "regulatory"),
        ("The institution must comply with regulatory reporting, stress testing, and capital rules.", "regulatory"),
        ("Supervisory requirements affect liquidity management, capital planning, and public disclosures.", "regulatory"),
    ]

    return pd.DataFrame(examples, columns=["text", "label"])


def train_classifier(df: pd.DataFrame) -> Pipeline:
    """
    Train a Scikit-learn text classification pipeline.
    """

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.30,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            ("classifier", LinearSVC()),
        ]
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("Classification report:")
    print(classification_report(y_test, predictions))

    print("Confusion matrix:")
    labels = sorted(df["label"].unique())
    print(pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels))

    return model


def main() -> None:
    df = build_training_data()

    df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"Saved training data to: {TRAINING_DATA_PATH}")

    model = train_classifier(df)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained classifier to: {MODEL_PATH}")


if __name__ == "__main__":
    main()