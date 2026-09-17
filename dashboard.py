import json
import time
import os
import pandas as pd
import streamlit as st
from kafka import KafkaConsumer


# ============================================================
# Configuration
# ============================================================

KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
RESULT_TOPIC = "fraud_results"
THRESHOLD = 0.992


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        margin-bottom: 1.5rem;
    }

    .fraud-box {
        padding: 1rem;
        border-radius: 0.6rem;
        background: #ffe5e5;
        border: 1px solid #ff9b9b;
    }

    .normal-box {
        padding: 1rem;
        border-radius: 0.6rem;
        background: #e8f7ed;
        border: 1px solid #9ad8ad;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Kafka connection
# ============================================================

@st.cache_resource
def create_consumer():

    return KafkaConsumer(
        RESULT_TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="fraud-dashboard-consumer-all",
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
    )


consumer = create_consumer()


# ============================================================
# Session state
# ============================================================

if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "last_update" not in st.session_state:
    st.session_state.last_update = None


# ============================================================
# Receive new Kafka results
# ============================================================

records = consumer.poll(
    timeout_ms=100,
    max_records=50,
)

for _, messages in records.items():

    for message in messages:

        result = message.value

        # ----------------------------------------------------
        # Get SHAP explanation
        # ----------------------------------------------------

        shap_explanations = result.get(
            "shap_explanations",
            []
        )

        # ----------------------------------------------------
        # Create transaction record
        # ----------------------------------------------------

        transaction = {
            "Time": time.strftime("%H:%M:%S"),

            "Type": result.get(
                "transaction_type",
                "UNKNOWN"
            ),

            "Amount": float(
                result.get("amount", 0)
            ),

            "Probability": float(
                result.get("probability", 0)
            ),

            "Prediction": result.get(
                "prediction",
                "UNKNOWN"
            ),

            "SHAP": shap_explanations,
        }

        # ----------------------------------------------------
        # Store transaction
        # ----------------------------------------------------

        st.session_state.transactions.append(
            transaction
        )

        # ----------------------------------------------------
        # Update timestamp
        # ----------------------------------------------------

        st.session_state.last_update = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )


# ============================================================
# Auto refresh
# ============================================================

try:
    fragment = st.fragment
except AttributeError:
    fragment = None


# ============================================================
# Dashboard content
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ Real-Time Fraud Detection Dashboard'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Kafka → ML Model → SHAP → Dashboard'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Check whether transactions exist
# ============================================================

transactions = st.session_state.transactions


if not transactions:

    st.info(
        "Waiting for fraud detection results from Kafka..."
    )

    st.caption(
        f"Listening to topic: {RESULT_TOPIC}"
    )


else:

    # ========================================================
    # Convert transactions to DataFrame
    # ========================================================

    df = pd.DataFrame(transactions)


    # ========================================================
    # KPI calculations
    # ========================================================

    total = len(df)

    fraud_count = int(
        (df["Prediction"] == "FRAUD").sum()
    )

    normal_count = int(
        (df["Prediction"] == "NORMAL").sum()
    )

    fraud_rate = (
        fraud_count / total * 100
        if total > 0
        else 0
    )

    total_amount = df["Amount"].sum()


    # ========================================================
    # KPI cards
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total
    )

    col2.metric(
        "Fraud Detected",
        fraud_count
    )

    col3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    col4.metric(
        "Transaction Amount",
        f"RM {total_amount:,.2f}"
    )


    st.divider()


    # ========================================================
    # Charts
    # ========================================================

    left, right = st.columns(2)


    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Prediction Distribution"
        )

        prediction_counts = (
            df["Prediction"]
            .value_counts()
        )

        st.bar_chart(
            prediction_counts
        )


    # --------------------------------------------------------
    # Transaction type
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Transaction Type"
        )

        type_counts = (
            df["Type"]
            .value_counts()
        )

        st.bar_chart(
            type_counts
        )


    st.divider()


    # ========================================================
    # Probability chart
    # ========================================================

    st.subheader(
        "Fraud Probability"
    )

    probability_df = df[
        ["Probability"]
    ].copy()

    probability_df.index = range(
        1,
        len(probability_df) + 1
    )

    st.line_chart(
        probability_df
    )

    st.caption(
        f"Fraud threshold = {THRESHOLD:.3f}"
    )


    st.divider()


    # ========================================================
    # All transactions
    # ========================================================

    st.subheader(
        "All Transactions"
    )

    display_df = df[
        [
            "Time",
            "Type",
            "Amount",
            "Probability",
            "Prediction",
        ]
    ].copy()

    display_df["Amount"] = (
        display_df["Amount"]
        .map(lambda x: f"RM {x:,.2f}")
    )

    display_df["Probability"] = (
        display_df["Probability"]
        .map(lambda x: f"{x:.6f}")
    )

    # Show newest transaction first
    st.dataframe(
        display_df.iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )


    st.divider()


    # ========================================================
    # Fraud alerts
    # ========================================================

    fraud_df = df[
        df["Prediction"] == "FRAUD"
    ]

    st.subheader(
        "🚨 Fraud Alerts"
    )


    if fraud_df.empty:

        st.success(
            "No fraud transactions detected."
        )


    else:

        st.warning(
            f"{len(fraud_df)} fraud transaction(s) detected."
        )

        # Show ALL fraud transactions
        for _, row in fraud_df.iloc[::-1].iterrows():

            st.error(
                f"FRAUD | {row['Type']} | "
                f"RM {row['Amount']:,.2f} | "
                f"Probability: {row['Probability']:.6f}"
            )


    st.divider()


    # ========================================================
    # SHAP explanation
    # ========================================================

    st.subheader(
        "Explainable AI — SHAP"
    )

    selected_index = st.selectbox(

        "Select a transaction to inspect",

        options=list(
            range(len(df))
        ),

        format_func=lambda i: (

            f"{i + 1}. "
            f"{df.iloc[i]['Type']} | "
            f"RM {df.iloc[i]['Amount']:,.2f} | "
            f"{df.iloc[i]['Prediction']} | "
            f"P={df.iloc[i]['Probability']:.6f}"

        ),
    )


    selected = df.iloc[selected_index]

    shap_data = selected["SHAP"]


    if shap_data:

        shap_df = pd.DataFrame(
            shap_data
        )


        shap_df["abs_value"] = (
            shap_df["shap_value"].abs()
        )


        shap_df = (
            shap_df
            .sort_values(
                "abs_value",
                ascending=False
            )
            .head(10)
        )


        chart_df = shap_df[
            [
                "feature",
                "shap_value",
            ]
        ].set_index(
            "feature"
        )


        st.bar_chart(
            chart_df
        )


        st.dataframe(

            shap_df[
                [
                    "feature",
                    "shap_value",
                    "direction",
                ]
            ],

            use_container_width=True,

            hide_index=True,
        )


    else:

        st.info(
            "No SHAP explanation was included in this result."
        )


    # ========================================================
    # Status
    # ========================================================

    st.divider()

    col1, col2 = st.columns(2)


    with col1:

        st.caption(
            f"Kafka: {KAFKA_SERVER}"
        )


    with col2:

        st.caption(
            f"Last update: "
            f"{st.session_state.last_update or 'Waiting...'}"
        )


# ============================================================
# Refresh every 2 seconds
# ============================================================

time.sleep(2)

st.rerun()
