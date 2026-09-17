from kafka import KafkaConsumer
import json


# ==========================================
# Configuration
# ==========================================

KAFKA_SERVER = "localhost:9092"
RESULT_TOPIC = "fraud_results"


# ==========================================
# Create Kafka consumer
# ==========================================

consumer = KafkaConsumer(
    RESULT_TOPIC,
    bootstrap_servers=KAFKA_SERVER,

    # Only receive new results
    auto_offset_reset="latest",

    enable_auto_commit=True,

    group_id="fraud-dashboard-test",

    value_deserializer=lambda value:
        json.loads(value.decode("utf-8"))
)


# ==========================================
# Start consumer
# ==========================================

print()
print("========================================")
print("       FRAUD RESULT CONSUMER")
print("========================================")
print(f"Kafka : {KAFKA_SERVER}")
print(f"Topic : {RESULT_TOPIC}")
print()
print("Waiting for ML results...")
print("Press Ctrl+C to stop.")
print("========================================")


# ==========================================
# Receive results
# ==========================================

try:

    for message in consumer:

        result = message.value

        print()
        print("========================================")
        print("          RESULT RECEIVED")
        print("========================================")

        print(
            f"Transaction Type : "
            f"{result['transaction_type']}"
        )

        print(
            f"Amount           : "
            f"{result['amount']:.2f}"
        )

        print(
            f"Probability      : "
            f"{result['probability']:.6f}"
        )

        print(
            f"Threshold        : "
            f"{result['threshold']}"
        )

        print(
            f"Prediction       : "
            f"{result['prediction']}"
        )

        print()
        print("----------------------------------------")
        print("SHAP EXPLANATIONS")
        print("----------------------------------------")

        for explanation in result["shap_explanations"]:

            print(
                f"{explanation['feature']:28s} "
                f"{explanation['shap_value']: .6f} "
                f"→ {explanation['direction']}"
            )

        print("========================================")


except KeyboardInterrupt:

    print()
    print("========================================")
    print("Result consumer stopped.")
    print("========================================")


finally:

    consumer.close()