from kafka import KafkaConsumer
import json


# =========================
# Kafka configuration
# =========================

KAFKA_SERVER = "localhost:9092"
TOPIC = "fraud_transactions"


# =========================
# Create Kafka consumer
# =========================

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_SERVER,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="fraud-detection-group",
    value_deserializer=lambda value: json.loads(value.decode("utf-8"))
)


print("Waiting for transactions...")
print("Press Ctrl+C to stop.")


# =========================
# Listen for messages
# =========================

for message in consumer:

    transaction = message.value

    print()
    print("==============================")
    print("Transaction received")
    print("==============================")
    print(transaction)