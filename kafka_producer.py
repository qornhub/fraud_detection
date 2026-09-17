from kafka import KafkaProducer
import json


# =========================
# Kafka configuration
# =========================

KAFKA_SERVER = "localhost:9092"
TOPIC = "fraud_transactions"


# =========================
# Create Kafka producer
# =========================

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)


# =========================
# Test transaction
# =========================
#normal transction

#transaction = {
 #   "step": 1,
  #  "type": "TRANSFER",
  #  "amount": 1000,
   # "oldbalanceOrg": 5000,
   # "newbalanceOrig": 4000,
   # "oldbalanceDest": 1000,
   # "newbalanceDest": 2000
#}
#fraud transaction

transaction = {
    "step": 100,
    "type": "TRANSFER",
    "amount": 5000000,
    "oldbalanceOrg": 5000000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 0
}

# =========================
# Send transaction
# =========================

producer.send(
    TOPIC,
    value=transaction
)

producer.flush()

print("Transaction sent to Kafka:")
print(transaction)

producer.close()