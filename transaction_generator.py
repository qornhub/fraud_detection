import random
import time
import json
import os
from kafka import KafkaProducer


KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
TOPIC = "fraud_transactions"


producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)


def generate_normal_transaction():
    old_balance_org = round(random.uniform(500, 20000), 2)
    amount = round(random.uniform(10, old_balance_org * 0.5), 2)

    old_balance_dest = round(random.uniform(500, 10000), 2)

    return {
        "step": random.randint(1, 744),
        "type": random.choice([
            "PAYMENT",
            "CASH_OUT",
            "TRANSFER"
        ]),
        "amount": amount,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": round(old_balance_org - amount, 2),
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": round(old_balance_dest + amount, 2)
    }


def generate_suspicious_transaction():
    old_balance_org = round(random.uniform(300000, 1000000), 2)
    amount = round(random.uniform(200000, 600000), 2)

    old_balance_dest = round(random.uniform(0, 5000), 2)

    return {
        "step": random.randint(1, 744),
        "type": random.choice([
            "TRANSFER",
            "CASH_OUT"
        ]),
        "amount": amount,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": round(old_balance_org - amount, 2),
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": round(old_balance_dest + amount, 2)
    }


try:
    print("Transaction generator started.")
    print("Press Ctrl+C to stop.")
    print()

    while True:

        # Mostly normal transactions,
        # occasionally generate a suspicious one.
        if random.random() < 0.15:
            transaction = generate_suspicious_transaction()
        else:
            transaction = generate_normal_transaction()

        producer.send(TOPIC, value=transaction)
        producer.flush()

        print("==============================")
        print("Transaction sent")
        print("==============================")
        print(transaction)
        print()

        # Simulate transactions arriving over time
        time.sleep(2)


except KeyboardInterrupt:
    print("\nTransaction generator stopped.")


finally:
    producer.close()
