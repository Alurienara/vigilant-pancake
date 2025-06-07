import json

def test_accuracy_threshold():
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    assert metrics["accuracy"] > 0.7, f"Accuracy too low: {metrics['accuracy']}"