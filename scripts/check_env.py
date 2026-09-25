"""Verifies the local dev environment is ready: Spark starts, MLflow tracking
server is reachable. Prints a pass/fail line per check and exits non-zero on
any failure, so it can be used as a CI or pre-flight gate.
"""

import os
import sys

import requests

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return False


def check_spark() -> None:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("check_env")
        .getOrCreate()
    )
    df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "value"])
    assert df.count() == 2
    spark.stop()


def check_mlflow() -> None:
    response = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=5)
    response.raise_for_status()


def check_xgboost() -> None:
    import xgboost as xgb
    import numpy as np

    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([1, 1, 0, 0])
    model = xgb.XGBClassifier(n_estimators=2, max_depth=2)
    model.fit(X, y)
    model.predict(X)


def main() -> int:
    checks = [
        ("Spark local session starts", check_spark),
        ("MLflow tracking server reachable", check_mlflow),
        ("XGBoost trains and predicts", check_xgboost),
    ]

    results = [check(name, fn) for name, fn in checks]

    print()
    if all(results):
        print("All checks passed.")
        return 0

    print("One or more checks failed. See above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
