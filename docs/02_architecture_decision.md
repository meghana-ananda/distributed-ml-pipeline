# Architecture Decisions

## PySpark vs Dask

**Decision: PySpark.**

The Avazu dataset (about 40M rows, high cardinality categorical columns, ~7GB uncompressed) is large enough to need out of core, partitioned processing, but small enough that either engine could technically handle it on a single beefy machine. The deciding factors are:

- Spark's mature handling of wide groupby and join operations on high cardinality categorical keys (site_id, app_id, device_id) is more battle tested than Dask's, and this dataset leans heavily on those operations for feature engineering (frequency encoding, target encoding, per device aggregates).
- Spark is the de facto standard in production data engineering, so this choice demonstrates a skill that maps directly to how most companies run distributed ETL. Dask is closer to a "scale up pandas" tool, which is a weaker signal for a portfolio aimed at showing distributed systems skill.
- Local development is realistic either way using `local[*]` mode with a capped executor memory setting, so the same code that runs on a laptop with 8 to 16GB RAM would also run unmodified on a real cluster (EMR, Databricks, or a Kubernetes Spark operator). That "write once, run at any scale" property is the strongest argument for Spark in a portfolio piece meant to show production readiness.

I considered Dask for its tighter pandas API compatibility and pure Python stack, but for this dataset and for demonstrating distributed systems fundamentals, I chose Spark.

## Airflow vs Kubeflow Pipelines

**Decision: Airflow.**

- Kubeflow Pipelines assumes a Kubernetes cluster as the baseline environment. Standing up and maintaining a local Kubernetes cluster just to orchestrate a portfolio pipeline adds infrastructure overhead that is disproportionate to the project's scope.
- Airflow can be run locally with Docker Compose in minutes, has first class Spark and Python operators, and is the most widely used orchestrator in data engineering job postings, which makes it a more transferable skill to showcase.
- The pipeline's actual orchestration needs are straightforward: run a Spark feature job, wait for it to complete, run a training step, log to MLflow, and conditionally deploy. Airflow's DAG model expresses this cleanly without needing Kubeflow's container per step complexity.

I considered Kubeflow Pipelines, and it would fit better if this project were built around a Kubernetes native ML platform from the start, but for a self contained portfolio project I chose Airflow to keep the setup approachable while still showing real orchestration skill.

## XGBoost vs a Simple Neural Network

**Decision: XGBoost.**

- The feature set is almost entirely categorical (device type, connection type, app id, site id, banner position) with only a handful of numeric or hashed columns. Gradient boosted trees handle high cardinality categorical features (via target or frequency encoding) more effectively and with far less tuning than a neural network, which would typically need embedding layers to do the same job well.
- Click through rate datasets are tabular by nature, and XGBoost is the established strong baseline for this problem type in both industry and Kaggle competitions, including the original Avazu competition itself.
- Training time and resource cost are lower with XGBoost on a local machine, which matters for iteration speed during development and for keeping the pipeline runnable without GPU access.

I see a simple neural network with embedding layers as a valid extension for later work, but I did not pick it as the first model given the tabular, categorical heavy nature of the features and the local resource constraints.

## Wiring MLflow into the Spark Job and the Training Step

MLflow tracking is split across two stages of the pipeline, both writing to the same tracking server so that a full run can be traced end to end:

**Spark feature engineering job:**
- Logs run level parameters: input row count, date range processed, number of partitions, and the encoding strategy used for categorical columns.
- Logs metrics about the job itself rather than the model: rows processed per second, job wall clock duration, and output row count after feature engineering.
- Logs the output feature set (schema and a small sample) as an artifact so a later training run can be traced back to the exact feature version it consumed.

**Training step (XGBoost):**
- Starts a new MLflow run nested under (or tagged with) the same pipeline run id as the feature job, so both stages are linked.
- Logs hyperparameters (max depth, learning rate, number of estimators, regularization terms).
- Logs metrics: log loss and AUC on the validation split, plus training duration.
- Logs the trained model itself using `mlflow.xgboost.log_model`, along with the feature encoding artifacts needed to serve it, so the FastAPI service can load a single MLflow model URI at deploy time.

Both stages point at the same MLflow tracking URI (a local or lightweight hosted tracking server backed by a database and artifact store), which means the entire pipeline, from raw data through a deployable model, is reconstructable from MLflow's UI alone.

## Pipeline Summary Diagram

```
                 +----------------------+
                 |   Raw Avazu CSV      |
                 |   (~40M rows)        |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |   PySpark Job        |
                 |   feature engineering|
                 |   (partitioned)      |
                 +----------+-----------+
                            |  logs params/metrics/artifacts
                            v
                 +----------------------+
                 |     MLflow           |
                 |  Tracking Server     |
                 +----------+-----------+
                            ^
                            |  logs hyperparams/metrics/model
                 +----------+-----------+
                 |   XGBoost Training   |
                 |   step               |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  Airflow DAG         |
                 |  orchestrates:       |
                 |  Spark job then      |
                 |  training step then  |
                 |  deploy step         |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  FastAPI service     |
                 |  loads model from    |
                 |  MLflow model URI    |
                 |  (in Docker)         |
                 +----------------------+
```
