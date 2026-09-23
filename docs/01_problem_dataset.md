# Problem & Dataset

## Dataset: Avazu Click-Through Rate Prediction

[Avazu CTR](https://www.kaggle.com/c/avazu-ctr-prediction) — **~40M rows**, 10 days of real mobile ad impression logs, ~1.5GB compressed / ~7GB uncompressed CSV, 24 categorical features (device, app, site, banner position, anonymized category IDs) plus a binary `click` label.

**Why distributed processing, not pandas:**
- 40M rows of high-cardinality categorical data (some fields have 100K+ unique values) blow past comfortable single-machine RAM once one-hot/hash encoding and joins are applied — pandas either swaps to disk or OOMs on a typical 16GB dev laptop.
- The workload is embarrassingly parallel (row-independent feature engineering, groupby aggregations per device/app/site), which is exactly the shape distributed engines (Spark / Dask) are built to exploit — it's a realistic testbed for partitioning strategy, shuffle cost, and cluster-scaling tradeoffs, not just a toy dataset forced into a big-data tool.
- Time-based train/test split across 10 days mirrors production streaming ingestion, motivating a pipeline (not a one-off notebook) that can reprocess new daily partitions incrementally.

## Prediction Task

**Task:** Binary classification — predict whether a mobile ad impression will be clicked.

**Input (single row):**
```
hour=14102100, C1=1005, banner_pos=0, site_id=1fbe01fe, app_id=ecad2386,
device_type=1, device_conn_type=0, C14=15706, C17=1722, ...
```

**Output:**
```
click_probability = 0.037   →  predicted_label = 0 (no click)
```

Model output is a calibrated probability used downstream for ad-ranking/bid decisions, not just a hard label.

## Success Metrics

**Model quality:**
- Primary: **Log Loss** (competition metric; CTR is heavily imbalanced, so accuracy is misleading)
- Secondary: **AUC-ROC**, to sanity-check ranking quality independent of calibration

**Pipeline performance** (the portfolio's actual focus):
- **Throughput:** rows/sec processed during distributed feature engineering (target: sustain >100K rows/sec on a small cluster/local multi-worker setup)
- **End-to-end runtime:** raw CSV → engineered features → trained model, measured wall-clock, tracked across cluster sizes (1 vs 4 vs 8 workers) to demonstrate scaling
- **Resource ceiling:** peak memory per worker, to show the pipeline stays within a fixed, modest budget regardless of dataset size — the point pandas alone can't guarantee here
