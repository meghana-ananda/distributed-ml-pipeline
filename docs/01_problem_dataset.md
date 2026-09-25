# Problem & Dataset

## Dataset: Avazu Click-Through Rate Prediction

I picked [Avazu CTR](https://www.kaggle.com/c/avazu-ctr-prediction) for this project: about 40M rows, 10 days of real mobile ad impression logs, roughly 1.5GB compressed / 7GB uncompressed as CSV, 24 categorical features (device, app, site, banner position, anonymized category IDs) plus a binary `click` label.

**Why I'm processing this with a distributed pipeline instead of pandas:**
- 40M rows of high cardinality categorical data (some fields have 100K+ unique values) blow past comfortable single machine RAM once I apply one-hot/hash encoding and joins. Pandas either swaps to disk or OOMs on a typical 16GB dev laptop at this scale.
- The workload is embarrassingly parallel (row-independent feature engineering, groupby aggregations per device/app/site), which is exactly the shape distributed engines like Spark are built to exploit. I wanted a realistic testbed for partitioning strategy, shuffle cost, and cluster scaling tradeoffs, not a toy dataset forced into a big data tool just to check a box.
- The 10 day span lets me do a time based train/test split that mirrors production streaming ingestion, which is why I'm building this as a pipeline that can reprocess new daily partitions incrementally, not a one off notebook.

## Prediction Task

**Task:** Binary classification. I'm predicting whether a mobile ad impression will be clicked.

**Input (single row):**
```
hour=14102100, C1=1005, banner_pos=0, site_id=1fbe01fe, app_id=ecad2386,
device_type=1, device_conn_type=0, C14=15706, C17=1722, ...
```

**Output:**
```
click_probability = 0.037   ->  predicted_label = 0 (no click)
```

I'm treating the model output as a calibrated probability used downstream for ad ranking/bid decisions, not just a hard label.

## Success Metrics

**Model quality:**
- Primary: **Log Loss** (this is the original competition metric, and CTR is heavily imbalanced, so accuracy alone would be misleading)
- Secondary: **AUC-ROC**, to sanity check ranking quality independent of calibration

**Pipeline performance** (this is the actual focus of the portfolio piece):
- **Throughput:** rows/sec processed during distributed feature engineering. I'm targeting sustained throughput above 100K rows/sec on a small cluster/local multi-worker setup.
- **End-to-end runtime:** raw CSV through engineered features to trained model, measured wall clock, tracked across cluster sizes (1 vs 4 vs 8 workers) so I can demonstrate scaling.
- **Resource ceiling:** peak memory per worker, to show the pipeline stays within a fixed, modest budget regardless of dataset size. This is the constraint pandas alone can't guarantee, and it's the core reason I built this as a distributed pipeline.
