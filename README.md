# A Hybrid Deep-Learning Approach to Customer Lifetime Value

## Project Overview

This project builds a machine learning pipeline to predict Customer Lifetime Value (CLV) — how much revenue each customer will generate over the next 28 days — using the UCI Online Retail dataset, a year of UK e-commerce transactions. The core challenge is zero-inflation: most customers buy nothing in any given window, while a small number of high-value customers drive the bulk of revenue. To address this, the pipeline culminates in a two-stage hybrid model that first predicts purchase probability with a calibrated XGBoost classifier, then estimates conditional spend using a stacking ensemble of a gradient boosting model and a sequence-to-sequence GRU neural network. Customer behaviour is encoded through rich snapshot features, daily time-series signals, and 64-dimensional skip-gram embeddings derived from item co-purchase patterns. The two-stage model achieves the best overall performance, capturing over 50% of total spend in the top 30% of predicted customers — making it well-suited for targeting and marketing prioritisation.

## How to Run clv.ipynb

### Requirements

- Python 3.12
- The following packages:

```
xgboost
tensorflow
pandas
numpy
scipy
scikit-learn
openpyxl
```

Install them with:

```bash
pip install xgboost tensorflow pandas numpy scipy scikit-learn openpyxl
```

---

### Running on Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `clv_new (2).ipynb`
4. In the first code cell, uncomment the install line:
   ```python
   !pip install xgboost tensorflow pandas numpy scipy scikit-learn openpyxl
   ```
5. Run all cells in order: **Runtime → Run all**

The notebook downloads the dataset automatically from GitHub — no file uploads needed.

> **Tip:** Use a GPU runtime for faster training. Go to **Runtime → Change runtime type → T4 GPU**.

---

### What Gets Cached After a Run

After the first full run, the following `.pkl` files are saved to the working directory. On future runs, these are loaded instead of recomputed, saving significant time.

| File | What it stores | Slow to recompute? |
|---|---|---|
| `customer_order_dt_UKRetail.pkl` | Cleaned and preprocessed raw order data | No |
| `ts_panel_UKRetail.pkl` | Daily time-series panel (customer × day) with sequential features | Moderate |
| `cust_feats_all_UKRetail.pkl` | Snapshot feature table across all time periods | **Yes** (~minutes) |
| `target_all_UKRetail.pkl` | Target profit values for each snapshot | **Yes** |
| `subcat_sparse_UKRetail.pkl` | Sparse subcategory purchase matrix | **Yes** |
| `skipgram_embeddings_UKRetail_train.pkl` | 64-dim customer embeddings from skip-gram training | **Yes** (~minutes) |

If you are running on Colab and want to reuse the cache across sessions, download these files after the first run and re-upload them before future runs. Otherwise Colab will recompute everything from scratch each session.
