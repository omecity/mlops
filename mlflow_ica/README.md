# MLflow Linear Models with x1 and x2

This project trains and tracks simple linear regression models using [MLflow](https://mlflow.org/).
The script automatically evaluates models on subsets of the columns **x1** and **x2** against the target column **y**, then logs metrics and models to MLflow.

---

## 📂 Requirements

* Python 3.8+
* Packages:

  * numpy
  * pandas
  * scikit-learn
  * mlflow

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the script by providing a CSV file with columns `y`, `x1`, and `x2`:

```bash
python mlflow_ica.py <file.csv>
```

Example:

```bash
python mlflow_ica.py reg2.csv
```

The script will:

1. Read the dataset and select columns `y`, `x1`, `x2`.
2. Train separate linear regression models using different subsets of `x1` and `x2`.
3. Log metrics (MAE, MSE, RMSE, R²) and register the models in MLflow.

---

## 📊 View Results

Start the MLflow UI to inspect runs, metrics, and models:

```bash
mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

You will see the experiment: **`linear_models_with_columns_x1_x2`**, with one run for each model trained.

---
