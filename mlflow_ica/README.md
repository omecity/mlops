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
