# Linear Regression from Scratch in Python

**A custom ordinary least squares (OLS) regression library, tested against Statsmodels and applied to real-world business analytics problems.**

This repository contains `lm_manual`, a Python class for fitting **simple and multiple linear regression** using NumPy, along with a validation notebook and two end-to-end business analysis projects. The goal is to demonstrate how regression works under the hood—not just how to call a modeling package—and how to translate statistical results into business insights.

## Repository Contents

``` text
.
├── lm_manual_class.py
├── test.ipynb
├── TV_Advertising_Sales_Analysis.ipynb
├── King_County_Housing_Multiple_Regression.ipynb
├── README.md
└── LICENSE                        
```

| File | Description |
|----|----|
| `lm_manual_class.py` | Custom OLS regression class for one or more numeric predictors |
| `test.ipynb` | Tests simple and multiple regression results against Statsmodels, including missing-data handling, predictions, intervals, and diagnostics |
| `TV_Advertising_Sales_Analysis.ipynb` | Business analysis of TV advertising expenditure and product sales |
| `King_County_Housing_Multiple_Regression.ipynb` | Business analysis of property characteristics and historical King County home prices |

## Features of `lm_manual`

- **Simple and multiple regression:** Fits one or more numeric predictors with an intercept using NumPy least squares.
- **Regression summaries:** Coefficients, standard errors, t-statistics, p-values, 95% confidence intervals, residual standard error, R², adjusted R², and the overall F-test.
- **Data validation:** Accepts NumPy arrays and Pandas objects, retains predictor names, drops rows with non-finite values, and reports how many were removed.
- **Predictions:** Point estimates, confidence intervals for the mean response, and conventional prediction intervals for individual observations.
- **Multicollinearity:** Variance inflation factors (VIF) through `model.vif()`.
- **Influence diagnostics:** Leverage, standardized residuals, and Cook's distance are calculated and stored by the fitted model.
- **Visualization:** A fitted-line plot for simple regression and a four-panel diagnostic figure for simple or multiple regression.
- **Large-data plotting:** Diagnostic plots can sample observations for display while keeping model estimation based on all valid observations.

### R-style diagnostic plots

The `plot_diagnostics()` method is **inspired by the familiar four-panel regression diagnostics produced by R's `plot(lm(...))`**. It displays:

1.  **Residuals vs Fitted** — patterns, nonlinearity, and changes in residual spread.
2.  **Normal Q-Q** — departures from approximately normal residuals.
3.  **Scale-Location** — whether residual spread changes across fitted values.
4.  **Residuals vs Leverage** — potentially influential observations, with Cook's distance reference contours when visible.

The Python implementation uses **Matplotlib, Seaborn, SciPy, and LOWESS**. It is designed to offer a comparable diagnostic workflow, not to reproduce R's plots pixel-for-pixel.

``` python
model.plot_diagnostics(max_points=2000)
```

## Getting Started

### Requirements

Python 3 with the following packages:

``` bash
pip install numpy pandas matplotlib seaborn scipy statsmodels notebook
```

`statsmodels` is used for LOWESS smoothing in the library's diagnostic plots and for benchmark comparisons in the testing notebook.

Clone or download this repository, open a terminal in the repository folder, and start Jupyter:

``` bash
jupyter notebook
```

Keep `lm_manual_class.py` in the same directory as the notebooks (or make it importable on your Python path).

### Example: Simple linear regression

``` python
import pandas as pd
from lm_manual_class import lm_manual

df = pd.read_csv(
    "https://raw.githubusercontent.com/RamakshaB/Sklearn-Datasets/main/Advertising.csv"
)

model = lm_manual(df["TV"], df["sales"])
model.summary()
model.plot_diagnostics()

# Budget values are in thousands of dollars
print(model.predict([50, 100, 150]))
print(model.predict([50, 100, 150], interval="prediction"))
```

### Example: Multiple linear regression

``` python
import pandas as pd
from lm_manual_class import lm_manual

url = "https://raw.githubusercontent.com/46n/king-county-house-sales-analysis/main/data/housing.csv"
df = pd.read_csv(url, index_col=0)

features = ["sqft_living", "bedrooms", "bathrooms", "grade", "condition"]
model = lm_manual(df[features], df["price"])

model.summary()
print(model.vif())
model.plot_diagnostics(max_points=2000)

# Predict using the same feature names
example_home = pd.DataFrame({
    "sqft_living": [2400],
    "bedrooms": [3],
    "bathrooms": [2.5],
    "grade": [8],
    "condition": [3],
})
print(model.predict(example_home, interval="prediction"))
```

## Project 1: TV Advertising and Sales

**Question:** Is TV advertising expenditure associated with higher product sales, and can it be used to predict sales?

The simple regression notebook examines **200 observations** from the Advertising dataset. It includes exploratory scatterplots and boxplots, Pearson correlation, OLS modeling, regression diagnostics, a manually constructed studentized Breusch–Pagan test, HC3 robust standard errors, and sales forecasts with prediction intervals.

| Result                       |   Value |
|------------------------------|--------:|
| TV–sales Pearson correlation |  0.7822 |
| TV slope estimate            |  0.0475 |
| R²                           |  0.6119 |
| Breusch–Pagan LM statistic   | 48.0380 |

**Business takeaway:** An additional **\$1,000 in TV advertising expenditure** is associated with approximately **47.5 more units sold** on average. The relationship is statistically significant, but the model shows heteroscedasticity and does **not** establish causation.

**Dataset:** [Advertising.csv](https://raw.githubusercontent.com/RamakshaB/Sklearn-Datasets/main/Advertising.csv)

## Project 2: King County Housing Prices

**Question:** How do property characteristics relate to sale prices, and how reliably can they be used for historical property valuations?

The multiple regression notebook analyzes historical **2014–2015 King County, Washington** housing transactions. It models `price` using `sqft_living`, `bedrooms`, `bathrooms`, `grade`, and `condition`, with **21,597 valid observations** in the fitted model.

It includes a scatterplot matrix, correlation heatmap, boxplots, OLS inference, VIF analysis, R-style diagnostics, a manually constructed studentized Breusch–Pagan test, HC3 robust inference, Cook's distance, a sensitivity analysis excluding the 10 highest-Cook's-distance observations, and property price predictions.

| Result                                            |        Value |
|---------------------------------------------------|-------------:|
| R²                                                |       0.5563 |
| Adjusted R²                                       |       0.5562 |
| Residual standard error                           | \$244,571.63 |
| Highest VIF                                       |       3.8394 |
| Breusch–Pagan LM statistic                        |    2284.6415 |
| Observations above Cook's 4/n screening threshold |        1,042 |

**Business takeaway:** Living area, construction grade, and condition have positive conditional associations with historical sale prices. All five selected predictors remain statistically significant with HC3 robust standard errors. However, heteroscedasticity, extreme observations, potential nonlinearity, and omitted location factors limit the model's suitability for precise property valuation.

**Dataset:** [King County housing CSV](https://raw.githubusercontent.com/46n/king-county-house-sales-analysis/main/data/housing.csv)

> **Important:** These are historical valuations using 2014–2015 sales, not estimates of current housing prices.

## Validation Against Statsmodels

`test.ipynb` contains **10 validation exercises** across simple and multiple linear regression. The notebook compares the custom implementation with Statsmodels for:

- Coefficient estimates and model summary statistics
- Handling of missing values
- Point predictions, confidence intervals, and prediction intervals
- Regression diagnostics and VIF-related calculations
- Plotting and influential-observation examples

The notebook documents closely matching results in the tested scenarios. It is an educational validation exercise, **not a claim that the library has passed every possible numerical or statistical edge case**.

## Notes and Limitations

- This is an **educational OLS implementation**, not a replacement for a production statistical modeling package.
- The built-in coefficient intervals and prediction intervals use conventional OLS formulas. They may be unreliable when underlying assumptions fail.
- **Breusch–Pagan tests and HC3 robust standard errors are implemented in the analysis notebooks**, not as built-in `lm_manual` methods.
- HC3 adjusts coefficient standard errors but does not fix nonlinearity, omitted-variable bias, or individual prediction intervals.
- Influential observations should be investigated, not automatically discarded.
- Model estimates describe associations; observational data alone do not demonstrate causal effects.

## License

This project is released under the **MIT License**. See the [`LICENSE`](LICENSE) file for the full license text.
