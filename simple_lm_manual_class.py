import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, f
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess


class lm_manual:
    def __init__(self, x, y):
        # Save variable names BEFORE converting inputs to NumPy arrays.
        # Lists, arrays, and unnamed pandas columns use these defaults.
        self.x_name = "beta_1"
        self.y_name = "y"
        if isinstance(x, pd.Series):
            if isinstance(x.name, str) and x.name.strip():
                self.x_name = x.name
        elif isinstance(x, pd.DataFrame):
            if x.shape[1] != 1:
                raise ValueError("x DataFrame must have exactly one column")
            name = x.columns[0]
            if isinstance(name, str) and name.strip():
                self.x_name = name
        if isinstance(y, pd.Series):
            if isinstance(y.name, str) and y.name.strip():
                self.y_name = y.name
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y DataFrame must have exactly one column")
            name = y.columns[0]
            if isinstance(name, str) and name.strip():
                self.y_name = name
        # Convert supported inputs to numeric NumPy arrays.
        # na_value=np.nan also supports pandas nullable values (pd.NA).
        try:
            if isinstance(x, (pd.Series, pd.DataFrame)):
                x = x.to_numpy(dtype=float, na_value=np.nan)
            else:
                x = np.asarray(x, dtype=float)

            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.to_numpy(dtype=float, na_value=np.nan)
            else:
                y = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("x and y must contain numeric data") from exc
        # Allow single-column DataFrames and 2D NumPy arrays.
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.ravel()
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x and y must be 1D arrays of equal length")
        # Remove matching rows with missing or infinite values.
        self.original_n = len(y)
        self.valid_mask = np.isfinite(x) & np.isfinite(y)
        self.independent = x[self.valid_mask]
        self.dependent = y[self.valid_mask]
        self.n = len(self.dependent)
        self.removed_rows = self.original_n - self.n
        if self.n < 3:
            raise ValueError("At least 3 valid observations required")
        if len(np.unique(self.independent)) < 2:
            raise ValueError("X-values must have variation")
        if len(np.unique(self.dependent)) < 2:
            raise ValueError("Y-values must have variation")
        
        # Mean calculations
        sum_x = 0.0
        sum_y = 0.0
        for num_x, num_y in zip(self.independent, self.dependent):
            sum_x += num_x
            sum_y += num_y
        self.mean_x = sum_x / self.n
        self.mean_y = sum_y / self.n

        # Slope calculation
        self.sxx = 0.0
        self.sxy = 0.0
        for num_x, num_y in zip(self.independent, self.dependent):
            self.sxy += (num_x - self.mean_x) * (num_y - self.mean_y)
            self.sxx += (num_x - self.mean_x) ** 2
        self.beta_1 = self.sxy / self.sxx

        # Intercept calculation
        self.beta_0 = self.mean_y - self.beta_1 * self.mean_x

        # Fitted values and residuals
        self.fitted_y = self.beta_0 + self.beta_1 * self.independent
        self.residuals = self.dependent - self.fitted_y

        # Sums of squares
        self.rss = np.sum(self.residuals ** 2)
        self.tss = np.sum((self.dependent - self.mean_y) ** 2)

        # Degrees of freedom and fit statistics
        self.df = self.n - 2
        self.R_Squared = 1.0 - self.rss / self.tss
        self.adjusted_r_squared = 1 - (1 - self.R_Squared) * (self.n - 1) / self.df
        self.rse = np.sqrt(self.rss / self.df)

        # F-statistic and overall model p-value
        self.f_stat = ((self.tss - self.rss) / (self.rss / self.df)if self.rss > 0 else np.inf)
        self.p_value = f.sf(self.f_stat, 1, self.df)

        # Coefficient standard errors
        self.se_beta_1 = self.rse / np.sqrt(self.sxx)
        self.se_beta_0 = self.rse * np.sqrt(1 / self.n + self.mean_x ** 2 / self.sxx)

        # Coefficient t-statistics (undefined when residual error is zero)
        self.t_beta_0 = self.beta_0 / self.se_beta_0 if self.se_beta_0 > 0 else np.nan
        self.t_beta_1 = self.beta_1 / self.se_beta_1 if self.se_beta_1 > 0 else np.nan

        # Coefficient p-values
        self.p_beta_0 = 2 * t.sf(abs(self.t_beta_0), self.df)
        self.p_beta_1 = 2 * t.sf(abs(self.t_beta_1), self.df)

        # 95% coefficient confidence intervals
        t_critical = t.ppf(1 - 0.05 / 2, self.df)
        self.ci_beta_0 = (self.beta_0 - t_critical * self.se_beta_0,self.beta_0 + t_critical * self.se_beta_0,)
        self.ci_beta_1 = (self.beta_1 - t_critical * self.se_beta_1,self.beta_1 + t_critical * self.se_beta_1,)

        # Diagnostics
        self.leverage = 1 / self.n + (self.independent - self.mean_x) ** 2 / self.sxx
        if self.rse > 0 and np.all(self.leverage < 1):
            self.std_residuals = self.residuals / (self.rse * np.sqrt(1 - self.leverage))
            self.scale_location = np.sqrt(np.abs(self.std_residuals))
            self.cooks_distance = ((self.std_residuals ** 2 / 2) * self.leverage / (1 - self.leverage))
        else:
            self.std_residuals = np.full(self.n, np.nan)
            self.scale_location = np.full(self.n, np.nan)
            self.cooks_distance = np.full(self.n, np.nan)

        self.summary()

    def summary(self):
        print("OLS Summary: ")
        print("-" * 72)
        print(f"Response variable:   {self.y_name}")
        print(f"Predictor variable:  {self.x_name}")
        print(f"Observations used:   {self.n}")
        print(f"Rows removed:        {self.removed_rows}")
        print("-" * 72)
        print("Coefficient Estimates")
        print(f"{'Term':<15}{'Estimate':>12}{'Std. Error':>14}{'t value':>12}{'P-value':>14}")
        print("-" * 72)
        print(f"{'Intercept':<15}{self.beta_0:>12.4f}{self.se_beta_0:>14.4f}{self.t_beta_0:>12.4f}{self.p_beta_0:>14.4g}")
        print(f"{self.x_name:<15}{self.beta_1:>12.4f}{self.se_beta_1:>14.4f}{self.t_beta_1:>12.4f}{self.p_beta_1:>14.4g}")
        print("-" * 72)
        print("95% Confidence Intervals")
        print(f"{'Term':<15}{'Lower':>12}{'Upper':>12}")
        print("-" * 72)
        print(f"{'Intercept':<15}{self.ci_beta_0[0]:>12.4f}{self.ci_beta_0[1]:>12.4f}")
        print(f"{self.x_name:<15}{self.ci_beta_1[0]:>12.4f}{self.ci_beta_1[1]:>12.4f}")
        print("-" * 72)
        print(f"Residual standard error: {self.rse:.4f} on {self.df} degrees of freedom")
        print(f"R-squared: {self.R_Squared:.4f}")
        print(f"Adjusted R-squared: {self.adjusted_r_squared:.4f}")
        print(f"F-statistic: {self.f_stat:.4f} on 1 and {self.df} DF")
        print(f"Model p-value: {self.p_value:.4g}")
        print("-" * 72)

    def predict(self, x_value, interval=None, confidence=0.95):
        if interval is not None:
            interval = interval.upper()
        if interval is not None and interval not in ("PREDICTION", "CONFIDENCE"):
            raise ValueError("Interval must be Prediction or Confidence")
        x_value = np.atleast_1d(np.asarray(x_value, dtype=float))
        if x_value.ndim != 1 or not np.isfinite(x_value).all():
            raise ValueError("Prediction inputs must be finite numeric values in 1D")
        predictions = self.beta_0 + self.beta_1 * x_value
        if interval is None:
            return predictions
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
        alpha = 1 - confidence
        t_critical = t.ppf(1 - alpha / 2, self.df)
        lower = np.array([])
        upper = np.array([])
        for x, predicted_y in zip(x_value, predictions):
            if interval == "CONFIDENCE":
                se = self.rse * np.sqrt(1 / self.n + (x - self.mean_x) ** 2 / self.sxx)
            else:
                se = self.rse * np.sqrt(1 + 1 / self.n + (x - self.mean_x) ** 2 / self.sxx)
            margin = t_critical * se
            lower = np.append(lower, predicted_y - margin)
            upper = np.append(upper, predicted_y + margin)
        return pd.DataFrame({
            self.x_name: x_value,
            f"Predicted {self.y_name}": predictions,
            "Lower Bound": lower,
            "Upper Bound": upper,
        }).round(4)

    def plot_regression(self, x_label=None, y_label=None, title=None):
        x_label = self.x_name if x_label is None else x_label
        y_label = self.y_name if y_label is None else y_label
        title = f"Simple Linear Regression: {self.y_name} vs {self.x_name}" if title is None else title

        sorted_indices = np.argsort(self.independent)
        x_sorted = self.independent[sorted_indices]
        y_sorted = self.fitted_y[sorted_indices]
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=self.independent, y=self.dependent, label=f"Actual {self.y_name}")
        sns.lineplot(x=x_sorted, y=y_sorted, label=f"Fitted {self.y_name}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_diagnostics(self):
        if not np.all(np.isfinite(self.std_residuals)):
            raise ValueError("Diagnostics unavailable for this model")
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))

        # Residuals vs Fitted
        sns.scatterplot(x=self.fitted_y, y=self.residuals, ax=ax[0, 0])
        smooth = lowess(self.residuals, self.fitted_y)
        ax[0, 0].plot(smooth[:, 0], smooth[:, 1], color="red")
        ax[0, 0].axhline(0, color="gray", linestyle="--")
        ax[0, 0].set_title("Residuals vs Fitted")
        ax[0, 0].set_xlabel(f"Fitted {self.y_name}")
        ax[0, 0].set_ylabel(f"{self.y_name} Residuals")

        # Normal Q-Q
        stats.probplot(self.std_residuals, dist="norm", plot=ax[0, 1])
        ax[0, 1].set_title("Normal Q-Q")

        # Scale-Location
        sns.scatterplot(x=self.fitted_y, y=self.scale_location, ax=ax[1, 0])
        smooth = lowess(self.scale_location, self.fitted_y)
        ax[1, 0].plot(smooth[:, 0], smooth[:, 1], color="red")
        ax[1, 0].set_title("Scale-Location")
        ax[1, 0].set_xlabel(f"Fitted {self.y_name}")
        ax[1, 0].set_ylabel("√|Standardized Residuals|")

        # Residuals vs Leverage
        sns.scatterplot(x=self.leverage, y=self.std_residuals, ax=ax[1, 1])
        ax[1, 1].axhline(0, color="gray", linestyle="--")
        h = np.linspace(
            max(0.000001, self.leverage.min() * 0.8),
            min(0.999, self.leverage.max() * 1.2),
            200,
        )
        for d in [0.5, 1.0]:
            bound = np.sqrt(d * 2 * (1 - h) / h)
            ax[1, 1].plot(h, bound, "r--", alpha=0.5)
            ax[1, 1].plot(h, -bound, "r--", alpha=0.5, label=f"Cook's D = {d}")
        limit = max(3, np.max(np.abs(self.std_residuals)) * 1.2)
        ax[1, 1].set_ylim(-limit, limit)
        ax[1, 1].set_title("Residuals vs Leverage")
        ax[1, 1].set_xlabel("Leverage")
        ax[1, 1].set_ylabel("Standardized Residuals")
        ax[1, 1].legend()
        plt.tight_layout()
        plt.show()


x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [3, 5, 8, 7, 11, 14, 13, 17]

model = lm_manual(x, y)

