import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t, f
from statsmodels.nonparametric.smoothers_lowess import lowess

class lm_manual:
    def __init__(self, X, y):
        # Record column names before converting pandas objects to NumPy arrays.
        self._named_features = isinstance(X, (pd.DataFrame, pd.Series))
        if isinstance(X, pd.DataFrame):
            self.feature_names = [
                name if isinstance(name, str) and name.strip() else f"beta_{i + 1}"
                for i, name in enumerate(X.columns)
            ]
        elif isinstance(X, pd.Series):
            name = X.name if isinstance(X.name, str) and X.name.strip() else "beta_1"
            self.feature_names = [name]
        else:
            self.feature_names = None
        self.y_name = "y"
        if isinstance(y, pd.Series) and isinstance(y.name, str) and y.name.strip():
            self.y_name = y.name
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y must have exactly one column")
            name = y.columns[0]
            if isinstance(name, str) and name.strip():
                self.y_name = name

        # Convert input data and validate its shape.
        try:
            if isinstance(X, (pd.Series, pd.DataFrame)):
                X = X.to_numpy(dtype=float, na_value=np.nan)
            else:
                X = np.asarray(X, dtype=float)
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.to_numpy(dtype=float, na_value=np.nan)
            else:
                y = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X and y must contain numeric data") from exc
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if X.ndim != 2 or y.ndim != 1 or X.shape[1] == 0:
            raise ValueError("X must be 2D with predictors; y must be 1D")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        self.original_n = len(y)
        self.valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        self.independent = X[self.valid_mask]
        self.dependent = y[self.valid_mask]
        self.removed_rows = self.original_n - len(self.dependent)
        self.n, self.p = self.independent.shape
        self.df = self.n - self.p - 1
        if self.df <= 0:
            raise ValueError("Not enough valid observations for the number of predictors")
        if self.feature_names is None:
            self.feature_names = [f"beta_{i + 1}" for i in range(self.p)]
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Predictor names must be unique")

        # Add the intercept column and estimate coefficients using least squares.
        self.X_design = np.column_stack((np.ones(self.n), self.independent))
        self.betas, _, rank, _ = np.linalg.lstsq(
            self.X_design, self.dependent, rcond=None
        )
        if rank != self.p + 1:
            raise ValueError("Predictors are linearly dependent (or constant)")
        self.beta_0 = self.betas[0]
        self.coefficients = self.betas[1:]
        self.coefficient_names = ["Intercept"] + self.feature_names
        self.mean_y = np.mean(self.dependent)
        self.fitted_y = self.X_design @ self.betas
        self.residuals = self.dependent - self.fitted_y

        # Model fit statistics.
        self.rss = np.sum(self.residuals ** 2)
        self.tss = np.sum((self.dependent - self.mean_y) ** 2)
        if self.tss <= 0:
            raise ValueError("The response variable must have variation")
        self.R_squared = 1 - self.rss / self.tss
        self.R_Squared = self.R_squared  # Original simple-class spelling.
        self.adjusted_r_squared = 1 - (1 - self.R_squared) * (self.n - 1) / self.df
        self.mse = self.rss / self.df
        self.rse = np.sqrt(self.mse)
        self.f_stat = max(self.tss - self.rss, 0) / self.p / self.mse if self.mse > 0 else np.inf
        self.p_value = f.sf(self.f_stat, self.p, self.df)

        # QR gives coefficient standard errors and leverage without explicitly
        # forming the inverse of X'X (which can be numerically unstable).
        Q, R = np.linalg.qr(self.X_design, mode="reduced")
        R_inverse = np.linalg.solve(R, np.eye(self.p + 1))
        self.XTX_inverse = R_inverse @ R_inverse.T
        self.XTX = self.X_design.T @ self.X_design
        self.se_betas = np.sqrt(np.maximum(np.diag(self.mse * self.XTX_inverse), 0))
        self.t_betas = np.full(self.p + 1, np.nan)
        np.divide(self.betas, self.se_betas, out=self.t_betas, where=self.se_betas > 0)
        zero_se = self.se_betas == 0
        self.t_betas[zero_se & (self.betas != 0)] = np.sign(
            self.betas[zero_se & (self.betas != 0)]
        ) * np.inf
        self.p_betas = 2 * t.sf(np.abs(self.t_betas), self.df)
        t_critical = t.ppf(0.975, self.df)
        self.ci_betas = np.column_stack((self.betas - t_critical * self.se_betas, self.betas + t_critical * self.se_betas,))

        # Diagnostics: leverage, standardized residuals and Cook's distance.
        self.leverage = np.clip(np.sum(Q ** 2, axis=1), 0, 1)
        self.std_residuals = np.full(self.n, np.nan)
        valid_diag = (self.rse > 0) & (self.leverage < 1 - 1e-12)
        self.std_residuals[valid_diag] = (
            self.residuals[valid_diag] /
            (self.rse * np.sqrt(1 - self.leverage[valid_diag]))
        )
        self.scale_location = np.sqrt(np.abs(self.std_residuals))
        self.cooks_distance = np.full(self.n, np.nan)
        self.cooks_distance[valid_diag] = (
            self.std_residuals[valid_diag] ** 2 / (self.p + 1)
            * self.leverage[valid_diag] / (1 - self.leverage[valid_diag])
        )

        # Keep frequently used original simple-regression attributes.
        self.se_beta_0 = self.se_betas[0]
        self.t_beta_0 = self.t_betas[0]
        self.p_beta_0 = self.p_betas[0]
        self.ci_beta_0 = tuple(self.ci_betas[0])
        if self.p == 1:
            x = self.independent[:, 0]
            self.x_name = self.feature_names[0]
            self.beta_1 = self.betas[1]
            self.mean_x = np.mean(x)
            self.sxx = np.sum((x - self.mean_x) ** 2)
            self.sxy = np.sum((x - self.mean_x) * (self.dependent - self.mean_y))
            self.se_beta_1 = self.se_betas[1]
            self.t_beta_1 = self.t_betas[1]
            self.p_beta_1 = self.p_betas[1]
            self.ci_beta_1 = tuple(self.ci_betas[1])

        self.summary()

    def summary(self):
        """Print an OLS summary for either simple or multiple regression."""
        print("OLS Summary:")
        print("-" * 108)
        print(f"Response variable:   {self.y_name}")
        print(f"Predictors:          {self.p}")
        print(f"Observations used:   {self.n}")
        print(f"Rows removed:        {self.removed_rows}")
        print("-" * 108)
        print("Coefficient Estimates")
        print(f"{'Term':<18}{'Estimate':>12}{'Std. Error':>14}"f"{'t value':>12}{'P-value':>14}{'95% CI':>30}")
        print("-" * 108)
        for name, beta, se, t_stat, p_val, ci in zip(self.coefficient_names, self.betas, self.se_betas, self.t_betas, self.p_betas, self.ci_betas):
            interval = f"({ci[0]:.4f}, {ci[1]:.4f})"
            print(f"{name:<18}{beta:>12.4f}{se:>14.4f}"f"{t_stat:>12.4f}{p_val:>14.4g}{interval:>30}")
        print("-" * 108)
        print(f"Residual standard error: {self.rse:.4f} on {self.df} degrees of freedom")
        print(f"R-squared: {self.R_squared:.4f}, "f"Adjusted R-squared: {self.adjusted_r_squared:.4f}")
        print(f"F-statistic: {self.f_stat:.4f} on {self.p} and {self.df} DF, "f"p-value: {self.p_value:.4g}")
        print("-" * 108)

    def predict(self, X_value, interval=None, confidence=0.95):
        """Predict using one or more rows; optionally add confidence/prediction bounds."""
        if interval is not None:
            if not isinstance(interval, str):
                raise ValueError("Interval must be 'prediction' or 'confidence'")
            interval = interval.upper()
            if interval not in ("PREDICTION", "CONFIDENCE"):
                raise ValueError("Interval must be 'prediction' or 'confidence'")
        if isinstance(X_value, pd.DataFrame):
            if self._named_features:
                if (len(X_value.columns) != self.p
                        or set(X_value.columns) != set(self.feature_names)):
                    raise ValueError("Prediction columns must match training predictors")
                X_value = X_value[self.feature_names]
            try:
                X_value = X_value.to_numpy(dtype=float, na_value=np.nan)
            except (TypeError, ValueError) as exc:
                raise ValueError("Prediction inputs must be numeric") from exc
        elif isinstance(X_value, pd.Series):
            X_value = X_value.to_numpy(dtype=float, na_value=np.nan)
        else:
            try:
                X_value = np.asarray(X_value, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Prediction inputs must be numeric") from exc
        if X_value.ndim == 0:
            X_value = X_value.reshape(1, 1)
        elif X_value.ndim == 1:
            if self.p == 1:
                X_value = X_value.reshape(-1, 1)  # Several single-x predictions.
            else:
                X_value = X_value.reshape(1, -1)  # One multi-x prediction.
        if X_value.ndim != 2 or X_value.shape[1] != self.p:
            raise ValueError(f"Predictions require exactly {self.p} predictor(s)")
        if not np.isfinite(X_value).all():
            raise ValueError("Prediction inputs must contain only finite numbers")
        X_new = np.column_stack((np.ones(len(X_value)), X_value))
        predictions = X_new @ self.betas
        if interval is None:
            return predictions
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
        t_critical = t.ppf(1 - (1 - confidence) / 2, self.df)
        # The quadratic form x'(X'X)^(-1)x gives uncertainty in the fitted mean.
        mean_variance = np.sum((X_new @ self.XTX_inverse) * X_new, axis=1)
        mean_variance = np.maximum(mean_variance, 0)
        if interval == "CONFIDENCE":
            se = self.rse * np.sqrt(mean_variance)
        else:
            se = self.rse * np.sqrt(1 + mean_variance)
        margin = t_critical * se
        result = pd.DataFrame(X_value, columns=self.feature_names)
        result[f"Predicted {self.y_name}"] = predictions
        result["Lower Bound"] = predictions - margin
        result["Upper Bound"] = predictions + margin
        return result.round(4)

    def plot_regression(self, x_label=None, y_label=None, title=None, max_points=3000):
        """Show a fitted regression line (available only for one predictor)."""
        if self.p != 1:
            raise ValueError("A 2D regression line requires one predictor; "
                             "use plot_diagnostics() for multiple regression")
        if max_points < 1:
            raise ValueError("max_points must be positive")
        rng = np.random.default_rng(42)
        indexes = (rng.choice(self.n, size=max_points, replace=False) if self.n > max_points else np.arange(self.n))
        x = self.independent[indexes, 0]
        y = self.dependent[indexes]
        x_line = np.linspace(self.independent[:, 0].min(),self.independent[:, 0].max(), 200)
        y_line = self.beta_0 + self.beta_1 * x_line
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=x, y=y, alpha=0.5, s=20, label=f"Actual {self.y_name}")
        plt.plot(x_line, y_line, color="red", label=f"Fitted {self.y_name}")
        plt.xlabel(self.x_name if x_label is None else x_label)
        plt.ylabel(self.y_name if y_label is None else y_label)
        plt.title(title if title is not None else f"Simple Linear Regression: {self.y_name} vs {self.x_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_diagnostics(self, max_points=2000, random_state=42):
        """Show the same four diagnostics for simple or multiple regression.

        Large datasets are sampled only for plotting/LOWESS; the model is
        always fitted and its diagnostics computed on every valid observation.
        """
        if max_points < 2:
            raise ValueError("max_points must be at least 2")
        valid = np.flatnonzero(np.isfinite(self.std_residuals))
        if len(valid) < 2:
            raise ValueError("Diagnostics unavailable for this model")
        if len(valid) > max_points:
            rng = np.random.default_rng(random_state)
            valid = rng.choice(valid, size=max_points, replace=False)

        fitted = self.fitted_y[valid]
        residuals = self.residuals[valid]
        standardized = self.std_residuals[valid]
        scale = self.scale_location[valid]
        leverage = self.leverage[valid]
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))

        # Residuals vs fitted values.
        sns.scatterplot(x=fitted, y=residuals, ax=ax[0, 0], s=18, alpha=0.6)
        smooth = lowess(residuals, fitted, return_sorted=True)
        ax[0, 0].plot(smooth[:, 0], smooth[:, 1], color="red")
        ax[0, 0].axhline(0, color="gray", linestyle="--")
        ax[0, 0].set_title("Residuals vs Fitted")
        ax[0, 0].set_xlabel(f"Fitted {self.y_name}")
        ax[0, 0].set_ylabel(f"{self.y_name} Residuals")

        # Normal Q-Q plot.
        stats.probplot(standardized, dist="norm", plot=ax[0, 1])
        ax[0, 1].set_title("Normal Q-Q")

        # Scale-location.
        sns.scatterplot(x=fitted, y=scale, ax=ax[1, 0], s=18, alpha=0.6)
        smooth = lowess(scale, fitted, return_sorted=True)
        ax[1, 0].plot(smooth[:, 0], smooth[:, 1], color="red")
        ax[1, 0].set_title("Scale-Location")
        ax[1, 0].set_xlabel(f"Fitted {self.y_name}")
        ax[1, 0].set_ylabel("√|Standardized Residuals|")

        # Standardized residuals vs leverage with Cook's distance contours.
        sns.scatterplot(x=leverage, y=standardized, ax=ax[1, 1], s=18, alpha=0.6)
        ax[1, 1].axhline(0, color="gray", linestyle="--")
        high = min(0.999, max(float(leverage.max()) * 1.2, 0.00001))
        low = max(0.000001, high / 1000)
        h = np.linspace(low, high, 200)
        for distance in [0.5, 1.0]:
            bound = np.sqrt(distance * (self.p + 1) * (1 - h) / h)
            ax[1, 1].plot(h, bound, "r--", alpha=0.5)
            ax[1, 1].plot(h, -bound, "r--", alpha=0.5, label=f"Cook's D = {distance}")
        limit = max(3, float(np.max(np.abs(standardized))) * 1.2)
        ax[1, 1].set_ylim(-limit, limit)
        ax[1, 1].set_title("Residuals vs Leverage")
        ax[1, 1].set_xlabel("Leverage")
        ax[1, 1].set_ylabel("Standardized Residuals")
        ax[1, 1].legend()
        plt.tight_layout()
        plt.show()
        
    def vif(self):
        if self.p == 1:
            values = np.array([1.0])
        else:
            correlations = np.corrcoef(self.independent, rowvar=False) 
            values = np.diag(np.linalg.solve(correlations, np.eye(self.p)))
        values = np.maximum(values, 1.0)
        return pd.DataFrame({"Predictor": self.feature_names,"VIF": values}).round(4)