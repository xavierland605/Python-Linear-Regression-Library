import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, f
from scipy import stats

class multiple_lm_manual:
    def __init__(self, X, y):
        # Check for variable names
        if isinstance(X, pd.DataFrame):
            self.feature_names = [
                name if isinstance(name, str) and name.strip()
                else f"beta_{i+1}"
                for i, name in enumerate(X.columns)
            ]
        elif isinstance(X, pd.Series):
            self.feature_names = [
                X.name if isinstance(X.name, str) and X.name.strip()
                else "beta_1"
            ]
        else:
            self.feature_names = None
        # Dependent variable name
        self.y_name = "y"
        if isinstance(y, pd.Series):
            if isinstance(y.name, str) and y.name.strip():
                self.y_name = y.name
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                name = y.columns[0]
                if isinstance(name, str) and name.strip():
                    self.y_name = name
        # Convert data into NumPy arrays
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.to_numpy(dtype=float, na_value=np.nan)
        else:
            X = np.asarray(X, dtype=float)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy(dtype=float, na_value=np.nan)
        else:
            y = np.asarray(y, dtype=float)
        # Make X two-dimensional
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Make y one-dimensional
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        # Validate dimensions
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid X or y dimensions")
        if len(X) != len(y):
            raise ValueError("X and y must have equal rows")
        if X.shape[1] == 0:
            raise ValueError("X must contain predictors")
        # Remove rows containing missing values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        self.independent = X[mask]
        self.dependent = y[mask]

        # Observations and predictors
        self.n, self.p = self.independent.shape
        self.df = self.n - self.p - 1
        if self.df <= 0:
            raise ValueError("Not enough valid observations")
        # Assign default predictor names
        if self.feature_names is None:
            self.feature_names = [f"beta_{i+1}" for i in range(self.p)]
        # Add column of ones for intercept
        self.X_design = np.column_stack((np.ones(self.n), self.independent))

        # Y mean
        self.mean_y = np.mean(self.dependent)

        # Calculate coefficients
        self.betas, _, rank, _ = np.linalg.lstsq(self.X_design, self.dependent, rcond=None)
        if rank < self.p + 1:
            raise ValueError("Predictors are linearly dependent")
        self.beta_0 = self.betas[0]
        self.coefficients = self.betas[1:]
        # Coefficient names
        self.coefficient_names = ["beta_0"] + self.feature_names

        # Fitted values and residuals
        self.fitted_y = self.X_design @ self.betas
        self.residuals = self.dependent - self.fitted_y

        # Sum of squares
        self.rss = np.sum(self.residuals ** 2)
        self.tss = np.sum((self.dependent - self.mean_y) ** 2)

        # Fit statistics
        self.R_squared = 1 - (self.rss / self.tss)
        self.adjusted_r_squared = 1 - ((1 - self.R_squared) * (self.n - 1) / self.df)
        self.rse = np.sqrt(self.rss / self.df)

        # F-statistic and overall model p-value
        self.f_stat = (((self.tss - self.rss) / self.p) / (self.rss / self.df))
        self.p_value = f.sf(self.f_stat, self.p, self.df)

        # Coefficient standard errors
        self.mse = self.rss / self.df
        self.XTX = self.X_design.T @ self.X_design
        self.se_betas = np.sqrt(np.diag(self.mse * np.linalg.solve(self.XTX,np.eye(self.p + 1))))

        # Coefficient t-statistics
        self.t_betas = self.betas / self.se_betas

        # Coefficient p-values
        self.p_betas = 2 * t.sf(np.abs(self.t_betas), self.df)

        # Coefficient 95% confidence intervals
        t_critical = t.ppf(1 - 0.05 / 2, self.df)
        self.ci_betas = np.column_stack((self.betas - t_critical * self.se_betas, self.betas + t_critical * self.se_betas))

        # Diagnostics

        # Leverage
        Q, R = np.linalg.qr(self.X_design, mode="reduced")
        self.leverage = np.clip(np.sum(Q ** 2, axis=1), 0, 1)

        # Standardized residuals
        if self.rse > 0 and np.all(self.leverage < 1 - 1e-12):
            self.std_residuals = self.residuals / (self.rse * np.sqrt(1 - self.leverage))

            # Scale-location
            self.scale_location = np.sqrt(np.abs(self.std_residuals))

            # Cook's distance
            self.cooks_distance = ((self.std_residuals ** 2 / (self.p + 1))* (self.leverage / (1 - self.leverage)))
        else:
            self.std_residuals = np.full(self.n, np.nan)
            self.scale_location = np.full(self.n, np.nan)
            self.cooks_distance = np.full(self.n, np.nan)
        self.summary()


    def summary(self):
        print("OLS Summary:")
        print("-" * 108)
        print(f"Response variable:   {self.y_name}")
        print(f"Predictors:          {self.p}")
        print(f"Observations used:   {self.n}")
        print("-" * 108)
        print("Coefficient Estimates")
        print(f"{'Term':<18}"f"{'Estimate':>12}"f"{'Std. Error':>14}"f"{'t value':>12}"f"{'P-value':>14}"f"{'95% CI':>30}")
        print("-" * 108)
        for name, beta, se, t_stat, p_val, ci in zip(self.coefficient_names,self.betas,self.se_betas,self.t_betas,self.p_betas,self.ci_betas):
            interval = f"({ci[0]:.4f}, {ci[1]:.4f})"
            print(f"{name:<18}"f"{beta:>12.4f}"f"{se:>14.4f}"f"{t_stat:>12.4f}"f"{p_val:>14.4g}"f"{interval:>30}")
        print("-" * 108)
        print(f"Residual standard error: "f"{self.rse:.4f} on {self.df} degrees of freedom")
        print(f"R-squared: {self.R_squared:.4f}, "f"Adjusted R-squared: {self.adjusted_r_squared:.4f}")
        print(f"F-statistic: {self.f_stat:.4f} "f"on {self.p} and {self.df} DF, "f"p-value: {self.p_value:.4g}")
        print("-" * 108)

