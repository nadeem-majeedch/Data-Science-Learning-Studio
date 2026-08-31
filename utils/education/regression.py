"""
Regression — learning topics for the Regression module.
"""
from utils.education.base import T

TOPICS = {
    # ── 1 ──────────────────────────────────────────────────────────
    "what_is_regression": T(
        title="What Is Regression",
        module="regression",
        what=(
            "Regression is a supervised learning task that predicts "
            "continuous numerical values. Given input features, the "
            "model outputs a number: price, temperature, salary."
        ),
        why=(
            "Regression problems are everywhere: predicting house "
            "prices, stock returns, demand forecasting, and medical "
            "dosage calculation."
        ),
        when=(
            "Use regression when your target is continuous. If the "
            "target is categorical (e.g., cheap/medium/expensive), "
            "use classification instead."
        ),
        example=(
            "California Housing dataset: predict median house value "
            "from features like income, house age, average rooms. "
            "Target is continuous ($15K - $500K)."
        ),
        mistakes=[
            "Using classification for continuous targets.",
            "Evaluating regression models with classification metrics.",
            "Ignoring residual analysis after training.",
        ],
        interpretation=(
            "A good regression model produces predictions close to "
            "actual values. R² measures proportion of variance explained; "
            "RMSE measures average prediction error in original units."
        ),
        think_about_it=(
            "You predict house prices. Your model gives R²=0.95 and "
            "RMSE=$15,000. Is this good? How would you decide?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.metrics import r2_score, mean_squared_error\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)\n"
            "y_pred = model.predict(X_test)\n"
            "print(f'R²: {r2_score(y_test, y_pred):.4f}')\n"
            "```"
        ),
        keywords=["regression", "continuous", "prediction", "numerical", "supervised"],
    ),

    # ── 2 ──────────────────────────────────────────────────────────
    "regression_vs_classification": T(
        title="Regression vs Classification",
        module="regression",
        what=(
            "Regression predicts numbers, classification predicts "
            "categories. The target variable determines the task type."
        ),
        why=(
            "Choosing the wrong task leads to meaningless results. "
            "Understanding the distinction is fundamental to ML."
        ),
        when=(
            "Check your target variable. Continuous values (price, "
            "temperature) → regression. Discrete labels (yes/no, "
            "category) → classification."
        ),
        example=(
            "Predicting house price: regression ($250,000).\n"
            "Predicting price category: classification (cheap/medium/expensive).\n"
            "Same data, different tasks."
        ),
        mistakes=[
            "Using regression when categories are what matter.",
            "Using classification when exact values matter.",
            "Forgetting that some algorithms only do one task.",
        ],
        interpretation=(
            "The target determines the task. Some algorithms support "
            "both (e.g., Decision Tree, Random Forest) but use "
            "different classes in sklearn."
        ),
        think_about_it=(
            "Predicting student exam scores (0-100). Is this "
            "regression or classification? What if you only care "
            "about pass/fail?"
        ),
        code_link=(
            "```python\n"
            "# Regression\n"
            "from sklearn.tree import DecisionTreeRegressor\n"
            "model = DecisionTreeRegressor()\n"
            "\n"
            "# Classification\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "model = DecisionTreeClassifier()\n"
            "```"
        ),
        keywords=["regression", "classification", "target", "continuous", "discrete"],
    ),

    # ── 3 ──────────────────────────────────────────────────────────
    "simple_linear_regression": T(
        title="Simple Linear Regression",
        module="regression",
        what=(
            "Simple linear regression fits a straight line to one "
            "feature: y = b0 + b1*x. b0 is the intercept, b1 is the "
            "slope (coefficient)."
        ),
        why=(
            "It's the simplest regression model and the foundation "
            "for understanding more complex models. The coefficients "
            "directly show the relationship between feature and target."
        ),
        when=(
            "Use as a baseline. Works when the relationship between "
            "one feature and the target is approximately linear."
        ),
        example=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "\n"
            "model = LinearRegression()\n"
            "model.fit(X_train[['Fare']], y_train)\n"
            "\n"
            "print(f'Intercept: {model.intercept_:.4f}')\n"
            "print(f'Coefficient: {model.coef_[0]:.4f}')\n"
            "# For every $1 increase in Fare, prediction changes by $b1\n"
            "```"
        ),
        mistakes=[
            "Assuming linear regression implies a linear relationship.",
            "Not checking residuals for patterns.",
            "Extrapolating beyond the range of training data.",
        ],
        interpretation=(
            "Coefficient b1: for each 1-unit increase in x, y "
            "changes by b1. Positive b1 = positive relationship. "
            "The intercept b0 is the predicted y when x=0."
        ),
        think_about_it=(
            "Linear regression on house data: coef for 'rooms'=50K. "
            "Does this mean adding one room increases value by $50K? "
            "Why might this be misleading?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "import numpy as np\n"
            "\n"
            "model = LinearRegression()\n"
            "model.fit(X_train[['rooms']], y_train)\n"
            "print(f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * rooms')\n"
            "```"
        ),
        keywords=["linear", "simple", "intercept", "coefficient", "slope"],
    ),

    # ── 4 ──────────────────────────────────────────────────────────
    "multiple_linear_regression": T(
        title="Multiple Linear Regression",
        module="regression",
        what=(
            "Multiple linear regression extends simple LR to multiple "
            "features: y = b0 + b1*x1 + b2*x2 + ... + bn*xn. "
            "Each coefficient measures the effect of that feature "
            "while holding others constant."
        ),
        why=(
            "Real problems have multiple features. Multiple regression "
            "captures the combined effect of many predictors."
        ),
        when=(
            "When you have multiple features and expect roughly linear "
            "relationships. Check for multicollinearity (correlated features)."
        ),
        example=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "for name, coef in zip(feature_names, model.coef_):\n"
            "    print(f'{name}: {coef:.4f}')\n"
            "print(f'Intercept: {model.intercept_:.4f}')\n"
            "```"
        ),
        mistakes=[
            "Including highly correlated features (multicollinearity).",
            "Not checking residual plots for patterns.",
            "Assuming coefficients mean causation.",
        ],
        interpretation=(
            "Each coefficient is the change in y per 1-unit change "
            "in that feature, holding all others constant. Large "
            "coefficients don't necessarily mean important features "
            "(scale matters)."
        ),
        think_about_it=(
            "Feature A has coef=100 and Feature B has coef=0.5. "
            "Feature A is measured in thousands, Feature B in single "
            "units. Which is actually more important?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)\n"
            "print(f'R² train: {model.score(X_train, y_train):.4f}')\n"
            "print(f'R² test:  {model.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        keywords=["multiple", "linear", "coefficients", "features", "multivariate"],
    ),

    # ── 5 ──────────────────────────────────────────────────────────
    "least_squares": T(
        title="Least Squares Method",
        module="regression",
        what=(
            "Least squares finds the line that minimises the sum of "
            "squared residuals (actual - predicted)². It's the "
            "mathematical foundation of linear regression."
        ),
        why=(
            "Understanding least squares helps you understand what "
            "the model optimises and why outliers have such a large "
            "effect on linear regression."
        ),
        when=(
            "Linear Regression uses ordinary least squares (OLS) by "
            "default. Regularised versions (Ridge, Lasso) add penalty "
            "terms to prevent overfitting."
        ),
        example=(
            "```python\n"
            "import numpy as np\n"
            "from sklearn.linear_model import LinearRegression\n"
            "\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Residual Sum of Squares\n"
            "y_pred = model.predict(X_train)\n"
            "rss = np.sum((y_train - y_pred) ** 2)\n"
            "print(f'RSS: {rss:.4f}')\n"
            "```"
        ),
        mistakes=[
            "Not understanding that outliers disproportionately affect least squares.",
            "Confusing RSS (sum) with MSE (mean).",
            "Forgetting that least squares assumes normally distributed residuals.",
        ],
        interpretation=(
            "RSS = sum of squared errors. Lower is better. OLS finds "
            "the unique line that minimises RSS. Adding a feature "
            "always decreases RSS (or keeps it same), but R² may not "
            "improve."
        ),
        think_about_it=(
            "You add a random noise feature to the model. Does RSS "
            "increase, decrease, or stay the same? What about R²?"
        ),
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "\n"
            "# Manual OLS (for understanding)\n"
            "x = X_train.values.flatten()\n"
            "y = y_train.values\n"
            "b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)\n"
            "b0 = y.mean() - b1 * x.mean()\n"
            "print(f'y = {b0:.4f} + {b1:.4f} * x')\n"
            "```"
        ),
        keywords=["least squares", "ols", "residual", "sum", "squared", "minimise"],
    ),

    # ── 6 ──────────────────────────────────────────────────────────
    "regression_coefficients": T(
        title="Regression Coefficients",
        module="regression",
        what=(
            "Coefficients (coef_) show how each feature affects the "
            "prediction. A positive coefficient means the feature "
            "increases the target; negative means it decreases it."
        ),
        why=(
            "Coefficients provide interpretability — you can explain "
            "why the model makes specific predictions. This is crucial "
            "for business applications."
        ),
        when=(
            "After training a linear model. Use coefficients for feature "
            "importance and to explain model decisions."
        ),
        example=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "coefs = pd.Series(model.coef_, index=feature_names)\n"
            "print(coefs.sort_values(ascending=False))\n"
            "# income        45000.2  → strongest positive\n"
            "# age           -500.3   → negative (older = cheaper?)\n"
            "```"
        ),
        mistakes=[
            "Comparing raw coefficients across features with different scales.",
            "Interpreting correlation as causation.",
            "Ignoring the intercept in interpretation.",
        ],
        interpretation=(
            "Standardised coefficients allow fair comparison across "
            "features. Without standardisation, a feature with range "
            "[0, 1000] will have a smaller coefficient than one with "
            "range [0, 1], even if it's more important."
        ),
        think_about_it=(
            "Feature A (income, range 0-200K) has coef=0.5. "
            "Feature B (rooms, range 1-10) has coef=50000. "
            "Which feature has a larger effect on prediction?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "# Standardised coefficients\n"
            "scaler = StandardScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train)\n"
            "model.fit(X_train_scaled, y_train)\n"
            "std_coefs = pd.Series(model.coef_, index=feature_names)\n"
            "print(std_coefs.sort_values(ascending=False))\n"
            "```"
        ),
        keywords=["coefficient", "coef", "importance", "linear", "interpret"],
    ),

    # ── 7 ──────────────────────────────────────────────────────────
    "residuals": T(
        title="Understanding Residuals",
        module="regression",
        what=(
            "A residual is the difference between actual and predicted "
            "value: residual = y_actual - y_predicted. Residual analysis "
            "is the primary diagnostic tool for regression models."
        ),
        why=(
            "Residuals reveal whether the model's assumptions are met: "
            "linearity, independence, homoscedasticity, and normality. "
            "Patterns in residuals indicate model problems."
        ),
        when=(
            "After training any regression model. Always plot residuals "
            "vs predicted values."
        ),
        example=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "residuals = y_test - y_pred\n"
            "\n"
            "plt.scatter(y_pred, residuals, alpha=0.5)\n"
            "plt.axhline(y=0, color='r', linestyle='--')\n"
            "plt.xlabel('Predicted')\n"
            "plt.ylabel('Residual')\n"
            "plt.title('Residual Plot')\n"
            "```"
        ),
        mistakes=[
            "Not plotting residuals — you're flying blind.",
            "Ignoring patterns in residuals (funnel shape, curve).",
            "Assuming residuals are normal without checking.",
        ],
        interpretation=(
            "Good residual plot: random scatter around 0, no pattern. "
            "Funnel shape → heteroscedasticity. Curve → non-linearity. "
            "Outliers → influential points."
        ),
        think_about_it=(
            "Your residual plot shows a clear U-shape. What does "
            "this tell you about your linear model?"
        ),
        code_link=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "\n"
            "residuals = y_test - model.predict(X_test)\n"
            "\n"
            "# Residual plot\n"
            "plt.scatter(model.predict(X_test), residuals, alpha=0.5)\n"
            "plt.axhline(y=0, color='r', linestyle='--')\n"
            "plt.xlabel('Predicted Values')\n"
            "plt.ylabel('Residuals')\n"
            "\n"
            "# Q-Q plot for normality\n"
            "from scipy import stats\n"
            "stats.probplot(residuals, dist='norm', plot=plt)\n"
            "```"
        ),
        keywords=["residual", "error", "residual plot", "diagnostic", "assumptions"],
    ),

    # ── 8 ──────────────────────────────────────────────────────────
    "regression_intercept": T(
        title="Regression Intercept",
        module="regression",
        what=(
            "The intercept (b0) is the predicted target value when "
            "all features are zero. It shifts the regression line "
            "up or down."
        ),
        why=(
            "The intercept anchors the regression line. Without it, "
            "the line would be forced through the origin, which is "
            "rarely appropriate."
        ),
        when=(
            "The intercept is automatically included in LinearRegression. "
            "You can remove it with fit_intercept=False, but this is "
            "rarely advisable."
        ),
        example=(
            "```python\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)\n"
            "print(f'Intercept: {model.intercept_:.2f}')\n"
            "# Intercept = 45000 means: when all features are 0,\n"
            "# predicted price is $45,000\n"
            "```"
        ),
        mistakes=[
            "Interpreting the intercept literally when features=0 is impossible.",
            "Removing intercept without understanding the implications.",
            "Forgetting that intercept changes with feature scaling.",
        ],
        interpretation=(
            "The intercept may not have practical meaning if features=0 "
            "is impossible (e.g., 0 rooms). Its main role is to position "
            "the regression line correctly."
        ),
        think_about_it=(
            "A model predicting house prices has intercept=-50000. "
            "Does this mean a house with all features=0 is worth -$50K? "
            "What does this tell you?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "\n"
            "model = LinearRegression(fit_intercept=True)  # default\n"
            "model.fit(X_train, y_train)\n"
            "print(f'b0 (intercept): {model.intercept_:.4f}')\n"
            "print(f'b1 (coef): {model.coef_[0]:.4f}')\n"
            "```"
        ),
        keywords=["intercept", "b0", "bias", "offset", "constant"],
    ),

    # ── 9 ──────────────────────────────────────────────────────────
    "regression_predictions": T(
        title="Making Regression Predictions",
        module="regression",
        what=(
            "After training, model.predict(X) outputs continuous "
            "predicted values. These are point estimates — the model's "
            "best guess for each input."
        ),
        why=(
            "Predictions are the model's output. Understanding their "
            "quality requires comparing to actual values using "
            "regression metrics."
        ),
        when=(
            "After training. Use model.predict(X_test) to evaluate "
            "and compare predictions to actual values."
        ),
        example=(
            "```python\n"
            "y_pred = model.predict(X_test)\n"
            "comparison = pd.DataFrame({\n"
            "    'Actual': y_test.values,\n"
            "    'Predicted': y_pred,\n"
            "    'Error': y_test.values - y_pred\n"
            "})\n"
            "print(comparison.head(10))\n"
            "```"
        ),
        mistakes=[
            "Not comparing predictions to actuals — predictions without context are meaningless.",
            "Ignoring large errors — check which samples have high residuals.",
            "Reporting only aggregate metrics without examining individual predictions.",
        ],
        interpretation=(
            "Close predictions (small residuals) are good. Check for "
            "patterns: are large errors concentrated in certain ranges? "
            "This reveals where the model struggles."
        ),
        think_about_it=(
            "Your model predicts house prices between $150K-$400K "
            "but some actual values are $500K+. What does this tell "
            "you about the model's limitations?"
        ),
        code_link=(
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "\n"
            "# Summary\n"
            "print(f'Pred range: {y_pred.min():.0f} - {y_pred.max():.0f}')\n"
            "print(f'Actual range: {y_test.min():.0f} - {y_test.max():.0f}')\n"
            "\n"
            "# Worst predictions\n"
            "errors = np.abs(y_test - y_pred)\n"
            "worst_idx = errors.argsort()[-5:]\n"
            "print(f'Worst errors: {errors[worst_idx].values}')"
        ),
        keywords=["predict", "prediction", "output", "actual", "compare"],
    ),

    # ── 10 ─────────────────────────────────────────────────────────
    "mae": T(
        title="Mean Absolute Error (MAE)",
        module="regression",
        what=(
            "MAE is the average of absolute differences between "
            "predicted and actual values: MAE = mean(|y - ŷ|). "
            "It's in the same units as the target."
        ),
        why=(
            "MAE is the most interpretable regression metric. If MAE "
            "= 5000, your model is off by $5,000 on average. It's "
            "robust to outliers (unlike MSE)."
        ),
        when=(
            "Use when you want interpretable error in original units "
            "and when outliers shouldn't dominate the metric."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import mean_absolute_error\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "mae = mean_absolute_error(y_test, y_pred)\n"
            "print(f'MAE: ${mae:,.0f}')  # Average error: $25,000\n"
            "```"
        ),
        mistakes=[
            "Comparing MAE across datasets with different target scales.",
            "Ignoring that MAE treats all errors equally (unlike MSE).",
            "Not visualising the error distribution.",
        ],
        interpretation=(
            "MAE = $25,000 means the model's predictions are off by "
            "$25,000 on average. Lower is better. MAE is more robust "
            "to outliers than MSE."
        ),
        think_about_it=(
            "Model A: MAE=$30K, RMSE=$45K. Model B: MAE=$35K, RMSE=$40K. "
            "Which model has more outlier predictions?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import mean_absolute_error\n"
            "\n"
            "mae = mean_absolute_error(y_test, y_pred)\n"
            "print(f'MAE: {mae:.4f}')\n"
            "\n"
            "# Per-sample absolute errors\n"
            "abs_errors = np.abs(y_test - y_pred)\n"
            "print(f'Median absolute error: {np.median(abs_errors):.4f}')\n"
            "```"
        ),
        keywords=["mae", "mean absolute", "error", "absolute", "interpret"],
    ),

    # ── 11 ─────────────────────────────────────────────────────────
    "mse": T(
        title="Mean Squared Error (MSE)",
        module="regression",
        what=(
            "MSE is the average of squared differences between "
            "predicted and actual values: MSE = mean((y - ŷ)²). "
            "It penalises large errors more heavily."
        ),
        why=(
            "MSE is the mathematical foundation for many regression "
            "algorithms. Squaring makes large errors cost more, which "
            "is often desirable (a $100K error is more than twice as "
            "bad as a $50K error)."
        ),
        when=(
            "Use when large errors are disproportionately costly. "
            "MSE is the default loss function for Linear Regression."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import mean_squared_error\n"
            "\n"
            "mse = mean_squared_error(y_test, y_pred)\n"
            "print(f'MSE: {mse:,.0f}')  # Units are squared!\n"
            "print(f'RMSE: {np.sqrt(mse):,.0f}')  # Back to original units\n"
            "```"
        ),
        mistakes=[
            "Interpreting MSE directly — its units are squared.",
            "Comparing MSE across datasets with different scales.",
            "Not using RMSE for interpretability.",
        ],
        interpretation=(
            "MSE=625,000,000 is hard to interpret. RMSE=$25,000 is "
            "the same information in original units. MSE > MAE² "
            "when there are outliers."
        ),
        think_about_it=(
            "Two models have MAE=$20K. Model A has MSE=$500B and "
            "Model B has MSE=$800B. What can you conclude about "
            "their error distributions?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import mean_squared_error\n"
            "import numpy as np\n"
            "\n"
            "mse = mean_squared_error(y_test, y_pred)\n"
            "rmse = np.sqrt(mse)\n"
            "print(f'MSE:  {mse:,.2f}')\n"
            "print(f'RMSE: {rmse:,.2f}')\n"
            "```"
        ),
        keywords=["mse", "mean squared", "squared", "penalty", "loss"],
    ),

    # ── 12 ─────────────────────────────────────────────────────────
    "rmse": T(
        title="Root Mean Squared Error (RMSE)",
        module="regression",
        what=(
            "RMSE is the square root of MSE: RMSE = √MSE. It returns "
            "error to the original units of the target, making it "
            "interpretable like MAE but sensitive to outliers like MSE."
        ),
        why=(
            "RMSE is the most commonly reported regression metric. "
            "It's in original units (like MAE) but penalises large "
            "errors (like MSE)."
        ),
        when=(
            "Use as the primary regression metric. Report RMSE alongside "
            "MAE to understand error distribution."
        ),
        example=(
            "```python\n"
            "import numpy as np\n"
            "from sklearn.metrics import mean_squared_error\n"
            "\n"
            "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n"
            "print(f'RMSE: ${rmse:,.0f}')  # $25,000\n"
            "# This means predictions are typically off by $25,000\n"
            "```"
        ),
        mistakes=[
            "Reporting RMSE without context (is $25K good for $200K houses?).",
            "Not comparing with MAE to understand error distribution.",
            "Ignoring RMSE when training — monitor it to detect overfitting.",
        ],
        interpretation=(
            "RMSE ≈ MAE → errors are evenly distributed. "
            "RMSE >> MAE → some large outlier errors exist. "
            "RMSE is always ≥ MAE."
        ),
        think_about_it=(
            "If RMSE is 3x larger than MAE, what does this tell you "
            "about the distribution of prediction errors?"
        ),
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "from sklearn.metrics import mean_squared_error, mean_absolute_error\n"
            "\n"
            "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n"
            "mae = mean_absolute_error(y_test, y_pred)\n"
            "print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}')\n"
            "print(f'RMSE/MAE ratio: {rmse/mae:.2f}')\n"
            "# ratio > 1.5 suggests outlier errors\n"
            "```"
        ),
        keywords=["rmse", "root", "squared", "error", "interpretable", "units"],
    ),

    # ── 13 ─────────────────────────────────────────────────────────
    "r_squared": T(
        title="R-Squared (R²)",
        module="regression",
        what=(
            "R² measures the proportion of variance in the target "
            "that the model explains: R² = 1 - (SS_res / SS_tot). "
            "R²=1.0 means perfect prediction; R²=0 means the model "
            "is no better than predicting the mean."
        ),
        why=(
            "R² gives a scale-free measure of model quality. R²=0.85 "
            "means the model explains 85% of the variance in house "
            "prices — regardless of the price range."
        ),
        when=(
            "Use as the primary metric for comparing models. R² "
            "allows comparison across datasets and scales."
        ),
        example=(
            "```python\n"
            "from sklearn.metrics import r2_score\n"
            "\n"
            "r2 = r2_score(y_test, y_pred)\n"
            "print(f'R²: {r2:.4f}')  # 0.85 = explains 85% of variance\n"
            "\n"
            "# Adjusted R² (penalises extra features)\n"
            "n, p = X_test.shape\n"
            "adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)\n"
            "print(f'Adjusted R²: {adj_r2:.4f}')\n"
            "```"
        ),
        mistakes=[
            "R² can be negative (worse than predicting the mean).",
            "Not using adjusted R² when comparing models with different feature counts.",
            "Assuming R²=0.9 is 'good' without domain context.",
        ],
        interpretation=(
            "R²=0.85: model explains 85% of price variation. "
            "R²=0: model is no better than the mean. "
            "R²<0: model is worse than the mean. "
            "Adding features always increases R² but may not improve "
            "adjusted R²."
        ),
        think_about_it=(
            "Model A has R²=0.92 on 5 features. Model B has R²=0.93 "
            "on 50 features. Which is better? Why?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.metrics import r2_score\n"
            "\n"
            "r2 = r2_score(y_test, y_pred)\n"
            "print(f'R²: {r2:.4f}')\n"
            "\n"
            "# Adjusted R²\n"
            "n, p = X_test.shape\n"
            "adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)\n"
            "print(f'Adjusted R²: {adj_r2:.4f}')\n"
            "```"
        ),
        keywords=["r-squared", "r2", "variance", "explained", "score"],
    ),

    # ── 14 ─────────────────────────────────────────────────────────
    "ridge_regression": T(
        title="Ridge Regression",
        module="regression",
        what=(
            "Ridge adds L2 regularization to linear regression: it "
            "penalises large coefficients. This prevents overfitting "
            "and handles multicollinearity."
        ),
        why=(
            "When features are correlated (multicollinearity), standard "
            "linear regression has unstable, large coefficients. Ridge "
            "shrinks them towards zero, improving generalisation."
        ),
        when=(
            "Use when: you have many features, features are correlated, "
            "or the model overfits. Tune alpha (regularisation strength)."
        ),
        example=(
            "```python\n"
            "from sklearn.linear_model import Ridge\n"
            "\n"
            "ridge = Ridge(alpha=1.0)\n"
            "ridge.fit(X_train, y_train)\n"
            "print(f'R² train: {ridge.score(X_train, y_train):.4f}')\n"
            "print(f'R² test:  {ridge.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Using too large alpha — underfits by shrinking all coefficients too much.",
            "Not scaling features before Ridge — regularization is scale-dependent.",
            "Forgetting that Ridge doesn't perform feature selection.",
        ],
        interpretation=(
            "alpha=0 → same as Linear Regression. Larger alpha → "
            "smaller coefficients → more regularisation. Ridge shrinks "
            "but never sets coefficients to exactly zero."
        ),
        think_about_it=(
            "Linear Regression gives R²=0.85 train, 0.80 test. "
            "Ridge(alpha=10) gives 0.84 train, 0.83 test. Which is "
            "the better model?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('ridge', Ridge(alpha=1.0))\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "print(f'R²: {pipe.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        keywords=["ridge", "l2", "regularization", "penalty", "coefficients"],
    ),

    # ── 15 ─────────────────────────────────────────────────────────
    "lasso_regression": T(
        title="Lasso Regression",
        module="regression",
        what=(
            "Lasso adds L1 regularization: it penalises the sum of "
            "absolute coefficient values. Unlike Ridge, Lasso can "
            "shrink coefficients to exactly zero, performing feature "
            "selection."
        ),
        why=(
            "Lasso automatically selects features by zeroing out "
            "irrelevant ones. This makes the model simpler and more "
            "interpretable."
        ),
        when=(
            "Use when you suspect many features are irrelevant. "
            "Tune alpha to control how many features are zeroed out."
        ),
        example=(
            "```python\n"
            "from sklearn.linear_model import Lasso\n"
            "\n"
            "lasso = Lasso(alpha=1.0)\n"
            "lasso.fit(X_train, y_train)\n"
            "\n"
            "zero_coefs = (lasso.coef_ == 0).sum()\n"
            "print(f'Zeroed out {zero_coefs} of {len(lasso.coef_)} features')\n"
            "```"
        ),
        mistakes=[
            "Using too large alpha — zeros out too many useful features.",
            "Not scaling features — Lasso is scale-dependent.",
            "Expecting Lasso to always outperform Ridge.",
        ],
        interpretation=(
            "Lasso sets some coefficients to 0, effectively removing "
            "features. The remaining features are the most important. "
            "Use cross-validation to choose alpha."
        ),
        think_about_it=(
            "You have 50 features. Lasso with alpha=0.1 keeps 30, "
            "with alpha=1.0 keeps 8. How do you choose the right alpha?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import LassoCV\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "# Lasso with cross-validated alpha selection\n"
            "scaler = StandardScaler()\n"
            "X_train_s = scaler.fit_transform(X_train)\n"
            "lasso = LassoCV(cv=5)\n"
            "lasso.fit(X_train_s, y_train)\n"
            "print(f'Best alpha: {lasso.alpha_:.4f}')\n"
            "print(f'Features kept: {(lasso.coef_ != 0).sum()}')\n"
            "```"
        ),
        keywords=["lasso", "l1", "feature selection", "sparse", "zero"],
    ),

    # ── 16 ─────────────────────────────────────────────────────────
    "polynomial_regression": T(
        title="Polynomial Regression",
        module="regression",
        what=(
            "Polynomial regression adds polynomial terms (x², x³, "
            "interactions) to capture non-linear relationships while "
            "still using linear regression."
        ),
        why=(
            "Real relationships are often non-linear. Polynomial "
            "features let linear models capture curves without "
            "switching to a non-linear algorithm."
        ),
        when=(
            "Use when the residual plot shows a non-linear pattern. "
            "Start with degree=2. Higher degrees overfit easily."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('poly', PolynomialFeatures(degree=2)),\n"
            "    ('lr', LinearRegression())\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "print(f'R²: {pipe.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Using high degree (>3) — overfits badly.",
            "Not regularising polynomial features (combine with Ridge).",
            "Adding polynomial features without checking multicollinearity.",
        ],
        interpretation=(
            "Degree=2 adds squared terms and interactions. This captures "
            "curves and combined effects. Check: does the added complexity "
            "improve test performance?"
        ),
        think_about_it=(
            "Polynomial degree=5 gives R²=0.99 on train and 0.75 on "
            "test. Degree=2 gives 0.88 on both. Which is better?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('poly', PolynomialFeatures(degree=2, include_bias=False)),\n"
            "    ('ridge', Ridge(alpha=1.0))\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "```"
        ),
        keywords=["polynomial", "degree", "nonlinear", "curve", "features"],
    ),

    # ── 17 ─────────────────────────────────────────────────────────
    "decision_tree_regression": T(
        title="Decision Tree Regression",
        module="regression",
        what=(
            "Decision Tree Regression splits data into regions and "
            "predicts the mean target value in each region. It "
            "creates a step function."
        ),
        why=(
            "Tree regression captures non-linear relationships "
            "automatically. It handles feature interactions and "
            "mixed data types without preprocessing."
        ),
        when=(
            "Use for baseline non-linear regression. Control "
            "complexity with max_depth. Combine in ensembles "
            "(Random Forest, Gradient Boosting) for better results."
        ),
        example=(
            "```python\n"
            "from sklearn.tree import DecisionTreeRegressor\n"
            "\n"
            "tree = DecisionTreeRegressor(max_depth=5, random_state=42)\n"
            "tree.fit(X_train, y_train)\n"
            "print(f'R² train: {tree.score(X_train, y_train):.4f}')\n"
            "print(f'R² test:  {tree.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Using without max_depth — unpruned trees overfit completely.",
            "Expecting smooth predictions — trees create step functions.",
            "Using alone instead of in ensembles.",
        ],
        interpretation=(
            "Tree regression predictions are piecewise constant — "
            "the same value for all samples in a leaf. More leaves = "
            "more flexible but more prone to overfitting."
        ),
        think_about_it=(
            "A decision tree regressor achieves R²=1.0 on training "
            "and 0.6 on test. Why, and how would you fix it?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.tree import DecisionTreeRegressor\n"
            "\n"
            "tree = DecisionTreeRegressor(\n"
            "    max_depth=5,\n"
            "    min_samples_leaf=10,\n"
            "    random_state=42\n"
            ")\n"
            "tree.fit(X_train, y_train)\n"
            "print(f'Feature importances: {tree.feature_importances_}')\n"
            "```"
        ),
        keywords=["decision", "tree", "step", "nonlinear", "regression"],
    ),

    # ── 18 ─────────────────────────────────────────────────────────
    "random_forest_regression": T(
        title="Random Forest Regression",
        module="regression",
        what=(
            "Random Forest Regression averages predictions from many "
            "decision trees trained on random subsets. It reduces "
            "overfitting and improves accuracy."
        ),
        why=(
            "Random Forest is one of the best out-of-the-box regression "
            "algorithms. It handles non-linearity, feature interactions, "
            "and outliers well."
        ),
        when=(
            "Use as a strong default regressor. Works well for most "
            "tabular data. Tune n_estimators, max_depth, "
            "min_samples_leaf."
        ),
        example=(
            "```python\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
            "\n"
            "rf = RandomForestRegressor(\n"
            "    n_estimators=100,\n"
            "    max_depth=10,\n"
            "    random_state=42\n"
            ")\n"
            "rf.fit(X_train, y_train)\n"
            "print(f'R²: {rf.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Not tuning max_depth — default 'None' lets trees overfit.",
            "Using too few trees — more trees improve stability.",
            "Ignoring feature importances.",
        ],
        interpretation=(
            "Feature importance shows which features drive predictions. "
            "Random Forest provides out-of-bag (OOB) score as a "
            "built-in validation estimate."
        ),
        think_about_it=(
            "Random Forest gives R²=0.92 with 100 trees and 0.93 "
            "with 500 trees. Is the extra computation worth it?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
            "import pandas as pd\n"
            "\n"
            "rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)\n"
            "rf.fit(X_train, y_train)\n"
            "\n"
            "imp = pd.Series(rf.feature_importances_, index=feature_names)\n"
            "imp.sort_values(ascending=False).head(10).plot(kind='bar')\n"
            "```"
        ),
        keywords=["random", "forest", "regression", "ensemble", "trees"],
    ),

    # ── 19 ─────────────────────────────────────────────────────────
    "gradient_boosting_regression": T(
        title="Gradient Boosting Regression",
        module="regression",
        what=(
            "Gradient Boosting builds regression trees sequentially, "
            "each correcting the errors of the previous ones. It's "
            "among the highest-performing regression algorithms."
        ),
        why=(
            "Gradient Boosting often achieves the best performance on "
            "tabular data. It's used in winning Kaggle solutions and "
            "production systems."
        ),
        when=(
            "Use when maximum accuracy is needed. Tune learning_rate "
            "(0.01-0.3) and n_estimators carefully. Use early stopping "
            "to prevent overfitting."
        ),
        example=(
            "```python\n"
            "from sklearn.ensemble import GradientBoostingRegressor\n"
            "\n"
            "gb = GradientBoostingRegressor(\n"
            "    n_estimators=200,\n"
            "    learning_rate=0.1,\n"
            "    max_depth=4,\n"
            "    random_state=42\n"
            ")\n"
            "gb.fit(X_train, y_train)\n"
            "print(f'R²: {gb.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Using too high learning_rate with many trees — overfits.",
            "Not using early stopping.",
            "Ignoring training time — Gradient Boosting is slower than Random Forest.",
        ],
        interpretation=(
            "Lower learning_rate with more trees usually gives better "
            "generalisation. The learning_rate controls how much each "
            "tree contributes."
        ),
        think_about_it=(
            "GBR with lr=0.1, 100 trees gives R²=0.89. "
            "GBR with lr=0.01, 1000 trees gives R²=0.91. "
            "Is the 2% improvement worth 10x more trees?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.ensemble import GradientBoostingRegressor\n"
            "\n"
            "gb = GradientBoostingRegressor(\n"
            "    n_estimators=200,\n"
            "    learning_rate=0.05,\n"
            "    max_depth=4,\n"
            "    subsample=0.8,\n"
            "    random_state=42\n"
            ")\n"
            "gb.fit(X_train, y_train)\n"
            "```"
        ),
        keywords=["gradient", "boosting", "regression", "ensemble", "sequential"],
    ),

    # ── 20 ─────────────────────────────────────────────────────────
    "knn_regression": T(
        title="KNN Regression",
        module="regression",
        what=(
            "KNN Regression predicts the target as the average of "
            "the K nearest training samples. It's a non-parametric "
            "method that makes no assumptions about the data."
        ),
        why=(
            "KNN regression is intuitive and works for non-linear "
            "relationships. It's useful as a baseline and for "
            "understanding local patterns."
        ),
        when=(
            "Use for small datasets (<10K). Always scale features. "
            "Choose K: small K = noisy predictions, large K = "
            "over-smoothed."
        ),
        example=(
            "```python\n"
            "from sklearn.neighbors import KNeighborsRegressor\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_train_s = scaler.fit_transform(X_train)\n"
            "X_test_s = scaler.transform(X_test)\n"
            "\n"
            "knn = KNeighborsRegressor(n_neighbors=5)\n"
            "knn.fit(X_train_s, y_train)\n"
            "print(f'R²: {knn.score(X_test_s, y_test):.4f}')\n"
            "```"
        ),
        mistakes=[
            "Not scaling features — dominates by large-magnitude features.",
            "Using on large datasets — prediction is O(n) per point.",
            "Choosing K=1 — too sensitive to noise.",
        ],
        interpretation=(
            "KNN regression gives smooth predictions for K>1. "
            "The prediction is the mean of K neighbors. Larger K "
            "smooths more but may miss local patterns."
        ),
        think_about_it=(
            "KNN with K=3 gives good predictions but is slow on "
            "100K samples. What algorithms could give similar quality "
            "with faster prediction?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.neighbors import KNeighborsRegressor\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('knn', KNeighborsRegressor(n_neighbors=5))\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "print(f'R²: {pipe.score(X_test, y_test):.4f}')\n"
            "```"
        ),
        keywords=["knn", "k-nearest", "neighbors", "local", "average"],
    ),

    # ── 21 ─────────────────────────────────────────────────────────
    "regression_underfitting": T(
        title="Underfitting in Regression",
        module="regression",
        what=(
            "Underfitting occurs when the model is too simple to "
            "capture the underlying relationship. Both training and "
            "test R² are low."
        ),
        why=(
            "An underfitting model cannot learn the data's patterns. "
            "No amount of tuning can fix it — you need more features "
            "or a more complex model."
        ),
        when=(
            "When train R² is low (<0.5) and test R² is similarly low. "
            "Common with linear models on non-linear data."
        ),
        example=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "\n"
            "lr = LinearRegression()\n"
            "lr.fit(X_train, y_train)\n"
            "print(f'Train R²: {lr.score(X_train, y_train):.4f}')  # 0.35\n"
            "print(f'Test R²:  {lr.score(X_test, y_test):.4f}')    # 0.33\n"
            "# Both low → underfitting\n"
            "```"
        ),
        mistakes=[
            "Adding more data when the model is too simple.",
            "Not checking residual plots for non-linear patterns.",
            "Sticking with the same model type.",
        ],
        interpretation=(
            "Solutions: add polynomial features, use a non-linear model "
            "(Random Forest, Gradient Boosting), or engineer better "
            "features."
        ),
        think_about_it=(
            "Linear regression gives R²=0.30. Random Forest gives 0.85. "
            "What does this tell you about the data?"
        ),
        code_link=(
            "```python\n"
            "# Instead of linear regression:\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "\n"
            "# Option 1: Non-linear model\n"
            "rf = RandomForestRegressor(n_estimators=100)\n"
            "\n"
            "# Option 2: Polynomial features\n"
            "from sklearn.pipeline import Pipeline\n"
            "pipe = Pipeline([\n"
            "    ('poly', PolynomialFeatures(degree=2)),\n"
            "    ('lr', LinearRegression())\n"
            "])\n"
            "```"
        ),
        keywords=["underfit", "simple", "bias", "linear", "complexity"],
    ),

    # ── 22 ─────────────────────────────────────────────────────────
    "regression_overfitting": T(
        title="Overfitting in Regression",
        module="regression",
        what=(
            "Overfitting in regression occurs when the model memorises "
            "training data. Training R² is high but test R² is much lower."
        ),
        why=(
            "Overfitting produces models that look great on paper but "
            "fail on new data. It's the most common regression failure."
        ),
        when=(
            "When train R² ≫ test R². Common with deep decision trees, "
            "high-degree polynomials, and unregularised models."
        ),
        example=(
            "```python\n"
            "from sklearn.tree import DecisionTreeRegressor\n"
            "\n"
            "tree = DecisionTreeRegressor()  # no max_depth\n"
            "tree.fit(X_train, y_train)\n"
            "print(f'Train R²: {tree.score(X_train, y_train):.4f}')  # 1.00\n"
            "print(f'Test R²:  {tree.score(X_test, y_test):.4f}')    # 0.55\n"
            "# Large gap → overfitting\n"
            "```"
        ),
        mistakes=[
            "Only looking at training R².",
            "Not using cross-validation.",
            "Adding more features without checking relevance.",
        ],
        interpretation=(
            "Solutions: regularise (Ridge/Lasso), limit tree depth, "
            "reduce features, get more data, or use ensemble methods."
        ),
        think_about_it=(
            "Your model has train R²=0.99, test R²=0.72. "
            "Name three strategies to improve the test score."
        ),
        code_link=(
            "```python\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "# Ridge to prevent overfitting\n"
            "ridge = Ridge(alpha=10.0)\n"
            "scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')\n"
            "print(f'CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})')\n"
            "```"
        ),
        keywords=["overfit", "complexity", "variance", "regularization", "gap"],
    ),

    # ── 23 ─────────────────────────────────────────────────────────
    "regression_assumptions": T(
        title="Regression Assumptions",
        module="regression",
        what=(
            "Linear regression assumes: (1) linear relationship, "
            "(2) independent residuals, (3) normally distributed "
            "residuals, (4) constant variance (homoscedasticity), "
            "(5) no multicollinearity."
        ),
        why=(
            "Violating assumptions makes coefficient estimates "
            "unreliable and predictions biased. Checking assumptions "
            "is part of good regression practice."
        ),
        when=(
            "After training a linear model. Check residual plots "
            "for linearity, normality (Q-Q plot), and "
            "homoscedasticity."
        ),
        example=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "\n"
            "residuals = y_test - model.predict(X_test)\n"
            "\n"
            "# Check normality\n"
            "from scipy import stats\n"
            "stats.probplot(residuals, dist='norm', plot=plt)\n"
            "plt.show()\n"
            "\n"
            "# Check homoscedasticity\n"
            "plt.scatter(model.predict(X_test), residuals)\n"
            "plt.axhline(y=0, color='r')\n"
            "plt.show()\n"
            "```"
        ),
        mistakes=[
            "Ignoring assumptions — results may be misleading.",
            "Trying to fix assumptions instead of the underlying model.",
            "Assuming trees need these assumptions — they don't.",
        ],
        interpretation=(
            "If residuals show a pattern, the linear model is wrong. "
            "Non-linear patterns → add features or use non-linear model. "
            "Funnel shape → heteroscedasticity (use weighted regression)."
        ),
        think_about_it=(
            "Your residual plot shows a funnel shape (spread increases "
            "with predicted value). What does this violate, and how "
            "would you fix it?"
        ),
        code_link=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from scipy import stats\n"
            "\n"
            "residuals = y_test - model.predict(X_test)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "axes[0].scatter(model.predict(X_test), residuals, alpha=0.5)\n"
            "axes[0].set_title('Residuals vs Predicted')\n"
            "stats.probplot(residuals, dist='norm', plot=axes[1])\n"
            "axes[1].set_title('Q-Q Plot')\n"
            "axes[2].hist(residuals, bins=30)\n"
            "axes[2].set_title('Residual Distribution')\n"
            "plt.tight_layout()\n"
            "```"
        ),
        keywords=["assumptions", "linearity", "normality", "homoscedasticity", "diagnostic"],
    ),

    # ── 24 ─────────────────────────────────────────────────────────
    "residual_analysis": T(
        title="Residual Analysis",
        module="regression",
        what=(
            "Residual analysis examines the differences between "
            "actual and predicted values to diagnose model problems. "
            "It checks whether the model's assumptions hold."
        ),
        why=(
            "Residual analysis is the primary diagnostic tool for "
            "regression. It reveals whether the model is appropriate, "
            "whether assumptions are met, and where improvements are "
            "needed."
        ),
        when=(
            "After training any regression model. Always check: "
            "residual plot, Q-Q plot, and residual histogram."
        ),
        example=(
            "```python\n"
            "y_pred = model.predict(X_test)\n"
            "residuals = y_test - y_pred\n"
            "\n"
            "print(f'Mean residual: {residuals.mean():.4f}')  # should be ~0\n"
            "print(f'Std residual:  {residuals.std():.4f}')\n"
            "```"
        ),
        mistakes=[
            "Only looking at aggregate metrics (R², RMSE) without residual plots.",
            "Ignoring patterns that suggest non-linearity.",
            "Not checking for outliers in residuals.",
        ],
        interpretation=(
            "Random scatter around 0 → good. U-shape → non-linearity. "
            "Funnel → heteroscedasticity. Points far from 0 → outliers. "
            "Skewed distribution → non-normal errors."
        ),
        think_about_it=(
            "After residual analysis, you find: (1) residuals are "
            "roughly normal, (2) mean is 0, (3) no funnel pattern, "
            "(4) one point has residual=500,000. What should you do?"
        ),
        code_link=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from scipy import stats\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "residuals = y_test - y_pred\n"
            "\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n"
            "axes[0, 0].scatter(y_pred, residuals, alpha=0.5)\n"
            "axes[0, 0].axhline(y=0, color='r')\n"
            "axes[0, 0].set_title('Residuals vs Predicted')\n"
            "stats.probplot(residuals, dist='norm', plot=axes[0, 1])\n"
            "axes[0, 1].set_title('Q-Q Plot')\n"
            "axes[1, 0].hist(residuals, bins=30)\n"
            "axes[1, 0].set_title('Residual Distribution')\n"
            "axes[1, 1].scatter(y_test, y_pred, alpha=0.5)\n"
            "axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')\n"
            "axes[1, 1].set_title('Actual vs Predicted')\n"
            "plt.tight_layout()\n"
            "```"
        ),
        keywords=["residual", "diagnostic", "plot", "normal", "pattern"],
    ),

    # ── 25 ─────────────────────────────────────────────────────────
    "regression_case_study": T(
        title="Regression Case Study",
        module="regression",
        what=(
            "A complete regression workflow on the California Housing "
            "dataset: loading, preprocessing, training multiple models, "
            "and evaluation."
        ),
        why=(
            "Seeing the full workflow connects individual concepts "
            "into a practical regression pipeline."
        ),
        when=(
            "Reference this workflow for any regression project."
        ),
        example="Complete California Housing prediction workflow.",
        mistakes=[
            "Not comparing multiple models.",
            "Only looking at R² without residual analysis.",
            "Skipping preprocessing.",
        ],
        interpretation=(
            "The best regression model balances R², RMSE, "
            "interpretability, and training time."
        ),
        think_about_it=(
            "After completing this workflow, your best model has "
            "R²=0.80 and RMSE=$45,000. What next steps would you try?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.datasets import fetch_california_housing\n"
            "from sklearn.model_selection import train_test_split, cross_val_score\n"
            "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n"
            "from sklearn.linear_model import LinearRegression, Ridge\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.metrics import r2_score, mean_squared_error\n"
            "import numpy as np\n"
            "\n"
            "# Load\n"
            "data = fetch_california_housing()\n"
            "X, y = data.data, data.target\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
            "\n"
            "# Compare models\n"
            "models = {\n"
            "    'LR': LinearRegression(),\n"
            "    'Ridge': Ridge(alpha=1.0),\n"
            "    'RF': RandomForestRegressor(n_estimators=100),\n"
            "    'GBR': GradientBoostingRegressor(n_estimators=100),\n"
            "}\n"
            "\n"
            "for name, model in models.items():\n"
            "    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')\n"
            "    print(f'{name}: R²={scores.mean():.4f} (+/- {scores.std():.4f})')\n"
            "```"
        ),
        keywords=["case study", "workflow", "end-to-end", "complete", "housing"],
    ),
}
