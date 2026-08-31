"""Regression curriculum — 25 topics with full educational content."""

from learning import QuizQuestion, Topic

TOPICS = [
    Topic(
        id="reg_01", title="What is Regression?", section="regression", order=1,
        difficulty="beginner",
        objectives=[
            "Define regression and its goal",
            "Identify regression problems in real-world scenarios",
            "Distinguish regression from other ML tasks",
        ],
        concept=(
            "Regression is a supervised learning technique where the model learns a mapping "
            "from input features to a continuous numerical output. Given feature vector X, "
            "the model predicts y ∈ ℝ. The goal is to approximate the true function f(X) = y "
            "so that predictions on unseen data are accurate."
        ),
        why_matters=(
            "Regression is used in housing price prediction, stock forecasting, medical dosage "
            "optimisation, weather prediction, demand forecasting, and energy consumption "
            "estimation. Nearly any problem involving 'how much' or 'how many' is a regression task."
        ),
        simple_explanation=(
            "Regression answers: 'How much?' How much will this house sell for? "
            "How many customers will visit today? How many degrees will the temperature drop?"
        ),
        example=(
            "Consider predicting house prices based on features like size, location, "
            "and number of bedrooms. The model learns that each additional bedroom adds "
            "approximately £15,000 to the price, and proximity to a school adds £20,000."
        ),
        python_example=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "import pandas as pd\n\n"
            "# Load data\n"
            "df = pd.read_csv('housing.csv')\n"
            "X = df[['size_sqft', 'bedrooms', 'distance_to_school']]\n"
            "y = df['price']  # continuous target\n\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)\n"
            "predictions = model.predict(X_test)\n"
            "```"
        ),
        interpretation=(
            "The model outputs continuous numbers. A prediction of £285,000 means the model "
            "estimates the house is worth that amount. The error (e.g., MAE = £15,000) tells "
            "you how far off predictions are on average."
        ),
        common_mistakes=[
            "Using classification metrics (accuracy) for regression problems",
            "Treating ordinal categories ('low/medium/high') as regression targets without encoding",
            "Not checking whether the relationship between features and target is roughly linear",
            "Ignoring outlier target values that can heavily influence the model",
        ],
        practice_exercise=(
            "Using the California Housing dataset in the Dataset Explorer:\n"
            "1. Identify which columns could serve as features and which is the target.\n"
            "2. What type of problem is this — classification or regression?\n"
            "3. What would be a reasonable baseline prediction? (Hint: predict the mean.)"
        ),
        quiz=[
            QuizQuestion(
                question="Which of the following is a regression problem?",
                options=[
                    "Predicting whether an email is spam or not",
                    "Predicting the temperature tomorrow",
                    "Predicting the species of a flower",
                    "Predicting whether a customer will churn",
                ],
                correct_index=1,
                explanation=(
                    "Predicting temperature is regression because temperature is a continuous value. "
                    "Spam detection, species prediction, and churn prediction are all classification "
                    "problems with discrete outcomes."
                ),
            ),
        ],
        takeaways=[
            "Regression predicts continuous numerical values",
            "The target variable must be continuous (or treated as such)",
            "Linear regression is the foundational starting point",
            "Different metrics apply than classification (MAE, RMSE, R²)",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_02", title="Regression vs Classification", section="regression", order=2,
        difficulty="beginner",
        objectives=[
            "Distinguish regression from classification",
            "Identify when each is appropriate",
            "Handle borderline cases where the distinction is unclear",
        ],
        concept=(
            "Regression predicts continuous output (price, temperature, salary). "
            "Classification predicts discrete output (spam/not spam, species, yes/no). "
            "The nature of the target variable determines the problem type."
        ),
        why_matters=(
            "Choosing the wrong type leads to incorrect models, wrong metrics, and meaningless results. "
            "Predicting house price as categories ('low/medium/high') loses valuable information "
            "that a continuous prediction preserves."
        ),
        simple_explanation=(
            "'How much?' → Regression. 'Which category?' → Classification."
        ),
        example=(
            "A hospital wants to predict:\n"
            "• Patient blood pressure (continuous) → Regression\n"
            "• Whether patient has diabetes (yes/no) → Classification\n"
            "• Patient risk score 0-100 (continuous) → Regression\n"
            "• Risk category low/medium/high (discrete) → Classification"
        ),
        common_mistakes=[
            "Using regression for categorical target variables",
            "Using classification for continuous targets by discretising them unnecessarily",
            "Treating label-encoded categories (0,1,2) as a regression target",
        ],
        practice_exercise=(
            "Classify each scenario as regression or classification:\n"
            "1. Predicting exam scores (0-100)\n"
            "2. Predicting whether a student passes or fails\n"
            "3. Predicting the number of customer complaints per month\n"
            "4. Predicting product quality grade (A/B/C/D)"
        ),
        quiz=[
            QuizQuestion(
                question="A dataset has a column 'customer_rating' with values 1.0, 2.5, 3.7, 4.2 (out of 5). Is predicting this regression or classification?",
                options=[
                    "Regression — the values are continuous decimals",
                    "Classification — ratings are discrete categories",
                    "It depends on whether the values are truly continuous or just appear so",
                    "Neither — ratings are ordinal, not numerical",
                ],
                correct_index=0,
                explanation=(
                    "Since the values are continuous decimals (1.0, 2.5, 3.7, 4.2), this is a "
                    "regression problem. If ratings were only integers (1, 2, 3, 4, 5), you "
                    "could argue either way, but continuous values make it clearly regression."
                ),
            ),
        ],
        takeaways=[
            "Target variable type determines the problem type",
            "Continuous output → regression, discrete output → classification",
            "The same features can be used for either, depending on the target",
            "Discretising a continuous target loses information",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_03", title="Continuous Target Variables", section="regression", order=3,
        difficulty="beginner",
        objectives=[
            "Understand what continuous variables are",
            "Check the distribution of the target variable",
            "Apply log transformation to skewed targets",
        ],
        concept=(
            "Continuous variables take any real value within a range (e.g., £50,000.50, 23.7°C). "
            "Unlike discrete counts, they can be fractional. The target distribution affects model "
            "choice: linear models assume roughly normal residuals, so skewed targets often need "
            "transformation."
        ),
        why_matters=(
            "If the target is heavily skewed (e.g., house prices with a long right tail), "
            "the model will be biased toward predicting the mean. A log transform of the target "
            "often normalises the distribution and improves predictions."
        ),
        example=(
            "House prices: most are £100K-£300K, but a few are £5M+. The distribution is "
            "right-skewed. Log-transforming the target (log(price)) makes the distribution "
            "more symmetric, helping linear models perform better."
        ),
        python_example=(
            "```python\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n\n"
            "# Check target distribution\n"
            "plt.hist(y_train, bins=50)\n"
            "plt.title('Target Distribution')\n"
            "plt.show()\n\n"
            "# If skewed, apply log transform\n"
            "y_train_log = np.log1p(y_train)  # log(1 + x) handles zeros\n"
            "plt.hist(y_train_log, bins=50)\n"
            "plt.title('Log-Transformed Target')\n"
            "plt.show()\n\n"
            "# Train on transformed target\n"
            "model.fit(X_train, y_train_log)\n"
            "# Reverse transform predictions\n"
            "y_pred = np.expm1(model.predict(X_test))  # exp(x) - 1\n"
            "```"
        ),
        interpretation=(
            "If the histogram shows a long right tail, log transformation will help. "
            "After transformation, check that the distribution is more symmetric. "
            "Always reverse the transformation on predictions."
        ),
        common_mistakes=[
            "Not checking target distribution before modelling",
            "Using linear models on highly skewed targets without transformation",
            "Forgetting to reverse the log transformation on predictions",
            "Using np.log() instead of np.log1p() when the target contains zeros",
        ],
        practice_exercise=(
            "Load the California Housing dataset. Plot the distribution of the 'MedHouseVal' column. "
            "Is it skewed? If so, try np.log1p() transformation and plot again. "
            "Which distribution looks more normal?"
        ),
        quiz=[
            QuizQuestion(
                question="You have a target variable 'income' with values ranging from £20,000 to £500,000, heavily right-skewed. What should you do first?",
                options=[
                    "Use a decision tree which handles any distribution",
                    "Apply log transformation to the target",
                    "Remove the highest values as outliers",
                    "Nothing — most models work fine with skewed targets",
                ],
                correct_index=1,
                explanation=(
                    "Log transformation normalises skewed targets. This helps linear models "
                    "which assume roughly normal residuals. Removing high values loses data, "
                    "and while trees handle skew better, log transformation improves linear models."
                ),
            ),
        ],
        takeaways=[
            "Always check target distribution before modelling",
            "Log transform (np.log1p) helps with right-skewed targets",
            "Remember to reverse the transformation (np.expm1) on predictions",
            "Skewed targets bias linear models toward predicting the mean",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_04", title="Simple Linear Regression", section="regression", order=4,
        difficulty="beginner",
        objectives=[
            "Understand the equation y = β₀ + β₁x",
            "Fit a simple linear model with one feature",
            "Interpret slope and intercept in context",
        ],
        concept=(
            "Simple linear regression fits a straight line through the data: y = β₀ + β₁x. "
            "β₀ is the intercept (predicted y when x = 0). β₁ is the slope (change in y "
            "for each one-unit increase in x). The line minimises the sum of squared residuals."
        ),
        why_matters=(
            "Despite its simplicity, linear regression is the foundation of all regression. "
            "Understanding it is essential before moving to complex models. Its coefficients "
            "are directly interpretable — each coefficient tells you the effect of a feature."
        ),
        example=(
            "Predicting exam score from hours studied:\n"
            "• Intercept (β₀) = 45: a student who studies 0 hours scores ~45\n"
            "• Slope (β₁) = 5: each additional hour of study adds ~5 marks\n"
            "• Equation: score = 45 + 5 × hours_studied"
        ),
        python_example=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "import numpy as np\n\n"
            "model = LinearRegression()\n"
            "model.fit(X_train[['hours_studied']], y_train)\n\n"
            "print(f'Intercept: {model.intercept_:.3f}')\n"
            "print(f'Slope: {model.coef_[0]:.3f}')\n\n"
            "# Predict for a student studying 7 hours\n"
            "model.predict([[7]])  # → 45 + 5*7 = 80\n"
            "```"
        ),
        interpretation=(
            "The slope tells you the rate of change: each additional hour of study "
            "increases the predicted score by the slope value. The intercept is the "
            "baseline prediction when x = 0. Note: the intercept may not always be "
            "meaningful (e.g., a house with 0 square feet)."
        ),
        common_mistakes=[
            "Interpreting the intercept literally when x = 0 is outside the data range",
            "Assuming a linear relationship without first plotting a scatter plot",
            "Using simple linear regression when you have multiple features (use multiple regression)",
            "Extrapolating predictions far beyond the training data range",
        ],
        practice_exercise=(
            "Using the California Housing dataset, fit a simple linear regression using "
            "only 'MedInc' (median income) to predict 'MedHouseVal'. "
            "1. What is the slope? What does it mean?\n"
            "2. What is the intercept? Is it meaningful?\n"
            "3. Plot the scatter plot with the regression line."
        ),
        quiz=[
            QuizQuestion(
                question="A linear regression model gives: price = 50000 + 150 × size_sqft. What does the coefficient 150 mean?",
                options=[
                    "The house price is £150",
                    "Each additional square foot increases the predicted price by £150",
                    "The model is 150% accurate",
                    "There are 150 features in the model",
                ],
                correct_index=1,
                explanation=(
                    "The coefficient 150 means that for each one-unit increase in size_sqft "
                    "(each additional square foot), the predicted price increases by £150, "
                    "holding all other features constant."
                ),
            ),
        ],
        takeaways=[
            "y = intercept + slope × x",
            "Slope = how much y changes per one-unit increase in x",
            "Intercept = predicted y when x = 0 (may not be meaningful)",
            "Always check linearity with a scatter plot before fitting",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_05", title="Multiple Linear Regression", section="regression", order=5,
        difficulty="beginner",
        objectives=[
            "Extend linear regression to multiple features",
            "Interpret multiple coefficients",
            "Understand adjusted R²",
        ],
        concept=(
            "Multiple linear regression: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ. Each coefficient "
            "represents the effect of its feature, holding all others constant (ceteris paribus). "
            "This captures the combined effect of multiple predictors simultaneously."
        ),
        why_matters=(
            "Real-world prediction involves many factors. A house price depends on size, "
            "location, age, condition, and more — all simultaneously. Multiple regression "
            "captures this reality."
        ),
        example=(
            "House price = £30,000 + £100 × size + £5,000 × bedrooms - £200 × age_years\n"
            "• Each additional sq ft adds £100 to the price\n"
            "• Each additional bedroom adds £5,000\n"
            "• Each year of age decreases the price by £200\n"
            "All effects are estimated simultaneously."
        ),
        python_example=(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)  # X has multiple columns\n\n"
            "for name, coef in zip(feature_names, model.coef_):\n"
            "    print(f'{name}: {coef:.3f}')\n"
            "print(f'Intercept: {model.intercept_:.3f}')\n"
            "print(f'R²: {model.score(X_test, y_test):.3f}')\n"
            "```"
        ),
        interpretation=(
            "Each coefficient is the effect of that feature while holding all other features "
            "constant. A coefficient of -500 for 'distance_to_centre' means each additional km "
            "from the centre decreases the predicted value by 500, regardless of the house's "
            "size or number of bedrooms."
        ),
        common_mistakes=[
            "Interpreting coefficients without considering multicollinearity (correlated features distort coefficients)",
            "Ignoring adjusted R² — plain R² always increases with more features even if they are useless",
            "Not scaling features before comparing coefficient magnitudes",
            "Assuming causation from correlation in coefficients",
        ],
        practice_exercise=(
            "Fit a multiple linear regression on California Housing using all numerical features. "
            "1. Which feature has the largest absolute coefficient?\n"
            "2. Does a large coefficient mean the feature is most important?\n"
            "3. Compare R² with the simple linear regression (single feature)."
        ),
        quiz=[
            QuizQuestion(
                question="In multiple linear regression, what does a coefficient of -3.2 for 'room_temperature' mean?",
                options=[
                    "The room temperature is -3.2 degrees",
                    "For each one-unit increase in room_temperature, the target decreases by 3.2, holding other features constant",
                    "The model predicts -3.2 for all inputs",
                    "The feature 'room_temperature' has a -3.2% effect",
                ],
                correct_index=1,
                explanation=(
                    "In multiple regression, each coefficient represents the change in the target "
                    "for a one-unit increase in that feature, holding all other features constant. "
                    "The ceteris paribus (all else equal) condition is crucial."
                ),
            ),
        ],
        takeaways=[
            "Multiple regression handles many features simultaneously",
            "Each coefficient = effect holding all other features constant",
            "Use adjusted R² to penalise unnecessary features",
            "Multicollinearity can make individual coefficients unreliable",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_06", title="Least Squares", section="regression", order=6,
        difficulty="beginner",
        objectives=[
            "Understand how regression finds the best line",
            "Interpret the cost function",
            "Know why squared residuals are minimised",
        ],
        concept=(
            "Ordinary Least Squares (OLS) finds the line that minimises the sum of squared "
            "residuals: Σ(y_actual - y_predicted)². Squaring ensures positive and negative "
            "errors don't cancel, penalises large errors more, and yields a unique analytical solution."
        ),
        why_matters=(
            "OLS is the mathematical foundation of linear regression. Understanding it explains "
            "why regression lines look the way they do, why outliers have disproportionate influence, "
            "and how regularised variants (Ridge, Lasso) modify the objective."
        ),
        simple_explanation=(
            "The 'best' line is the one where the total squared distance from all data points "
            "to the line is the smallest possible value."
        ),
        example=(
            "Given 5 data points, OLS finds the line that minimises the sum of squared vertical "
            "distances from each point to the line. If one point is far away, its squared distance "
            "dominates the total, pulling the line toward it."
        ),
        common_mistakes=[
            "Not understanding why we square errors (gives unique solution, penalises large errors)",
            "Assuming OLS always finds the 'true' relationship (it finds the best linear approximation)",
            "Ignoring that outliers heavily influence OLS due to squaring",
        ],
        practice_exercise=(
            "Load a simple dataset with two columns. Fit a linear regression and compute the "
            "residuals manually (actual - predicted). Square them and sum them. "
            "This sum is what OLS minimises."
        ),
        quiz=[
            QuizQuestion(
                question="Why does OLS minimise squared residuals instead of absolute residuals?",
                options=[
                    "Squared residuals are easier to calculate",
                    "Squaring ensures a unique solution and gives more weight to large errors",
                    "Absolute residuals don't work with linear algebra",
                    "It doesn't matter — both give the same result",
                ],
                correct_index=1,
                explanation=(
                    "Squaring has two key benefits: (1) it produces a smooth, differentiable "
                    "function with a unique minimum, and (2) it penalises large errors more than "
                    "small ones, pushing the model to reduce extreme predictions."
                ),
            ),
        ],
        takeaways=[
            "OLS minimises the sum of squared residuals",
            "Squaring penalises large errors more heavily",
            "OLS finds the best linear approximation, not the 'true' relationship",
            "Outliers have disproportionate influence due to squaring",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_07", title="Regression Line", section="regression", order=7,
        difficulty="beginner",
        objectives=[
            "Visualise the regression line on a scatter plot",
            "Understand what the line represents",
            "Identify when a straight line is inappropriate",
        ],
        concept=(
            "The regression line represents the model's predicted value for each feature value. "
            "Points above the line are underpredicted (actual > predicted); points below are "
            "overpredicted. The line is the model's 'best guess' given the data."
        ),
        why_matters=(
            "Visualising the regression line reveals whether linear regression is appropriate. "
            "If points show a curved pattern around the line, a linear model is insufficient."
        ),
        example=(
            "Plotting house price vs. size: points scatter around the line. Houses above the "
            "line are pricier than expected for their size (perhaps in a desirable location). "
            "Houses below are cheaper than expected (perhaps in poor condition)."
        ),
        python_example=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "plt.scatter(X_test['feature'], y_test, alpha=0.5, label='Actual data')\n"
            "x_line = np.linspace(X_test['feature'].min(), X_test['feature'].max(), 100)\n"
            "y_line = model.predict(X_test[['feature']].assign(feature=x_line))\n"  
            "plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression line')\n"
            "plt.xlabel('Feature')\n"
            "plt.ylabel('Target')\n"
            "plt.legend()\n"
            "plt.title('Regression Line Fit')\n"
            "plt.show()\n"
            "```"
        ),
        interpretation=(
            "A good fit: points scatter randomly around the line with no clear pattern. "
            "A poor fit: points show curves, clusters, or increasing/decreasing spread. "
            "The regression line alone doesn't tell you if the model is good — check residuals."
        ),
        common_mistakes=[
            "Assuming the regression line represents the 'true' relationship",
            "Not checking residual plots to assess fit quality",
            "Extrapolating the line far beyond the observed data range",
        ],
        practice_exercise=(
            "Plot a scatter plot of 'MedInc' vs 'MedHouseVal' from California Housing. "
            "Overlay the linear regression line. Does the relationship look linear? "
            "Are there regions where the line systematically over- or under-predicts?"
        ),
        quiz=[
            QuizQuestion(
                question="If all data points lie above the regression line, what does this mean?",
                options=[
                    "The model is overpredicting for all points",
                    "The model is underpredicting for all points",
                    "The regression line is correct but the data is wrong",
                    "Nothing — this is normal",
                ],
                correct_index=1,
                explanation=(
                    "If points lie above the line, actual values are higher than predicted — "
                    "the model is systematically underpredicting. This indicates the model "
                    "is biased, possibly due to missing features or non-linearity."
                ),
            ),
        ],
        takeaways=[
            "Regression line = model's best prediction for each x value",
            "Points above = underpredicted, points below = overpredicted",
            "Visual inspection reveals non-linearity",
            "Always check residuals to complement the visual",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_08", title="Regression Coefficients", section="regression", order=8,
        difficulty="beginner",
        objectives=[
            "Understand intercept and coefficient meaning",
            "Interpret coefficients in the context of the problem",
            "Identify when multicollinearity affects interpretation",
        ],
        concept=(
            "The intercept (β₀) is the predicted y when all features are zero. "
            "Each coefficient (βᵢ) is the predicted change in y for a one-unit increase "
            "in xᵢ, holding all other features constant. Together they define the regression equation."
        ),
        why_matters=(
            "Coefficients are the primary tool for interpreting linear regression. They tell you "
            "which features matter, in what direction, and by how much. This interpretability is "
            "why linear regression remains popular despite lower accuracy than tree-based models."
        ),
        example=(
            "A model predicting salary:\n"
            "• Intercept: £25,000 (base salary with 0 experience, 0 degrees)\n"
            "• Experience coefficient: +£3,000/year (each year adds £3K)\n"
            "• Degree coefficient: +£8,000 (having a degree adds £8K)\n"
            "• Note: intercept may not be realistic (0 experience, 0 degrees is unusual)"
        ),
        python_example=(
            "```python\n"
            "model = LinearRegression()\n"
            "model.fit(X_train, y_train)\n\n"
            "print(f'Intercept: {model.intercept_:.2f}')\n"
            "for name, coef in sorted(\n"
            "    zip(feature_names, model.coef_),\n"
            "    key=lambda x: abs(x[1]), reverse=True\n"
            "):\n"
            "    print(f'  {name}: {coef:.4f}')\n"
            "```"
        ),
        common_mistakes=[
            "Interpreting the intercept when x = 0 is far outside the data range",
            "Comparing coefficient magnitudes when features are on different scales",
            "Ignoring multicollinearity which makes individual coefficients unreliable",
            "Assuming correlation means causation",
        ],
        practice_exercise=(
            "Fit a multiple regression on a dataset with at least 5 features. "
            "1. Rank features by absolute coefficient value.\n"
            "2. Scale all features using StandardScaler and re-fit. Does the ranking change?\n"
            "3. Which interpretation is more trustworthy — before or after scaling?"
        ),
        quiz=[
            QuizQuestion(
                question="Two features are highly correlated (r = 0.95). How does this affect regression coefficients?",
                options=[
                    "No effect — coefficients are independent",
                    "Coefficients become unstable and may flip signs",
                    "Both coefficients increase proportionally",
                    "The model automatically drops one feature",
                ],
                correct_index=1,
                explanation=(
                    "Multicollinearity makes coefficients unstable. The model cannot distinguish "
                    "the individual effects of correlated features, so coefficients may have "
                    "unexpected signs or magnitudes. Use VIF to detect multicollinearity."
                ),
            ),
        ],
        takeaways=[
            "Intercept = baseline prediction when all features = 0",
            "Coefficient = change in target per unit feature change (holding others constant)",
            "Scale features before comparing coefficient magnitudes",
            "Multicollinearity makes individual coefficients unreliable",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_09", title="Residuals", section="regression", order=9,
        difficulty="beginner",
        objectives=[
            "Calculate residuals (actual - predicted)",
            "Plot and interpret residual patterns",
            "Use residuals to diagnose model fit",
        ],
        concept=(
            "A residual is the difference between the actual and predicted value: eᵢ = yᵢ - ŷᵢ. "
            "For a good model, residuals should be randomly scattered around zero with no "
            "discernible pattern. Patterns (curves, funnels) indicate the model is missing structure."
        ),
        why_matters=(
            "Residual analysis is the most important diagnostic for regression. Metrics like R² "
            "tell you HOW WELL the model fits; residual analysis tells you WHETHER the model is "
            "appropriate. A high R² with curved residuals means the linear model is wrong."
        ),
        example=(
            "A model predicting house prices from size: residual plot shows a U-shape. "
            "This means small and large houses are underpredicted, while medium houses are "
            "overpredicted — the relationship is non-linear. A polynomial feature would help."
        ),
        python_example=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "residuals = y_test - model.predict(X_test)\n\n"
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n\n"
            "# 1. Residuals vs Predicted\n"
            "axes[0].scatter(model.predict(X_test), residuals, alpha=0.5)\n"
            "axes[0].axhline(y=0, color='r', linestyle='--')\n"
            "axes[0].set_xlabel('Predicted')\n"
            "axes[0].set_ylabel('Residuals')\n"
            "axes[0].set_title('Residuals vs Predicted')\n\n"
            "# 2. Histogram of residuals\n"
            "axes[1].hist(residuals, bins=30, edgecolor='black')\n"
            "axes[1].set_title('Residual Distribution')\n\n"
            "# 3. Q-Q plot for normality\n"
            "from scipy import stats\n"
            "stats.probplot(residuals, dist='norm', plot=axes[2])\n"
            "axes[2].set_title('Q-Q Plot')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "```"
        ),
        interpretation=(
            "Good residuals: random scatter around 0, approximately normal distribution, "
            "constant spread across predicted values. Bad residuals: curved pattern (non-linearity), "
            "funnel shape (heteroscedasticity), heavy tails (outliers)."
        ),
        common_mistakes=[
            "Only looking at R² without checking residuals",
            "Ignoring funnel shapes that indicate heteroscedasticity",
            "Not checking the Q-Q plot for normality of residuals",
        ],
        practice_exercise=(
            "Fit a linear regression on California Housing. Create all three residual plots "
            "(vs predicted, histogram, Q-Q). Do the residuals look random? "
            "Is there a funnel shape? What does the Q-Q plot tell you?"
        ),
        quiz=[
            QuizQuestion(
                question="Your residual plot shows a clear U-shaped pattern. What does this indicate?",
                options=[
                    "The model has heteroscedasticity",
                    "The relationship is non-linear — a linear model is insufficient",
                    "The residuals are normally distributed",
                    "The model is overfitting",
                ],
                correct_index=1,
                explanation=(
                    "A U-shaped pattern in the residuals means the model systematically "
                    "overpredicts in the middle and underpredicts at the extremes (or vice versa). "
                    "This indicates a non-linear relationship that a straight line cannot capture."
                ),
            ),
        ],
        takeaways=[
            "Residuals = actual - predicted values",
            "Random scatter around zero = good model fit",
            "Curved patterns = non-linearity, funnel shapes = heteroscedasticity",
            "Always check residuals — R² alone is insufficient",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_10", title="Mean Absolute Error (MAE)", section="regression", order=10,
        difficulty="beginner",
        objectives=[
            "Calculate MAE",
            "Interpret MAE in the original units of the target",
            "Compare MAE with other error metrics",
        ],
        concept=(
            "MAE = mean(|actual - predicted|). It measures the average absolute prediction error "
            "in the original units of the target. MAE = £5,000 means predictions are off by "
            "£5,000 on average."
        ),
        why_matters=(
            "MAE is the most intuitive error metric — it directly tells you the average magnitude "
            "of errors in real units. It is robust to outliers because every error is weighted equally."
        ),
        example=(
            "House price predictions: actuals are [£200K, £300K, £400K], predictions are "
            "[£210K, £280K, £420K]. MAE = mean(|10K|, |20K|, |20K|) = £16.7K. "
            "On average, predictions are off by about £17K."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import mean_absolute_error\n"
            "import numpy as np\n\n"
            "y_pred = model.predict(X_test)\n"
            "mae = mean_absolute_error(y_test, y_pred)\n"
            "print(f'MAE: £{mae:,.0f}')  # Average error in pounds\n\n"
            "# Manual calculation\n"
            "mae_manual = np.mean(np.abs(y_test - y_pred))\n"
            "print(f'MAE (manual): £{mae_manual:,.0f}')\n"
            "```"
        ),
        interpretation=(
            "MAE = 0 means perfect predictions. Higher MAE means larger average errors. "
            "MAE is in the same units as the target, so it's directly interpretable. "
            "Compare MAE to the range of the target: if MAE/target_range is small, "
            "the model is performing well."
        ),
        common_mistakes=[
            "Comparing MAE across datasets with different scales",
            "Confusing MAE with MSE (MSE squares errors before averaging)",
            "Not reporting MAE in the original units of the target",
        ],
        practice_exercise=(
            "Train a linear regression on California Housing. Calculate MAE on the test set. "
            "Is a £40,000 average error acceptable for housing predictions? "
            "How does this compare to the range of house values?"
        ),
        quiz=[
            QuizQuestion(
                question="MAE = £25,000 for house price prediction. What does this mean?",
                options=[
                    "The model predicts exactly £25,000 for every house",
                    "On average, the model's predictions are £25,000 away from the actual price",
                    "The model is 25% accurate",
                    "The model overestimates by £25,000 on average",
                ],
                correct_index=1,
                explanation=(
                    "MAE measures the average magnitude of errors, regardless of direction. "
                    "A MAE of £25,000 means the model's predictions are, on average, "
                    "£25,000 away from the true values (could be over or under)."
                ),
            ),
        ],
        takeaways=[
            "MAE = mean absolute prediction error in original units",
            "Easy to interpret: 'off by £X on average'",
            "Robust to outliers (all errors weighted equally)",
            "Compare MAE to the target range to assess practical significance",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_11", title="Mean Squared Error (MSE)", section="regression", order=11,
        difficulty="beginner",
        objectives=[
            "Calculate MSE",
            "Understand why squaring matters",
            "Compare MSE with MAE",
        ],
        concept=(
            "MSE = mean((actual - predicted)²). Squaring penalises large errors more heavily. "
            "A single prediction that is off by 10 contributes 100 to MSE, while ten predictions "
            "off by 1 each contribute only 10 total."
        ),
        why_matters=(
            "MSE is the default loss function for regression algorithms (including OLS). "
            "Its mathematical properties (smooth, differentiable) make it ideal for gradient-based "
            "optimisation. However, its squared units make it hard to interpret directly."
        ),
        example=(
            "Predictions [100, 200, 300] vs actuals [110, 180, 330]:\n"
            "Errors: [-10, 20, -30]\n"
            "Squared errors: [100, 400, 900]\n"
            "MSE = (100 + 400 + 900) / 3 = 466.7\n"
            "Note: the single error of 30 contributes 900 (64% of total MSE)."
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import mean_squared_error\n"
            "import numpy as np\n\n"
            "mse = mean_squared_error(y_test, y_pred)\n"
            "print(f'MSE: {mse:,.0f}')  # In squared units (hard to interpret)\n\n"
            "# Root MSE for interpretable units\n"
            "rmse = np.sqrt(mse)\n"
            "print(f'RMSE: £{rmse:,.0f}')  # Same units as target\n"
            "```"
        ),
        common_mistakes=[
            "Reporting MSE without units (squared units are meaningless to stakeholders)",
            "Not using RMSE when you need interpretable units",
            "Forgetting that MSE penalises large errors more than MAE",
        ],
        practice_exercise=(
            "Calculate both MSE and MAE for a regression model. If RMSE >> MAE, "
            "what does this tell you about the error distribution? "
            "What if RMSE ≈ MAE?"
        ),
        quiz=[
            QuizQuestion(
                question="Why does MSE penalise large errors more than small errors?",
                options=[
                    "Because MSE uses absolute values",
                    "Because squaring amplifies the impact of larger numbers",
                    "Because MSE divides by the square of the number of samples",
                    "It doesn't — MSE treats all errors equally",
                ],
                correct_index=1,
                explanation=(
                    "Squaring amplifies large values. 10² = 100, but 2² = 4. So a single error of 10 "
                    "contributes 100 to MSE, while five errors of 2 contribute only 20 total. "
                    "This means MSE focuses the model on reducing large errors."
                ),
            ),
        ],
        takeaways=[
            "MSE = mean squared error",
            "Penalises large errors more heavily than small errors",
            "Squared units are hard to interpret — use RMSE instead",
            "MSE is the default loss function for most regression algorithms",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_12", title="Root Mean Squared Error (RMSE)", section="regression", order=12,
        difficulty="beginner",
        objectives=[
            "Calculate RMSE",
            "Interpret RMSE in the target's units",
            "Compare RMSE with MAE to detect outlier influence",
        ],
        concept=(
            "RMSE = √MSE. It is the most commonly reported regression metric because it is in "
            "the same units as the target. RMSE ≥ MAE always; the difference indicates how much "
            "outliers influence the error."
        ),
        why_matters=(
            "RMSE combines the best of both worlds: interpretability (same units as target) and "
            "sensitivity to large errors (from squaring). It is the standard metric for reporting "
            "regression performance."
        ),
        example=(
            "If MAE = £15,000 and RMSE = £22,000 for house prices:\n"
            "• Average error is £15K\n"
            "• But the RMSE being 47% higher means some errors are much larger\n"
            "• If RMSE ≈ MAE, errors are evenly distributed\n"
            "• If RMSE >> MAE, there are a few very large errors (outliers)"
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import mean_squared_error, mean_absolute_error\n"
            "import numpy as np\n\n"
            "y_pred = model.predict(X_test)\n"
            "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n"
            "mae = mean_absolute_error(y_test, y_pred)\n\n"
            "print(f'RMSE: £{rmse:,.0f}')\n"
            "print(f'MAE:  £{mae:,.0f}')\n"
            "print(f'RMSE/MAE ratio: {rmse/mae:.2f}')\n"
            "# Ratio > 1.5 suggests significant outlier influence\n"
            "```"
        ),
        interpretation=(
            "RMSE = 0 means perfect predictions. RMSE is always ≥ MAE. A large gap between "
            "RMSE and MAE indicates a few predictions with very large errors. "
            "RMSE is sensitive to outliers, so it penalises the model for occasional bad predictions."
        ),
        common_mistakes=[
            "Reporting MSE instead of RMSE (squared units are uninterpretable)",
            "Not comparing RMSE with MAE to check outlier influence",
            "Assuming RMSE is always better than MAE",
        ],
        practice_exercise=(
            "Calculate RMSE and MAE for a Random Forest regression model. "
            "1. Is the RMSE/MAE ratio close to 1 or much larger?\n"
            "2. What does this tell you about the error distribution?\n"
            "3. Try removing the top 5% largest errors and recalculate. How much does RMSE change?"
        ),
        quiz=[
            QuizQuestion(
                question="If RMSE = £50,000 and MAE = £20,000 for house price predictions, what does the large gap indicate?",
                options=[
                    "The model is very accurate",
                    "A few predictions have very large errors",
                    "All predictions are off by about £35,000",
                    "The model is biased toward overprediction",
                ],
                correct_index=1,
                explanation=(
                    "RMSE being 2.5× larger than MAE means a few predictions have very large errors. "
                    "Squaring amplifies these large errors, pulling RMSE up. If errors were evenly "
                    "distributed, RMSE and MAE would be much closer."
                ),
            ),
        ],
        takeaways=[
            "RMSE = square root of MSE, same units as target",
            "RMSE ≥ MAE always; large gap = outlier influence",
            "Most commonly reported regression metric",
            "Use MAE when you want outlier-robust error measurement",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_13", title="R-squared (R²)", section="regression", order=13,
        difficulty="beginner",
        objectives=[
            "Calculate and interpret R²",
            "Understand explained variance",
            "Know the limitations of R²",
        ],
        concept=(
            "R² = 1 - (SS_res / SS_tot). It measures the proportion of variance in the target "
            "that the model explains. R² = 0.85 means the model explains 85% of the target's "
            "variance. R² = 0 means the model is no better than predicting the mean."
        ),
        why_matters=(
            "R² is the most intuitive measure of model fit. Unlike MSE/RMSE, it's scale-independent — "
            "R² = 0.85 means the same thing whether predicting house prices or temperatures."
        ),
        example=(
            "California Housing: Linear Regression R² = 0.60, Random Forest R² = 0.80.\n"
            "• Linear Regression explains 60% of price variation\n"
            "• Random Forest explains 80% of price variation\n"
            "• The remaining 20% is unexplained (noise, missing features, non-linearity)"
        ),
        python_example=(
            "```python\n"
            "from sklearn.metrics import r2_score\n"
            "import numpy as np\n\n"
            "r2 = r2_score(y_test, y_pred)\n"
            "print(f'R²: {r2:.3f}')  # 0.85 = explains 85% of variance\n\n"
            "# Adjusted R² (penalises for extra features)\n"
            "n = len(y_test)\n"
            "p = X_test.shape[1]\n"
            "adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)\n"
            "print(f'Adjusted R²: {adj_r2:.3f}')\n"
            "```"
        ),
        interpretation=(
            "R² = 1.0 → perfect prediction. R² = 0 → model is no better than predicting the mean. "
            "R² < 0 → model is worse than predicting the mean (something is very wrong). "
            "R² = 0.85 → the model captures 85% of the variation in the target."
        ),
        common_mistakes=[
            "High R² doesn't mean the model is correct (could overfit or miss non-linearity)",
            "R² always increases with more features (use adjusted R² to penalise)",
            "R² doesn't indicate whether model assumptions are met",
            "Comparing R² across datasets with different target variance",
        ],
        practice_exercise=(
            "Fit Linear Regression and Random Forest on California Housing. "
            "Calculate R² for both on the test set. Which is higher? "
            "Now calculate adjusted R². Does the ranking change?"
        ),
        quiz=[
            QuizQuestion(
                question="R² = -0.15. What does this mean?",
                options=[
                    "The model explains 15% of the variance",
                    "The model is worse than simply predicting the mean",
                    "There is a 15% error rate",
                    "The model is overfitting by 15%",
                ],
                correct_index=1,
                explanation=(
                    "Negative R² means the model performs worse than a model that simply predicts "
                    "the mean of the target for every prediction. This usually indicates a very "
                    "poor model, inappropriate model choice, or data issues."
                ),
            ),
        ],
        takeaways=[
            "R² = proportion of variance in the target explained by the model",
            "Higher is better, but check for overfitting",
            "Use adjusted R² when comparing models with different numbers of features",
            "Negative R² means the model is worse than predicting the mean",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_14", title="Ridge Regression (L2)", section="regression", order=14,
        difficulty="intermediate",
        objectives=[
            "Understand L2 regularisation",
            "Apply Ridge regression and tune alpha",
            "Know when Ridge helps over ordinary linear regression",
        ],
        concept=(
            "Ridge adds an L2 penalty to the loss function: Loss = OLS + α × Σβᵢ². "
            "This shrinks all coefficients toward zero (but never exactly zero). "
            "It handles multicollinearity and prevents overfitting by constraining coefficient size."
        ),
        why_matters=(
            "When features are correlated or there are many features, OLS produces unstable "
            "coefficients. Ridge stabilises them by penalising large coefficients. It's the "
            "first thing to try when linear regression overfits."
        ),
        example=(
            "With 100 features and only 500 samples, OLS overfits (R²_train = 0.99, R²_test = 0.45). "
            "Ridge with α = 10 shrinks coefficients: R²_train = 0.82, R²_test = 0.78. "
            "The gap shrinks because the model generalises better."
        ),
        python_example=(
            "```python\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n\n"
            "# Ridge requires scaled features\n"
            "pipeline = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('model', Ridge(alpha=1.0))\n"
            "])\n"
            "pipeline.fit(X_train, y_train)\n"
            "print(f'R²: {pipeline.score(X_test, y_test):.3f}')\n\n"
            "# Tune alpha with cross-validation\n"
            "from sklearn.model_selection import GridSearchCV\n"
            "param_grid = {'model__alpha': [0.01, 0.1, 1, 10, 100]}\n"
            "grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')\n"
            "grid.fit(X_train, y_train)\n"
            "print(f'Best alpha: {grid.best_params_[\"model__alpha\"]}')\n"
            "```"
        ),
        common_mistakes=[
            "Using very large alpha (over-regularisation leads to underfitting)",
            "Not scaling features before Ridge (penalty operates on coefficient scale)",
            "Using Ridge when you need feature selection (use Lasso instead)",
        ],
        practice_exercise=(
            "Fit Ridge regression on California Housing with alpha values [0.01, 0.1, 1, 10, 100]. "
            "Plot R² vs alpha. What is the optimal alpha? "
            "How does it compare to ordinary linear regression?"
        ),
        quiz=[
            QuizQuestion(
                question="What happens to Ridge coefficients as alpha increases to infinity?",
                options=[
                    "They all become exactly zero",
                    "They all shrink toward zero but never reach it",
                    "They increase in magnitude",
                    "The model switches to Lasso",
                ],
                correct_index=1,
                explanation=(
                    "Ridge shrinks coefficients toward zero as alpha increases, but they never "
                    "reach exactly zero. At infinite alpha, they approach zero asymptotically. "
                    "Only Lasso can drive coefficients exactly to zero."
                ),
            ),
        ],
        takeaways=[
            "Ridge shrinks coefficients (never to exactly zero)",
            "Handles multicollinearity and prevents overfitting",
            "Requires scaled features (use Pipeline with StandardScaler)",
            "Tune alpha with cross-validation",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_15", title="Lasso Regression (L1)", section="regression", order=15,
        difficulty="intermediate",
        objectives=[
            "Understand L1 regularisation",
            "Apply Lasso regression for feature selection",
            "Tune alpha to control sparsity",
        ],
        concept=(
            "Lasso adds an L1 penalty: Loss = OLS + α × Σ|βᵢ|. Unlike Ridge, Lasso drives "
            "some coefficients exactly to zero, performing automatic feature selection. "
            "This produces sparse, interpretable models."
        ),
        why_matters=(
            "Lasso identifies the most important features by zeroing out irrelevant ones. "
            "With 100 features, Lasso might select only 15, making the model simpler and "
            "easier to understand."
        ),
        example=(
            "A housing dataset with 50 features: Lasso with α = 0.1 sets 35 coefficients "
            "to exactly zero, keeping only 15. The remaining features (size, location, age) "
            "are the true drivers. This is more interpretable than Ridge which keeps all 50."
        ),
        python_example=(
            "```python\n"
            "from sklearn.linear_model import Lasso\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n\n"
            "pipeline = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('model', Lasso(alpha=0.1))\n"
            "])\n"
            "pipeline.fit(X_train, y_train)\n\n"
            "# Check which features were selected\n"
            "coefs = pipeline.named_steps['model'].coef_\n"
            "selected = [n for n, c in zip(feature_names, coefs) if c != 0]\n"
            "print(f'Selected {len(selected)}/{len(feature_names)} features:')\n"
            "for name, coef in zip(selected, coefs[coefs != 0]):\n"
            "    print(f'  {name}: {coef:.4f}')\n"
            "```"
        ),
        common_mistakes=[
            "Using too large alpha (removes all features → model predicts the mean)",
            "Not scaling before Lasso (penalty is on coefficient scale)",
            "Assuming Lasso always selects the 'best' features (can be unstable with correlated features)",
        ],
        practice_exercise=(
            "Fit Lasso on California Housing with alpha values [0.001, 0.01, 0.1, 1.0]. "
            "For each alpha, count how many features have non-zero coefficients. "
            "Plot features selected vs alpha. What is the trade-off?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the key difference between Ridge and Lasso?",
                options=[
                    "Ridge is for classification, Lasso is for regression",
                    "Ridge shrinks coefficients toward zero; Lasso drives them exactly to zero",
                    "Ridge requires scaling; Lasso does not",
                    "Ridge is faster to train",
                ],
                correct_index=1,
                explanation=(
                    "Ridge (L2) shrinks all coefficients toward zero but never reaches exactly zero. "
                    "Lasso (L1) can drive coefficients exactly to zero, effectively removing features. "
                    "This makes Lasso useful for automatic feature selection."
                ),
            ),
        ],
        takeaways=[
            "Lasso drives some coefficients exactly to zero (feature selection)",
            "Creates sparse, interpretable models",
            "Tune alpha to control how many features are eliminated",
            "Use Lasso when you suspect many features are irrelevant",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_16", title="Polynomial Regression", section="regression", order=16,
        difficulty="intermediate",
        objectives=[
            "Fit polynomial curves to non-linear data",
            "Avoid overfitting with appropriate degree selection",
            "Combine polynomial features with regularisation",
        ],
        concept=(
            "Polynomial regression fits a polynomial: y = β₀ + β₁x + β₂x² + ... + βₖxᵏ. "
            "It captures non-linear relationships while remaining within the linear regression "
            "framework. The model is still 'linear in coefficients' — only the features are "
            "non-linear."
        ),
        why_matters=(
            "Many real relationships are non-linear: drug dosage peaks in effectiveness then "
            "declines, temperature effects follow curves, and growth rates slow down. Polynomial "
            "regression fits these patterns without switching to a completely different algorithm."
        ),
        example=(
            "Temperature vs. ice cream sales: sales increase with temperature up to 30°C, "
            "then decrease as people stay indoors. A linear model would miss this peak. "
            "A degree-2 polynomial captures the inverted-U shape."
        ),
        python_example=(
            "```python\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.pipeline import Pipeline\n\n"
            "pipeline = Pipeline([\n"
            "    ('poly', PolynomialFeatures(degree=2, include_bias=False)),\n"
            "    ('model', LinearRegression())\n"
            "])\n"
            "pipeline.fit(X_train, y_train)\n"
            "print(f'R²: {pipeline.score(X_test, y_test):.3f}')\n\n"
            "# Better: combine with Ridge to prevent overfitting\n"
            "from sklearn.linear_model import Ridge\n"
            "pipeline_ridge = Pipeline([\n"
            "    ('poly', PolynomialFeatures(degree=2)),\n"
            "    ('model', Ridge(alpha=1.0))\n"
            "])\n"
            "```"
        ),
        common_mistakes=[
            "Using high degree (3+) without regularisation — massive overfitting",
            "Not regularising polynomial models (use Ridge + PolynomialFeatures)",
            "Extrapolating beyond training range (polynomials diverge wildly outside training data)",
            "Creating too many features with high-degree polynomials on many features",
        ],
        practice_exercise=(
            "Create a synthetic dataset with a quadratic relationship (y = x² + noise). "
            "Fit degree-1 and degree-2 polynomial regression. "
            "1. Which fits better on training data? On test data?\n"
            "2. Now try degree-5. What happens?"
        ),
        quiz=[
            QuizQuestion(
                question="Why should you combine polynomial features with Ridge regularisation?",
                options=[
                    "Polynomial features increase model speed",
                    "High-degree polynomials create many features that can overfit without regularisation",
                    "Ridge automatically selects the polynomial degree",
                    "Polynomial features require scaled data which Ridge provides",
                ],
                correct_index=1,
                explanation=(
                    "Polynomial features (especially degree 3+) create many new features that "
                    "can cause overfitting. Ridge regularisation constrains the coefficient sizes, "
                    "preventing the model from fitting noise in the polynomial features."
                ),
            ),
        ],
        takeaways=[
            "Polynomial regression fits non-linear relationships",
            "Keep degree ≤ 2 for most practical cases",
            "Always combine with Ridge regularisation to prevent overfitting",
            "Never extrapolate polynomial models beyond the training data range",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_17", title="Decision Tree Regression", section="regression", order=17,
        difficulty="intermediate",
        objectives=[
            "Apply Decision Tree Regressor",
            "Understand tree-based prediction (step function)",
            "Control complexity with max_depth",
        ],
        concept=(
            "Decision tree regression predicts the mean target value of training samples in "
            "each leaf node. The tree splits features to minimise MSE within each node, "
            "creating a step-function approximation of the true relationship."
        ),
        why_matters=(
            "Decision trees naturally capture non-linear relationships and interactions without "
            "feature engineering. They handle mixed data types, don't need scaling, and provide "
            "feature importance scores."
        ),
        example=(
            "A decision tree for house prices might split first on 'size > 1500 sqft', "
            "then on 'location == central'. The prediction for each leaf is the average "
            "price of training houses in that leaf."
        ),
        python_example=(
            "```python\n"
            "from sklearn.tree import DecisionTreeRegressor\n\n"
            "model = DecisionTreeRegressor(max_depth=5, random_state=42)\n"
            "model.fit(X_train, y_train)\n"
            "print(f'R²: {model.score(X_test, y_test):.3f}')\n\n"
            "# Feature importance\n"
            "import pandas as pd\n"
            "importance = pd.Series(\n"
            "    model.feature_importances_, index=feature_names\n"
            ").sort_values(ascending=False)\n"
            "importance.head(10).plot(kind='barh')\n"
            "```"
        ),
        common_mistakes=[
            "Not limiting max_depth — unconstrained trees massively overfit",
            "Using decision trees for extrapolation — they can't predict beyond training range",
            "Trusting feature importance on small datasets",
        ],
        practice_exercise=(
            "Fit a Decision Tree Regressor on California Housing with max_depth "
            "in [2, 5, 10, None]. Plot R² vs max_depth. "
            "1. At what depth does R² stop improving?\n"
            "2. What happens with unlimited depth?"
        ),
        quiz=[
            QuizQuestion(
                question="Why can't decision trees extrapolate beyond training data?",
                options=[
                    "They don't have enough parameters",
                    "Each leaf predicts the mean of training samples in that region",
                    "They only work with categorical features",
                    "Trees are not mathematical models",
                ],
                correct_index=1,
                explanation=(
                    "A decision tree predicts the average of training samples in each leaf. "
                    "For a new point outside the training range, it falls into the nearest leaf, "
                    "whose prediction is still the mean of the training data in that region — "
                    "it can never predict beyond what it has seen."
                ),
            ),
        ],
        takeaways=[
            "Decision trees capture non-linear patterns naturally",
            "Always limit max_depth to prevent overfitting",
            "Trees create step-function predictions, not smooth curves",
            "Cannot extrapolate beyond training data range",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_18", title="Random Forest Regression", section="regression", order=18,
        difficulty="intermediate",
        objectives=[
            "Apply Random Forest Regressor",
            "Understand ensemble benefits (variance reduction)",
            "Extract and interpret feature importance",
        ],
        concept=(
            "Random Forest averages predictions from many decorrelated trees. Each tree sees "
            "a random subset of data (bootstrap) and features, reducing variance while "
            "maintaining predictive power. The ensemble effect makes the model more robust."
        ),
        why_matters=(
            "Random Forest is the most reliable default regressor for tabular data. It rarely "
            "overfits, handles mixed data types, provides feature importance, and usually "
            "outperforms single decision trees by a large margin."
        ),
        example=(
            "Training 100 trees: each tree sees a random 80% of data and considers only √n "
            "features at each split. This diversity means individual tree errors cancel out. "
            "If one tree overpredicts and another underpredicts, the average is closer to the truth."
        ),
        python_example=(
            "```python\n"
            "from sklearn.ensemble import RandomForestRegressor\n\n"
            "model = RandomForestRegressor(\n"
            "    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1\n"
            ")\n"
            "model.fit(X_train, y_train)\n"
            "print(f'R²: {model.score(X_test, y_test):.3f}')\n\n"
            "# Out-of-bag score (free validation, no CV needed)\n"
            "model_oob = RandomForestRegressor(n_estimators=100, oob_score=True)\n"
            "model_oob.fit(X_train, y_train)\n"
            "print(f'OOB R²: {model_oob.oob_score_:.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Not tuning max_depth (default unlimited can overfit on small datasets)",
            "Using too few trees (< 50) — the ensemble effect needs diversity",
            "Ignoring that trees still can't extrapolate beyond training data",
        ],
        practice_exercise=(
            "Train a Random Forest on California Housing with n_estimators in [10, 50, 100, 200]. "
            "1. How does R² change?\n"
            "2. What is the OOB score?\n"
            "3. Which features are most important?"
        ),
        quiz=[
            QuizQuestion(
                question="Why does Random Forest use random subsets of features at each split?",
                options=[
                    "To reduce training time",
                    "To make trees decorrelated so the ensemble reduces variance",
                    "To prevent data leakage",
                    "It's a default setting that doesn't matter",
                ],
                correct_index=1,
                explanation=(
                    "If all trees could use all features, the strongest feature would dominate "
                    "every split, making trees similar. Random feature subsets force diversity, "
                    "so different trees capture different patterns. This decorrelation is what "
                    "makes the ensemble effective."
                ),
            ),
        ],
        takeaways=[
            "Ensemble of random trees → robust, accurate predictions",
            "OOB score provides a free validation estimate",
            "Good default for tabular regression tasks",
            "Still cannot extrapolate beyond training data",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_19", title="Gradient Boosting Regression", section="regression", order=19,
        difficulty="intermediate",
        objectives=[
            "Apply Gradient Boosting Regressor",
            "Tune learning_rate and n_estimators",
            "Understand sequential error correction",
        ],
        concept=(
            "Gradient Boosting builds trees sequentially. Each new tree predicts the residuals "
            "(errors) of the current ensemble, then adds its prediction (scaled by learning_rate) "
            "to the ensemble. It's an additive model that gradually reduces bias."
        ),
        why_matters=(
            "Gradient Boosting typically achieves the highest accuracy on tabular regression tasks. "
            "It's the top choice for competitions and production systems. Understanding the "
            "learning_rate × n_estimators trade-off is essential."
        ),
        example=(
            "Tree 1 predicts average price → residuals are actual - predicted.\n"
            "Tree 2 fits those residuals → adds 10% (learning_rate=0.1) of its prediction.\n"
            "Tree 3 fits the new residuals → adds 10% more.\n"
            "After 200 trees, the ensemble gradually converges on the true relationship."
        ),
        python_example=(
            "```python\n"
            "from sklearn.ensemble import GradientBoostingRegressor\n\n"
            "model = GradientBoostingRegressor(\n"
            "    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42\n"
            ")\n"
            "model.fit(X_train, y_train)\n"
            "print(f'R²: {model.score(X_test, y_test):.3f}')\n\n"
            "# Monitor staged predictions\n"
            "import numpy as np\n"
            "from sklearn.metrics import r2_score\n"
            "train_r2 = [r2_score(y_train, yp) for yp in model.staged_predict(X_train)]\n"
            "test_r2 = [r2_score(y_test, yp) for yp in model.staged_predict(X_test)]\n"
            "```"
        ),
        common_mistakes=[
            "Using too many trees with high learning_rate (overfitting)",
            "Setting max_depth too high (each tree should be simple: 3-5)",
            "Not using early stopping to find the optimal number of trees",
        ],
        practice_exercise=(
            "Train Gradient Boosting on California Housing. Compare:\n"
            "1. learning_rate=0.1, n_estimators=200\n"
            "2. learning_rate=0.01, n_estimators=2000\n"
            "Which gives better R²? Which trains faster?"
        ),
        quiz=[
            QuizQuestion(
                question="What is the relationship between learning_rate and n_estimators in Gradient Boosting?",
                options=[
                    "They are independent parameters",
                    "Lower learning_rate needs more trees but produces a better model",
                    "Higher learning_rate always gives better results",
                    "n_estimators should always equal 100",
                ],
                correct_index=1,
                explanation=(
                    "A lower learning_rate means each tree contributes less, so you need more trees "
                    "to reach the same performance. But the slower learning typically produces a "
                    "more robust model. Typical trade-off: lr=0.1 with 200 trees vs lr=0.01 with 2000 trees."
                ),
            ),
        ],
        takeaways=[
            "Sequential ensemble — each tree corrects previous errors",
            "Often achieves highest accuracy on tabular data",
            "learning_rate × n_estimators is the key trade-off",
            "Keep max_depth low (3-5) — each tree should be simple",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_20", title="KNN Regression", section="regression", order=20,
        difficulty="intermediate",
        objectives=[
            "Apply KNN Regressor",
            "Choose an appropriate value for k",
            "Understand KNN limitations (scaling, dimensionality)",
        ],
        concept=(
            "KNN regression predicts the average of the k nearest training examples. "
            "It's a non-parametric method — it makes no assumptions about the data distribution. "
            "The 'distance' between points determines which neighbours are considered."
        ),
        why_matters=(
            "KNN regression is intuitive and useful as a baseline. It naturally captures local "
            "patterns but struggles with high dimensions and large datasets."
        ),
        example=(
            "To predict a house price: find the 5 most similar houses in the training data "
            "(similar size, location, bedrooms). Average their prices. That's the prediction. "
            "k=1 → use the single most similar house. k=100 → average 100 similar houses."
        ),
        python_example=(
            "```python\n"
            "from sklearn.neighbors import KNeighborsRegressor\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n\n"
            "# KNN REQUIRES scaling (it's distance-based!)\n"
            "pipeline = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('model', KNeighborsRegressor(n_neighbors=5))\n"
            "])\n"
            "pipeline.fit(X_train, y_train)\n"
            "print(f'R²: {pipeline.score(X_test, y_test):.3f}')\n"
            "```"
        ),
        common_mistakes=[
            "Not scaling features — distance-based methods are sensitive to feature scales",
            "Using too small k (overfitting) or too large k (underfitting)",
            "Using KNN on high-dimensional data (distance becomes meaningless — curse of dimensionality)",
        ],
        practice_exercise=(
            "Train KNN regression on California Housing with k in [1, 3, 5, 10, 50, 100]. "
            "1. Plot R² vs k. What is the optimal k?\n"
            "2. Try without scaling. How much does performance drop?"
        ),
        quiz=[
            QuizQuestion(
                question="Why must you scale features before using KNN?",
                options=[
                    "KNN uses Euclidean distance, which is dominated by features with larger scales",
                    "Scaling makes the model train faster",
                    "KNN requires integer inputs",
                    "Scaling prevents overfitting",
                ],
                correct_index=0,
                explanation=(
                    "KNN computes distances between points. If one feature ranges 0-1000 and "
                    "another ranges 0-1, the first feature will dominate the distance calculation. "
                    "Scaling ensures all features contribute equally to the distance metric."
                ),
            ),
        ],
        takeaways=[
            "KNN regression = average of k nearest training examples",
            "Requires feature scaling (distance-based method)",
            "Good baseline, but slow on large and high-dimensional data",
            "Choose k with cross-validation (small k = overfit, large k = underfit)",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_21", title="Underfitting", section="regression", order=21,
        difficulty="intermediate",
        objectives=[
            "Identify underfitting from training and test scores",
            "Understand causes of underfitting",
            "Apply remedies to reduce underfitting",
        ],
        concept=(
            "Underfitting occurs when a model is too simple to capture the underlying pattern. "
            "Signs: low training score AND low test score. The model has high bias — it makes "
            "strong, incorrect assumptions about the data."
        ),
        why_matters=(
            "An underfitted model wastes the information in your data. It performs poorly "
            "everywhere — training and test sets alike. No amount of data will help if the "
            "model is fundamentally too simple."
        ),
        example=(
            "Fitting a straight line to clearly curved data: the line misses the pattern "
            "everywhere. R²_train = 0.30, R²_test = 0.28. Both are low because the model "
            "can't capture the non-linear relationship."
        ),
        python_example=(
            "```python\n"
            "# Detect underfitting\n"
            "train_r2 = model.score(X_train, y_train)\n"
            "test_r2 = model.score(X_test, y_test)\n"
            "print(f'Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}')\n"
            "# If both are low → underfitting\n\n"
            "# Remedies:\n"
            "# 1. Add polynomial/interaction features\n"
            "# 2. Use a more complex model\n"
            "# 3. Reduce regularisation (smaller alpha in Ridge/Lasso)\n"
            "# 4. Add more relevant features\n"
            "```"
        ),
        common_mistakes=[
            "Adding more data when the model is too simple (won't help)",
            "Over-regularising (very large alpha in Ridge/Lasso)",
            "Using linear models for clearly non-linear data",
            "Adding more features without checking their relevance",
        ],
        practice_exercise=(
            "Fit a linear regression on data with a quadratic relationship. "
            "1. What are the train and test R²?\n"
            "2. Now add a polynomial feature (degree 2). Does R² improve?\n"
            "3. What does this tell you about model complexity?"
        ),
        quiz=[
            QuizQuestion(
                question="Train R² = 0.35, Test R² = 0.33. What is the problem?",
                options=[
                    "Overfitting — the model memorised training data",
                    "Underfitting — the model is too simple for the data",
                    "Data leakage — test data leaked into training",
                    "The model is perfectly calibrated",
                ],
                correct_index=1,
                explanation=(
                    "Both training and test R² are low and close together. This means the model "
                    "can't capture the pattern even in training data — it's too simple. Overfitting "
                    "would show high train, low test."
                ),
            ),
        ],
        takeaways=[
            "Underfitting: low train score AND low test score",
            "Cause: model too simple or too much regularisation",
            "Remedies: more complex model, fewer features removed, add polynomial features",
            "Adding more data won't fix underfitting",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_22", title="Regression Overfitting", section="regression", order=22,
        difficulty="intermediate",
        objectives=[
            "Identify overfitting in regression (high train, low test)",
            "Apply remedies (simpler model, regularisation, more data)",
            "Use learning curves to diagnose",
        ],
        concept=(
            "Overfitting in regression: high training score but low test score. The model "
            "memorises noise instead of learning the signal. Common with complex models on "
            "small datasets — the model fits training data perfectly but fails on new data."
        ),
        why_matters=(
            "Overfitted models are useless in production. They look perfect during development "
            "but fail when deployed. Detecting and preventing overfitting is the most critical "
            "skill in machine learning."
        ),
        example=(
            "A degree-10 polynomial on 50 data points: R²_train = 0.99, R²_test = -0.50. "
            "The model fits every training point (including noise) but the test predictions "
            "are wildly wrong. The polynomial has memorised the training data."
        ),
        python_example=(
            "```python\n"
            "from sklearn.model_selection import learning_curve\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "train_sizes, train_scores, val_scores = learning_curve(\n"
            "    model, X_train, y_train, cv=5, n_jobs=-1,\n"
            "    train_sizes=np.linspace(0.1, 1.0, 10)\n"
            ")\n\n"
            "plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')\n"
            "plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')\n"
            "plt.xlabel('Training Set Size')\n"
            "plt.ylabel('R²')\n"
            "plt.legend()\n"
            "plt.title('Learning Curve')\n"
            "plt.show()\n"
            "```"
        ),
        interpretation=(
            "Training curve much higher than validation = overfitting. Both curves converging "
            "= good fit. Both curves low = underfitting. A large gap between curves means "
            "the model needs more data or less complexity."
        ),
        common_mistakes=[
            "Only looking at training scores",
            "Not using learning curves to diagnose the problem",
            "Overcomplicating the model to improve training score",
        ],
        practice_exercise=(
            "Train a Decision Tree with max_depth in [2, 5, 10, 20, None]. "
            "Plot train and test R² vs max_depth. "
            "1. At what depth does overfitting start?\n"
            "2. What is the optimal max_depth?"
        ),
        quiz=[
            QuizQuestion(
                question="Train R² = 0.98, Test R² = 0.45. What should you do?",
                options=[
                    "Add more features to improve the model",
                    "Reduce model complexity, add regularisation, or get more data",
                    "The model is already good — 0.98 training R² is excellent",
                    "Switch to a more complex model",
                ],
                correct_index=1,
                explanation=(
                    "The large gap between training and test R² indicates severe overfitting. "
                    "The model memorises training data. Solutions: simplify the model (lower max_depth), "
                    "add regularisation, or collect more training data."
                ),
            ),
        ],
        takeaways=[
            "Overfitting: high train score, low test score",
            "Use learning curves to diagnose",
            "Remedies: simpler model, regularisation, more data, cross-validation",
            "A model that scores 0.98 on training but 0.45 on test is useless",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_23", title="Regression Assumptions", section="regression", order=23,
        difficulty="advanced",
        objectives=[
            "List the five assumptions of linear regression",
            "Use diagnostic plots to check each assumption",
            "Understand consequences of violated assumptions",
        ],
        concept=(
            "Linear regression assumes: (1) Linearity — relationship between features and target "
            "is linear, (2) Independence — residuals are independent, (3) Homoscedasticity — "
            "residual variance is constant, (4) Normality — residuals are normally distributed, "
            "(5) No multicollinearity — features are not highly correlated."
        ),
        why_matters=(
            "Violated assumptions make coefficient estimates unreliable, confidence intervals "
            "wrong, and p-values misleading. This matters for interpretation and inference, "
            "not just prediction. Tree-based models don't need these assumptions."
        ),
        simple_explanation=(
            "OLS assumes a specific 'shape' for the data. If the data doesn't match that shape, "
            "the mathematical guarantees (unbiased coefficients, valid p-values) break down."
        ),
        common_mistakes=[
            "Ignoring assumption violations because R² looks good",
            "Not checking residual plots for each assumption",
            "Confusing prediction performance with assumption validity",
        ],
        practice_exercise=(
            "Fit a linear regression on California Housing and check all five assumptions:\n"
            "1. Plot residuals vs predicted (linearity, homoscedasticity)\n"
            "2. Q-Q plot (normality)\n"
            "3. Compute VIF for multicollinearity\n"
            "Which assumptions are violated?"
        ),
        quiz=[
            QuizQuestion(
                question="Which models do NOT require the five linear regression assumptions?",
                options=[
                    "Ridge and Lasso regression",
                    "Simple and multiple linear regression",
                    "Decision Trees and Random Forests",
                    "All regression models require these assumptions",
                ],
                correct_index=2,
                explanation=(
                    "Tree-based models (Decision Trees, Random Forests, Gradient Boosting) make "
                    "no assumptions about linearity, normality, or homoscedasticity. They partition "
                    "the feature space and average targets in each partition."
                ),
            ),
        ],
        takeaways=[
            "Five assumptions: linearity, independence, homoscedasticity, normality, no multicollinearity",
            "Check with residual plots, Q-Q plot, and VIF",
            "Violated assumptions affect inference (coefficients, p-values), not just prediction",
            "Tree-based models don't need these assumptions",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_24", title="Residual Analysis", section="regression", order=24,
        difficulty="advanced",
        objectives=[
            "Perform complete residual analysis with four diagnostic plots",
            "Identify non-linearity, heteroscedasticity, and outliers",
            "Use Q-Q plots and scale-location plots",
        ],
        concept=(
            "Residual analysis checks four things: (1) Residuals vs Predicted: random scatter = "
            "linearity OK. (2) Constant spread = homoscedasticity OK. (3) Normal distribution in "
            "Q-Q plot = normality OK. (4) No autocorrelation = independence OK."
        ),
        why_matters=(
            "Residual analysis is the most powerful diagnostic tool. It reveals problems that "
            "metrics like R² completely miss. A model with R² = 0.85 might have severe "
            "heteroscedasticity making its predictions unreliable for extreme values."
        ),
        example=(
            "A funnel shape in the residual plot means the model's error increases with the "
            "predicted value. The model is less reliable for high-value predictions. "
            "Solution: log-transform the target or use weighted regression."
        ),
        python_example=(
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "from scipy import stats\n"
            "import numpy as np\n\n"
            "residuals = y_test - model.predict(X_test)\n"
            "y_pred = model.predict(X_test)\n\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n\n"
            "# 1. Residuals vs Predicted (linearity + homoscedasticity)\n"
            "axes[0, 0].scatter(y_pred, residuals, alpha=0.5)\n"
            "axes[0, 0].axhline(0, color='r', linestyle='--')\n"
            "axes[0, 0].set_title('Residuals vs Predicted')\n\n"
            "# 2. Histogram (normality)\n"
            "axes[0, 1].hist(residuals, bins=30, edgecolor='black')\n"
            "axes[0, 1].set_title('Residual Distribution')\n\n"
            "# 3. Q-Q Plot (normality)\n"
            "stats.probplot(residuals, dist='norm', plot=axes[1, 0])\n"
            "axes[1, 0].set_title('Q-Q Plot')\n\n"
            "# 4. Scale-Location (homoscedasticity)\n"
            "axes[1, 1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)\n"
            "axes[1, 1].set_title('Scale-Location Plot')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "```"
        ),
        common_mistakes=[
            "Skipping residual analysis because aggregate metrics look good",
            "Ignoring funnel shapes (heteroscedasticity)",
            "Not checking Q-Q plot for departures from normality in the tails",
        ],
        practice_exercise=(
            "Perform complete residual analysis on a regression model. For each of the four plots, "
            "state whether the assumption is satisfied and explain what you see."
        ),
        quiz=[
            QuizQuestion(
                question="A Q-Q plot shows points deviating from the diagonal in the upper tail. What does this mean?",
                options=[
                    "Residuals are perfectly normally distributed",
                    "The residual distribution has heavier tails than a normal distribution",
                    "The model is overfitting",
                    "There is heteroscedasticity",
                ],
                correct_index=1,
                explanation=(
                    "Points above the diagonal in the upper tail of a Q-Q plot mean the residuals "
                    "have more extreme positive values than expected under normality. This indicates "
                    "heavy tails, possibly due to outliers."
                ),
            ),
        ],
        takeaways=[
            "Residuals vs Predicted: check linearity and homoscedasticity",
            "Q-Q Plot: check normality of residuals",
            "Scale-Location: check constant variance across predictions",
            "Always use all four plots for comprehensive diagnosis",
        ],
        lab_module="regression",
    ),
    Topic(
        id="reg_25", title="Regression Case Study", section="regression", order=25,
        difficulty="advanced",
        objectives=[
            "Apply the complete regression workflow end-to-end",
            "Compare multiple algorithms systematically",
            "Document model selection decisions",
        ],
        concept=(
            "A complete regression project follows a systematic workflow: EDA → preprocessing → "
            "feature engineering → baseline model → compare algorithms → select best model → "
            "evaluate thoroughly → document. This is the process used in real data science projects."
        ),
        why_matters=(
            "Real regression projects require the full workflow. Understanding how all pieces "
            "fit together — from data loading to model deployment — is essential for independent "
            "work. This case study ties together everything you've learned."
        ),
        example=(
            "California Housing prediction workflow:\n"
            "1. EDA: Median income is most correlated with price (r=0.69). Geography matters.\n"
            "2. Preprocessing: Scale numerical features, handle any missing values.\n"
            "3. Models: Linear R²=0.60, Ridge R²=0.62, RF R²=0.80, GB R²=0.83.\n"
            "4. Winner: Gradient Boosting (R²=0.83, RMSE=£47K).\n"
            "5. Feature importance: median_income > latitude > longitude > housing_age."
        ),
        common_mistakes=[
            "Not comparing with a baseline (Linear Regression)",
            "Not documenting why you chose a particular model",
            "Forgetting to save the pipeline for production deployment",
        ],
        practice_exercise=(
            "Complete a regression case study using the California Housing dataset:\n"
            "1. EDA: identify key patterns\n"
            "2. Preprocess: handle missing values, scale features\n"
            "3. Train: Linear, Ridge, Random Forest, Gradient Boosting\n"
            "4. Evaluate: compare R², MAE, RMSE\n"
            "5. Select the best model and explain why"
        ),
        quiz=[
            QuizQuestion(
                question="In a regression case study, why should you always start with a simple baseline model?",
                options=[
                    "It's always the best model",
                    "It sets a performance benchmark — complex models must beat it to justify their complexity",
                    "Simple models are required by law",
                    "It takes the least time to train",
                ],
                correct_index=1,
                explanation=(
                    "A baseline (like Linear Regression or predicting the mean) establishes a "
                    "minimum performance level. If a complex model doesn't significantly beat "
                    "the baseline, its added complexity isn't justified. Simple models are also "
                    "more interpretable and faster."
                ),
            ),
        ],
        takeaways=[
            "Follow the complete workflow: EDA → preprocess → engineer → train → evaluate → compare",
            "Always start with a baseline and compare against it",
            "Document every decision: why you chose this model, this metric, this parameter",
            "Save the best pipeline (model + preprocessing) for deployment",
        ],
        lab_module="regression",
    ),
]
