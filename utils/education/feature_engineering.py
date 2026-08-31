"""
Feature Engineering — learning topics for the Feature Engineering module.
"""
from utils.education.base import T

TOPICS = {
    "what_is_feature_engineering": T(
        title="What Is Feature Engineering",
        module="feature_engineering",
        what=(
            "Feature engineering is the process of creating, transforming, "
            "and selecting input variables (features) to improve model "
            "performance. It bridges raw data and effective modelling."
        ),
        why=(
            "Better features often matter more than a better algorithm. "
            "A well-engineered feature can make a simple model outperform "
            "a complex one. Feature engineering is where domain expertise "
            "meets data science."
        ),
        when=(
            "After preprocessing, before modelling. It's an iterative "
            "process — you engineer features, test the model, and "
            "refine."
        ),
        example=(
            "From raw 'Date' column, create: Year, Month, Day, "
            "DayOfWeek, IsWeekend. These new features capture temporal "
            "patterns the raw date cannot."
        ),
        mistakes=[
            "Creating features from test data — always engineer on train only.",
            "Adding features without checking if they improve the model.",
            "Not removing redundant features after engineering.",
        ],
        interpretation=(
            "Good features are: informative (correlated with target), "
            "independent (not correlated with each other), and "
            "simple (easy to understand and maintain)."
        ),
        think_about_it="You have 'temperature' in Fahrenheit. Should you convert to Celsius before modelling? Why or why not?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "# Date features\n"
            "df['year'] = df['date'].dt.year\n"
            "df['month'] = df['date'].dt.month\n"
            "df['day_of_week'] = df['date'].dt.dayofweek\n"
            "\n"
            "# Mathematical transformation\n"
            "df['log_income'] = np.log1p(df['income'])\n"
            "```"
        ),
        keywords=["feature engineering", "creation", "transformation", "selection"],
    ),

    "why_features_matter": T(
        title="Why Features Matter",
        module="feature_engineering",
        what=(
            "Features determine what the model can learn. With poor "
            "features, even the best algorithm fails. With great features, "
            "simple algorithms can excel."
        ),
        why=(
            "The quality of your features directly determines model "
            "performance. Feature engineering is often the highest-ROI "
            "activity in a data science project."
        ),
        when=(
            "When model performance plateaus. Before trying more complex "
            "algorithms, try engineering better features."
        ),
        example="A model predicting house prices: 'number of rooms' is a good feature. 'Number of rooms / square footage' (room density) might be even better.",
        mistakes=[
            "Spending all time tuning hyperparameters instead of improving features.",
            "Ignoring domain expertise when creating features.",
            "Not evaluating whether new features actually help.",
        ],
        interpretation=(
            "If a model has high training error, the features may not "
            "contain enough information. If test error is high but "
            "training is low, the features may be too specific."
        ),
        think_about_it="If you could only add one new feature to predict house prices, what would it be and why?",
        code_link=(
            "```python\n"
            "# Feature importance after training\n"
            "import pandas as pd\n"
            "imp = pd.Series(model.feature_importances_, index=feature_names)\n"
            "imp.sort_values(ascending=False).head(10).plot(kind='bar')\n"
            "```"
        ),
        keywords=["features", "importance", "quality", "performance", "engineering"],
    ),

    "feature_creation": T(
        title="Feature Creation",
        module="feature_engineering",
        what=(
            "Feature creation means making new columns from existing "
            "data. This includes combining columns, extracting "
            "information, and deriving domain-specific features."
        ),
        why=(
            "Raw data rarely has the exact features needed for good "
            "predictions. Creation adds information that models can use."
        ),
        when=(
            "After understanding your data and before modelling. "
            "Common creation strategies: ratios, aggregates, date parts, "
            "and domain features."
        ),
        example=(
            "From 'first_name' and 'last_name' → 'name_length'.\n"
            "From 'price' and 'quantity' → 'total_revenue'.\n"
            "From 'date_of_birth' and today → 'age'."
        ),
        mistakes=[
            "Creating features that leak the target variable.",
            "Not checking if created features are correlated with existing ones.",
            "Over-engineering: creating too many features without evaluation.",
        ],
        interpretation=(
            "Created features should be: (1) logically sound, (2) "
            "computable from available data, and (3) potentially "
            "informative for the target."
        ),
        think_about_it="You have 'start_date' and 'end_date'. What new features could you create?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "# Ratio features\n"
            "df['price_per_sqft'] = df['price'] / df['sqft']\n"
            "\n"
            "# Date features\n"
            "df['age'] = (pd.Timestamp.now() - pd.to_datetime(df['dob'])).dt.days / 365\n"
            "\n"
            "# Aggregated features\n"
            "df['avg_score'] = df[['score1', 'score2', 'score3']].mean(axis=1)\n"
            "```"
        ),
        keywords=["creation", "new features", "derive", "combine", "generate"],
    ),

    "feature_transformation": T(
        title="Feature Transformation",
        module="feature_engineering",
        what=(
            "Feature transformation modifies existing features to "
            "make them more suitable for modelling. Common transforms: "
            "scaling, log, square root, power transforms."
        ),
        why=(
            "Transformations can: reduce skewness, handle outliers, "
            "linearise relationships, and make features more "
            "normally distributed."
        ),
        when=(
            "When features are skewed, have outliers, or have "
            "non-linear relationships with the target."
        ),
        example="Income is right-skewed (a few people earn millions). Log transform makes it more symmetric.",
        mistakes=[
            "Transforming without checking if it improves the model.",
            "Applying log to features with zeros or negatives (use log1p).",
            "Forgetting to inverse-transform predictions if the target was transformed.",
        ],
        interpretation=(
            "Transformations don't change the information content — "
            "they change how the model perceives it. A linear model "
            "may benefit; a tree model may not."
        ),
        think_about_it="You apply log transform to the target variable. How do you get predictions back in the original scale?",
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "from sklearn.preprocessing import PowerTransformer\n"
            "\n"
            "# Log transform\n"
            "df['log_price'] = np.log1p(df['price'])\n"
            "\n"
            "# Power transform (makes data more Gaussian)\n"
            "pt = PowerTransformer(method='yeo-johnson')\n"
            "df[['income_transformed']] = pt.fit_transform(df[['income']])\n"
            "```"
        ),
        keywords=["transformation", "log", "sqrt", "power", "skewness"],
    ),

    "log_transformation": T(
        title="Log Transformation",
        module="feature_engineering",
        what=(
            "Log transformation applies the natural logarithm to a "
            "feature: log(x) or log1p(x) = log(1+x). It compresses "
            "large values and stretches small ones, reducing right skew."
        ),
        why=(
            "Many real-world variables are log-normally distributed: "
            "income, prices, population. Log transform makes them "
            "more symmetric, which helps linear models."
        ),
        when=(
            "Use when: feature is right-skewed, has exponential "
            "relationship with target, or contains multiplicative "
            "effects. Use log1p for features with zeros."
        ),
        example=(
            "```python\n"
            "import numpy as np\n"
            "import plotly.express as px\n"
            "\n"
            "px.histogram(df['income'], title='Before Log')\n"
            "df['log_income'] = np.log1p(df['income'])\n"
            "px.histogram(df['log_income'], title='After Log')\n"
            "```"
        ),
        mistakes=[
            "Using log(x) when x can be zero — use log1p(x) instead.",
            "Not inverse-transforming predictions when target is log-transformed.",
            "Applying log to already symmetric data.",
        ],
        interpretation=(
            "After log transform: multiplicative relationships become "
            "additive. A 10% increase in income becomes a fixed "
            "additive change in log(income)."
        ),
        think_about_it="House prices range from $50K to $5M. After log transform, they range from ~10.8 to ~15.4. What happened to the distribution?",
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "\n"
            "df['log_price'] = np.log1p(df['price'])   # log(1+price)\n"
            "df['log_income'] = np.log1p(df['income'])  # log(1+income)\n"
            "\n"
            "# Inverse transform predictions\n"
            "y_pred_original = np.expm1(y_pred_log)  # exp(y_pred) - 1\n"
            "```"
        ),
        keywords=["log", "logarithm", "log1p", "skewness", "symmetry"],
    ),

    "sqrt_transformation": T(
        title="Square Root Transformation",
        module="feature_engineering",
        what=(
            "Square root transformation applies sqrt(x) to compress "
            "large values less aggressively than log. It works for "
            "count data and moderate skew."
        ),
        why=(
            "Square root is a milder version of log transform. It "
            "works well for count data (number of visits, purchases) "
            "and handles zeros naturally."
        ),
        when=(
            "Use for count data or mildly skewed features. "
            "sqrt(x) handles zeros (unlike log(x)), making it "
            "suitable for frequency counts."
        ),
        example=(
            "```python\n"
            "import numpy as np\n"
            "df['sqrt_visits'] = np.sqrt(df['visit_count'])\n"
            "```"
        ),
        mistakes=[
            "Using sqrt on negative values — not defined.",
            "Using sqrt when data is highly skewed (log is better).",
            "Not checking if the transform improves model performance.",
        ],
        interpretation=(
            "sqrt reduces right skew but less than log. It's the "
            "Variance Stabilizing Transform for Poisson-distributed data."
        ),
        think_about_it="A feature has values 0, 1, 4, 9, 16, 25. What does sqrt transform do to these values?",
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "df['sqrt_count'] = np.sqrt(df['count'])\n"
            "```"
        ),
        keywords=["square root", "sqrt", "count data", "mild skew"],
    ),

    "polynomial_features": T(
        title="Polynomial Features",
        module="feature_engineering",
        what=(
            "Polynomial features create new features by raising existing "
            "features to powers (x², x³) and creating interaction terms "
            "(x1 * x2). This captures non-linear relationships."
        ),
        why=(
            "Linear models can only learn linear relationships. "
            "Polynomial features allow linear models to fit curves "
            "and interaction effects."
        ),
        when=(
            "When residual plots show non-linearity. Start with "
            "degree=2. Higher degrees risk overfitting. Combine with "
            "regularisation (Ridge/Lasso)."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "\n"
            "poly = PolynomialFeatures(degree=2, interaction_only=False)\n"
            "X_poly = poly.fit_transform(X_train)\n"
            "print(f'Original: {X_train.shape[1]} features')\n"
            "print(f'After: {X_poly.shape[1]} features')\n"
            "```"
        ),
        mistakes=[
            "Using high degree (>3) — explodes feature count and overfits.",
            "Not regularising after adding polynomial features.",
            "Forgetting that polynomial features include the originals.",
        ],
        interpretation=(
            "Degree=2 adds: x1², x2², x1*x2 (interactions). "
            "The feature count grows combinatorially — 10 features "
            "become 65 with degree=2."
        ),
        think_about_it="You have 5 features and use PolynomialFeatures(degree=2). How many total features will you have?",
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('poly', PolynomialFeatures(degree=2, include_bias=False)),\n"
            "    ('ridge', Ridge(alpha=1.0))  # regularise!\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "```"
        ),
        keywords=["polynomial", "degree", "interaction", "nonlinear", "power"],
    ),

    "interaction_features": T(
        title="Interaction Features",
        module="feature_engineering",
        what=(
            "Interaction features are products of two or more features: "
            "x1 * x2. They capture combined effects that individual "
            "features cannot."
        ),
        why=(
            "Some effects only appear when two features are considered "
            "together. For example, the effect of 'study hours' on "
            "exam score may depend on 'prior knowledge'."
        ),
        when=(
            "When you suspect features interact, or when domain "
            "knowledge suggests combined effects."
        ),
        example=(
            "```python\n"
            "# Interaction: price per square foot\n"
            "df['price_per_sqft'] = df['price'] * df['sqft']\n"
            "\n"
            "# Using PolynomialFeatures for all interactions\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "poly = PolynomialFeatures(degree=2, interaction_only=True)\n"
            "X_interactions = poly.fit_transform(X_train)\n"
            "```"
        ),
        mistakes=[
            "Creating too many interactions — combinatorial explosion.",
            "Not checking if interactions improve the model.",
            "Forgetting that tree models capture interactions implicitly.",
        ],
        interpretation=(
            "Interaction features multiply two features. If the "
            "interaction coefficient is significant, the effect of "
            "one feature depends on the other."
        ),
        think_about_it="When would you NOT need to create interaction features?",
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import PolynomialFeatures\n"
            "\n"
            "# Only interactions (no squared terms)\n"
            "poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)\n"
            "X_inter = poly.fit_transform(X_train)\n"
            "print(f'Interaction features: {X_inter.shape[1]}')\n"
            "```"
        ),
        keywords=["interaction", "product", "combined", "effect", "multiply"],
    ),

    "binning": T(
        title="Binning (Discretisation)",
        module="feature_engineering",
        what=(
            "Binning converts continuous features into discrete "
            "categories (bins). E.g., ages 0-18='Young', 19-60='Adult', "
            "60+='Senior'."
        ),
        why=(
            "Binning can: handle outliers, capture non-linear "
            "relationships, simplify noisy data, and make features "
            "more interpretable."
        ),
        when=(
            "Use when: the relationship is non-monotonic (very young "
            "and very old both have high risk), when outliers dominate, "
            "or when interpretability matters."
        ),
        example=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "# Equal-width bins\n"
            "df['age_group'] = pd.cut(df['age'], bins=5, labels=['VLow','Low','Med','High','VHigh'])\n"
            "\n"
            "# Equal-frequency bins (quantiles)\n"
            "df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Q1','Q2','Q3','Q4'])\n"
            "```"
        ),
        mistakes=[
            "Using too many bins — defeats the purpose of binning.",
            "Creating bins that lose important information.",
            "Not checking if binning actually helps the model.",
        ],
        interpretation=(
            "Equal-width: bins have same range but different counts. "
            "Equal-frequency: bins have same count but different ranges. "
            "Choose based on data distribution."
        ),
        think_about_it="You bin 'age' into 5 equal-width bins. Bin 5 (80-100) has only 3 samples. Is this useful?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "# Equal-width\n"
            "df['age_bin'] = pd.cut(df['age'], bins=[0,18,35,60,100],\n"
            "                         labels=['Teen','Young','Middle','Senior'])\n"
            "\n"
            "# Equal-frequency (quantile)\n"
            "df['fare_q'] = pd.qcut(df['fare'], q=4, labels=['Low','Med','High','Premium'])\n"
            "```"
        ),
        keywords=["binning", "discretisation", "bins", "cut", "qcut", "categories"],
    ),

    "date_features": T(
        title="Date and Time Features",
        module="feature_engineering",
        what=(
            "Date/time columns can be decomposed into: year, month, "
            "day, hour, day of week, is_weekend, is_month_start, "
            "quarter, etc."
        ),
        why=(
            "Raw datetime values aren't useful for most models. "
            "Extracted components capture seasonal patterns, trends, "
            "and cyclical effects."
        ),
        when=(
            "Whenever you have date/time columns. Check if temporal "
            "patterns exist in the target before engineering."
        ),
        example=(
            "```python\n"
            "df['date'] = pd.to_datetime(df['date'])\n"
            "df['year'] = df['date'].dt.year\n"
            "df['month'] = df['date'].dt.month\n"
            "df['day_of_week'] = df['date'].dt.dayofweek  # 0=Mon\n"
            "df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)\n"
            "df['quarter'] = df['date'].dt.quarter\n"
            "```"
        ),
        mistakes=[
            "Using year as a continuous feature when it's really an identifier.",
            "Not capturing cyclical patterns (month 12 is close to month 1).",
            "Including future information (data leakage with dates).",
        ],
        interpretation=(
            "For cyclical features (hour, month, day), consider "
            "sin/cos encoding: sin(2π * month/12) preserves the "
            "circular nature."
        ),
        think_about_it="Month is encoded as 1-12. The model sees month 12 as very different from month 1. How would you fix this?",
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "\n"
            "# Basic date parts\n"
            "df['month'] = df['date'].dt.month\n"
            "df['day'] = df['date'].dt.day\n"
            "\n"
            "# Cyclical encoding\n"
            "df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)\n"
            "df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)\n"
            "```"
        ),
        keywords=["date", "time", "datetime", "temporal", "seasonal", "cyclical"],
    ),

    "text_features": T(
        title="Text-Based Features",
        module="feature_engineering",
        what=(
            "Text data can be converted to numerical features: "
            "word count, character count, average word length, "
            "presence of specific keywords, TF-IDF, or embeddings."
        ),
        why=(
            "Most ML algorithms require numerical input. Text features "
            "extract signal from free-text fields like reviews, "
            "descriptions, or comments."
        ),
        when=(
            "When you have text columns. Start simple (length, word count) "
            "before moving to TF-IDF or embeddings."
        ),
        example=(
            "```python\n"
            "# Simple text features\n"
            "df['review_length'] = df['review'].str.len()\n"
            "df['word_count'] = df['review'].str.split().str.len()\n"
            "df['has_positive'] = df['review'].str.contains('good|great|excellent', case=False).astype(int)\n"
            "\n"
            "# TF-IDF\n"
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "tfidf = TfidfVectorizer(max_features=100)\n"
            "text_features = tfidf.fit_transform(df['review'])\n"
            "```"
        ),
        mistakes=[
            "Using raw text without any feature extraction.",
            "Creating too many TF-IDF features (>1000) without dimensionality reduction.",
            "Not handling missing text values.",
        ],
        interpretation=(
            "Simple features (length, word count) are fast and "
            "interpretable. TF-IDF captures term importance. "
            "Embeddings (Word2Vec, BERT) capture meaning but are complex."
        ),
        think_about_it="A product review says 'Not bad at all'. Simple keyword matching might flag 'bad' negatively. How would a more sophisticated approach handle this?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "\n"
            "df['text_length'] = df['text'].str.len()\n"
            "df['word_count'] = df['text'].str.split().str.len()\n"
            "\n"
            "tfidf = TfidfVectorizer(max_features=50, stop_words='english')\n"
            "tfidf_df = pd.DataFrame(tfidf.fit_transform(df['text']).toarray(),\n"
            "                         columns=tfidf.get_feature_names_out())\n"
            "```"
        ),
        keywords=["text", "nlp", "tfidf", "word count", "string", "review"],
    ),

    "aggregated_features": T(
        title="Aggregated Features",
        module="feature_engineering",
        what=(
            "Aggregated features compute statistics (mean, sum, count, "
            "std) across groups. E.g., average purchase amount per "
            "customer, total orders per region."
        ),
        why=(
            "Aggregations capture group-level patterns. Individual "
            "transactions tell one story; customer-level aggregates "
            "tell another."
        ),
        when=(
            "When data has a group structure (customers, products, "
            "regions). Always compute aggregations on training data "
            "only to avoid leakage."
        ),
        example=(
            "```python\n"
            "# Customer-level aggregations from transaction data\n"
            "customer_stats = transactions.groupby('customer_id').agg(\n"
            "    total_purchases=('amount', 'sum'),\n"
            "    avg_purchase=('amount', 'mean'),\n"
            "    num_orders=('order_id', 'count'),\n"
            "    std_purchase=('amount', 'std')\n"
            ")\n"
            "```"
        ),
        mistakes=[
            "Computing aggregations on the entire dataset — data leakage!",
            "Not handling NaN in aggregation results.",
            "Creating redundant aggregates (mean and median are often similar).",
        ],
        interpretation=(
            "Aggregated features summarise behaviour at a higher level. "
            "They're particularly powerful for recommendation systems "
            "and customer analytics."
        ),
        think_about_it="You want to predict customer churn. What aggregated features from their purchase history would be most predictive?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "# Group and aggregate\n"
            "agg = df.groupby('category').agg(\n"
            "    mean_price=('price', 'mean'),\n"
            "    count=('price', 'count'),\n"
            "    std_price=('price', 'std')\n"
            ").reset_index()\n"
            "\n"
            "# Merge back\n"
            "df = df.merge(agg, on='category', how='left')\n"
            "```"
        ),
        keywords=["aggregation", "groupby", "mean", "sum", "group", "aggregate"],
    ),

    "encoding_as_feature_engineering": T(
        title="Encoding as Feature Engineering",
        module="feature_engineering",
        what=(
            "Encoding converts categorical features to numbers. "
            "Beyond basic one-hot encoding, advanced techniques like "
            "target encoding, frequency encoding, and binary encoding "
            "can be more effective."
        ),
        why=(
            "Different encoding strategies work better for different "
            "situations. High-cardinality features need different "
            "encoding than low-cardinality ones."
        ),
        when=(
            "When categorical features have many unique values, or "
            "when ordinal relationships exist."
        ),
        example=(
            "```python\n"
            "# Frequency encoding\n"
            "freq = df['city'].value_counts(normalize=True)\n"
            "df['city_freq'] = df['city'].map(freq)\n"
            "\n"
            "# Target encoding (use with care — potential leakage)\n"
            "target_mean = df.groupby('city')['target'].mean()\n"
            "df['city_target_enc'] = df['city'].map(target_mean)\n"
            "```"
        ),
        mistakes=[
            "Target encoding without cross-validation — causes leakage.",
            "One-hot encoding features with 500+ categories.",
            "Not saving the encoding mapping for test data.",
        ],
        interpretation=(
            "One-hot: many columns, good for low cardinality (<20). "
            "Target encoding: one column, good for high cardinality. "
            "Frequency encoding: captures popularity."
        ),
        think_about_it="A 'city' feature has 1000 unique values. One-hot creates 1000 columns. What encoding strategy would you use?",
        code_link=(
            "```python\n"
            "# Frequency encoding\n"
            "freq_map = df['city'].value_counts(normalize=True).to_dict()\n"
            "df['city_freq'] = df['city'].map(freq_map)\n"
            "\n"
            "# Label encoding for ordinal\n"
            "from sklearn.preprocessing import OrdinalEncoder\n"
            "oe = OrdinalEncoder(categories=[['Low','Medium','High']])\n"
            "df['priority_enc'] = oe.fit_transform(df[['priority']])\n"
            "```"
        ),
        keywords=["encoding", "target encoding", "frequency", "binary", "categorical"],
    ),

    "feature_scaling": T(
        title="Feature Scaling for Feature Engineering",
        module="feature_engineering",
        what=(
            "Scaling ensures all features contribute equally to "
            "distance-based models. StandardScaler (mean=0, std=1), "
            "MinMaxScaler (0-1), RobustScaler (median-based)."
        ),
        why=(
            "Features with larger scales dominate distance calculations. "
            "Scaling puts all features on equal footing."
        ),
        when=(
            "For KNN, SVM, K-Means, and neural networks. Not needed "
            "for tree-based models. Always fit on train only."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train)\n"
            "X_test_scaled = scaler.transform(X_test)  # use train stats!\n"
            "```"
        ),
        mistakes=[
            "Fitting scaler on test data — data leakage.",
            "Scaling tree-based model features unnecessarily.",
            "Scaling the target variable for classification.",
        ],
        interpretation=(
            "After scaling, features have mean≈0 and std≈1 "
            "(StandardScaler). This doesn't change relationships, "
            "only magnitudes."
        ),
        think_about_it="Why don't Random Forests need feature scaling while KNN does?",
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler\n"
            "\n"
            "scaler = StandardScaler()   # mean=0, std=1\n"
            "scaler = MinMaxScaler()     # range [0, 1]\n"
            "scaler = RobustScaler()     # median=0, IQR=1\n"
            "\n"
            "X_train_s = scaler.fit_transform(X_train)\n"
            "X_test_s = scaler.transform(X_test)\n"
            "```"
        ),
        keywords=["scaling", "standard", "minmax", "robust", "normalize"],
    ),

    "feature_selection_vs_engineering": T(
        title="Feature Selection vs Feature Engineering",
        module="feature_engineering",
        what=(
            "Feature engineering CREATES new features. Feature selection "
            "CHOOSES the best existing features. Both are essential "
            "parts of the ML pipeline."
        ),
        why=(
            "More features isn't always better. Irrelevant features "
            "add noise, increase computation, and cause overfitting. "
            "Selection removes the noise; engineering adds signal."
        ),
        when=(
            "Engineer first, then select. Create candidate features, "
            "then evaluate which ones improve model performance."
        ),
        example=(
            "Engineering: create 'price_per_sqft' from 'price' and 'sqft'.\n"
            "Selection: remove 'PassengerId' (not informative) and "
            "'Cabin' (too many missing)."
        ),
        mistakes=[
            "Selecting features before engineering — you may remove useful candidates.",
            "Not re-evaluating after engineering new features.",
            "Using all features without any selection.",
        ],
        interpretation=(
            "Feature selection reduces dimensionality and improves "
            "interpretability. Common methods: correlation-based, "
            "variance threshold, recursive feature elimination."
        ),
        think_about_it="You have 100 features. After engineering, you have 150. After selection, you use 30. What happened?",
        code_link=(
            "```python\n"
            "from sklearn.feature_selection import VarianceThreshold, SelectKBest\n"
            "from sklearn.feature_selection import f_classif\n"
            "\n"
            "# Variance threshold\n"
            "vt = VarianceThreshold(threshold=0.01)\n"
            "X_selected = vt.fit_transform(X_train)\n"
            "\n"
            "# SelectKBest\n"
            "selector = SelectKBest(f_classif, k=10)\n"
            "X_selected = selector.fit_transform(X_train, y_train)\n"
            "```"
        ),
        keywords=["selection", "engineering", "choose", "reduce", "features"],
    ),

    "correlation_based_selection": T(
        title="Correlation-Based Feature Selection",
        module="feature_engineering",
        what=(
            "Correlation-based selection removes features that are "
            "highly correlated with each other (redundant) or selects "
            "features most correlated with the target."
        ),
        why=(
            "Highly correlated features provide the same information. "
            "Keeping both adds noise and can cause multicollinearity "
            "in linear models."
        ),
        when=(
            "When you have many features and suspect redundancy. "
            "Remove one of each pair with correlation > 0.9."
        ),
        example=(
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "# Find highly correlated pairs\n"
            "corr_matrix = df.corr().abs()\n"
            "upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))\n"
            "to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]\n"
            "df_reduced = df.drop(columns=to_drop)\n"
            "```"
        ),
        mistakes=[
            "Using Pearson correlation for non-linear relationships.",
            "Removing features correlated with the target (you want those!).",
            "Using correlation on categorical features without encoding first.",
        ],
        interpretation=(
            "Correlation ≈ 1.0 means features are redundant. "
            "Correlation with target ≈ 0 means the feature is useless. "
            "Keep features with high target correlation and low "
            "inter-feature correlation."
        ),
        think_about_it="Features 'temperature_celsius' and 'temperature_fahrenheit' have correlation=1.0. Which should you keep?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "corr = df.corr()\n"
            "target_corr = corr['target'].abs().sort_values(ascending=False)\n"
            "print(target_corr)\n"
            "```"
        ),
        keywords=["correlation", "redundant", "multicollinearity", "selection", "pearson"],
    ),

    "variance_based_selection": T(
        title="Variance-Based Feature Selection",
        module="feature_engineering",
        what=(
            "Variance threshold removes features with very low variance. "
            "A feature with near-zero variance has almost no information "
            "and won't help the model."
        ),
        why=(
            "Constant or near-constant features waste computation and "
            "can confuse some algorithms. Removing them is a quick win."
        ),
        when=(
            "As a first step in feature selection. Remove features with "
            "variance < threshold. For standardised data, use 0.01."
        ),
        example=(
            "```python\n"
            "from sklearn.feature_selection import VarianceThreshold\n"
            "\n"
            "vt = VarianceThreshold(threshold=0.01)\n"
            "X_selected = vt.fit_transform(X_train)\n"
            "print(f'Kept {X_selected.shape[1]} of {X_train.shape[1]} features')\n"
            "```"
        ),
        mistakes=[
            "Setting threshold too high — removing useful features.",
            "Not standardising before using VarianceThreshold.",
            "Forgetting that variance depends on scale.",
        ],
        interpretation=(
            "After VarianceThreshold, check which features were "
            "removed. If a feature was expected to be important, "
            "the threshold may be too aggressive."
        ),
        think_about_it="A binary feature has values 0 and 1, with 99% being 0. What is its variance, and should it be removed?",
        code_link=(
            "```python\n"
            "from sklearn.feature_selection import VarianceThreshold\n"
            "\n"
            "vt = VarianceThreshold(threshold=0.01)\n"
            "X_new = vt.fit_transform(X_train)\n"
            "kept = X_train.columns[vt.get_support()]\n"
            "removed = X_train.columns[~vt.get_support()]\n"
            "print(f'Removed: {list(removed)}')\n"
            "```"
        ),
        keywords=["variance", "threshold", "constant", "low variance", "selection"],
    ),

    "multicollinearity": T(
        title="Multicollinearity",
        module="feature_engineering",
        what=(
            "Multicollinearity occurs when features are highly "
            "correlated with each other. It inflates coefficient "
            "variance and makes interpretation unreliable."
        ),
        why=(
            "In linear models, multicollinearity makes coefficients "
            "unstable — small data changes cause large coefficient "
            "swings. It doesn't hurt tree models but hurts interpretation."
        ),
        when=(
            "Check correlation matrix after feature engineering. "
            "Remove or combine features with correlation > 0.8-0.9."
        ),
        example=(
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "corr = df.corr()\n"
            "high_corr = [(i, j) for i in corr.columns for j in corr.columns\n"
            "              if i != j and abs(corr.loc[i, j]) > 0.9]\n"
            "print(f'Highly correlated pairs: {len(high_corr)}')\n"
            "```"
        ),
        mistakes=[
            "Ignoring multicollinearity in linear models.",
            "Removing features correlated with the target (that's good!).",
            "Not checking VIF (Variance Inflation Factor).",
        ],
        interpretation=(
            "VIF > 10 indicates problematic multicollinearity. "
            "VIF = 1/(1-R²) where R² is from regressing one feature "
            "on all others."
        ),
        think_about_it="You find 'area' and 'sqft' are correlated at 0.95. Which should you keep?",
        code_link=(
            "```python\n"
            "from statsmodels.stats.outliers_influence import variance_inflation_factor\n"
            "import pandas as pd\n"
            "\n"
            "# VIF calculation\n"
            "vif_data = pd.DataFrame()\n"
            "vif_data['Feature'] = X_train.columns\n"
            "vif_data['VIF'] = [variance_inflation_factor(X_train.values, i)\n"
            "                    for i in range(X_train.shape[1])]\n"
            "print(vif_data.sort_values('VIF', ascending=False))\n"
            "```"
        ),
        keywords=["multicollinearity", "correlation", "vif", "redundant", "linear"],
    ),

    "curse_of_dimensionality": T(
        title="Curse of Dimensionality",
        module="feature_engineering",
        what=(
            "As the number of features increases, data becomes "
            "increasingly sparse. Distance metrics lose meaning, "
            "models need exponentially more data, and overfitting "
            "becomes more likely."
        ),
        why=(
            "Feature engineering can create hundreds of new features. "
            "Without dimensionality reduction, performance degrades. "
            "Understanding this trade-off is essential."
        ),
        when=(
            "When feature count is large relative to sample count. "
            "Rule of thumb: you need at least 10x more samples than "
            "features for linear models."
        ),
        example=(
            "10 features in 1000 samples → manageable.\n"
            "1000 features in 100 samples → curse of dimensionality.\n"
            "Solution: PCA, feature selection, regularisation."
        ),
        mistakes=[
            "Creating hundreds of polynomial features without selection.",
            "Not reducing dimensions after one-hot encoding high-cardinality features.",
            "Using KNN in high dimensions — distances become meaningless.",
        ],
        interpretation=(
            "If you have more features than samples, you WILL overfit "
            "without regularisation or dimensionality reduction. "
            "Always check the feature-to-sample ratio."
        ),
        think_about_it="After one-hot encoding, you have 500 features but only 200 samples. What are your options?",
        code_link=(
            "```python\n"
            "from sklearn.decomposition import PCA\n"
            "\n"
            "# PCA for dimensionality reduction\n"
            "pca = PCA(n_components=0.95)  # keep 95% variance\n"
            "X_reduced = pca.fit_transform(X_scaled)\n"
            "print(f'{X_scaled.shape[1]} → {X_reduced.shape[1]} features')\n"
            "```"
        ),
        keywords=["dimensionality", "curse", "sparse", "high dimensional", "pca"],
    ),

    "domain_based_feature_engineering": T(
        title="Domain-Based Feature Engineering",
        module="feature_engineering",
        what=(
            "Domain-based feature engineering uses subject-matter "
            "knowledge to create meaningful features. It requires "
            "understanding the problem, not just the data."
        ),
        why=(
            "The best features often come from domain expertise. "
            "A data scientist who understands healthcare can engineer "
            "medical features that algorithms would never discover."
        ),
        when=(
            "Always. Domain knowledge should guide feature engineering. "
            "Talk to experts, read the literature, understand the "
            "business context."
        ),
        example=(
            "Healthcare: BMI from height and weight.\n"
            "Finance: debt-to-income ratio from debt and income.\n"
            "Retail: days since last purchase, purchase frequency."
        ),
        mistakes=[
            "Engineering features without understanding the domain.",
            "Creating domain features that leak future information.",
            "Not documenting why each feature was created.",
        ],
        interpretation=(
            "Domain features are often the most predictive because "
            "they encode real-world relationships that the model "
            "couldn't learn from raw features alone."
        ),
        think_about_it="You're predicting restaurant health inspection scores. What domain features could you create from the restaurant's data?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "# Domain features for housing\n"
            "df['rooms_per_person'] = df['total_rooms'] / df['household_size']\n"
            "df['income_per_room'] = df['median_income'] / df['rooms_per_person']\n"
            "df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']\n"
            "```"
        ),
        keywords=["domain", "expertise", "subject knowledge", "real world", "business"],
    ),

    "feature_engineering_case_study": T(
        title="Feature Engineering Case Study",
        module="feature_engineering",
        what=(
            "A complete feature engineering workflow: explore data, "
            "create features, transform, select, and evaluate."
        ),
        why=(
            "Seeing the full workflow helps you apply feature "
            "engineering systematically to real problems."
        ),
        when="Use as a reference for any feature engineering project.",
        example="Feature engineering for Titanic survival prediction.",
        mistakes=[
            "Not evaluating feature impact on model performance.",
            "Creating features without domain understanding.",
        ],
        interpretation=(
            "Good feature engineering is iterative: create → evaluate → "
            "refine → repeat. Track which features help."
        ),
        think_about_it="After engineering 20 new features, model accuracy increases by 1%. Was it worth it?",
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "# 1. Create features\n"
            "df['family_size'] = df['SibSp'] + df['Parch'] + 1\n"
            "df['is_alone'] = (df['family_size'] == 1).astype(int)\n"
            "df['fare_per_person'] = df['Fare'] / df['family_size']\n"
            "\n"
            "# 2. Transform\n"
            "df['log_fare'] = np.log1p(df['Fare'])\n"
            "\n"
            "# 3. Evaluate impact\n"
            "features_old = ['Pclass', 'Age', 'Fare']\n"
            "features_new = ['Pclass', 'Age', 'fare_per_person', 'family_size', 'is_alone']\n"
            "\n"
            "for feat_set in [features_old, features_new]:\n"
            "    scores = cross_val_score(RandomForestClassifier(), df[feat_set].fillna(0), y, cv=5)\n"
            "    print(f'{feat_set}: {scores.mean():.4f}')\n"
            "```"
        ),
        keywords=["case study", "workflow", "complete", "titanic", "example"],
    ),
}
