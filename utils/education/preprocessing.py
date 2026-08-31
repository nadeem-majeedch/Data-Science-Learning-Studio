"""
Preprocessing — learning topics for the Data Preprocessing module.
"""
from utils.education.base import T

TOPICS = {
    # ── 1 ──────────────────────────────────────────────────────────
    "why_preprocessing": T(
        title="Why Preprocessing Is Required",
        module="preprocessing",
        what=(
            "Data preprocessing is the step of cleaning and transforming "
            "raw data into a format suitable for machine learning. Real-world "
            "data is messy: it has missing values, inconsistent types, "
            "outliers, and categorical text."
        ),
        why=(
            "ML algorithms require clean, numerical input. Skipping "
            "preprocessing leads to errors, biased models, and unreliable "
            "predictions. Preprocessing is typically 60-80% of a data "
            "science project."
        ),
        when=(
            "After loading data and before any modelling. The typical "
            "workflow is: Load → Inspect → Clean → Preprocess → Split → Model."
        ),
        example=(
            "Raw Titanic data needs:\n"
            "- Missing Age values filled\n"
            "- Sex column encoded (male/female → 0/1)\n"
            "- Fare column scaled\n"
            "- Cabin column handled (too many missing)\n"
            "- Train/test split BEFORE any fitting"
        ),
        mistakes=[
            "Preprocessing before train/test split — causes data leakage.",
            "Applying different preprocessing to train and test sets.",
            "Preprocessing without understanding the data first.",
        ],
        interpretation=(
            "Good preprocessing preserves information while making data "
            "safe for algorithms. Bad preprocessing leaks information or "
            "destroys useful patterns."
        ),
        think_about_it=(
            "If you fill missing Age values using the mean of the entire "
            "dataset (including test), how does that affect your model's "
            "test performance?"
        ),
        code_link=(
            "```python\n"
            "# Typical preprocessing pipeline\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
            "\n"
            "num_pipe = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler())\n"
            "])\n"
            "cat_pipe = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='most_frequent')),\n"
            "    ('encoder', OneHotEncoder(handle_unknown='ignore'))\n"
            "])\n"
            "```"
        ),
        keywords=["preprocessing", "cleaning", "pipeline", "workflow"],
    ),

    # ── 2 ──────────────────────────────────────────────────────────
    "data_cleaning": T(
        title="Data Cleaning",
        module="preprocessing",
        what=(
            "Data cleaning removes or corrects inaccurate, incomplete, "
            "inconsistent, or irrelevant data. It includes handling "
            "missing values, fixing types, removing duplicates, and "
            "correcting errors."
        ),
        why=(
            "Dirty data leads to wrong conclusions. A single corrupted "
            "column can make an entire model useless. Cleaning ensures "
            "the data faithfully represents reality."
        ),
        when=(
            "Immediately after loading. Clean before you analyse or model. "
            "Document every cleaning decision for reproducibility."
        ),
        example=(
            "Cleaning steps for Titanic:\n"
            "1. Drop Cabin (77% missing)\n"
            "2. Fill Age with median\n"
            "3. Fill Embarked with mode\n"
            "4. Drop Name and Ticket (identifiers)\n"
            "5. Encode Sex and Embarked"
        ),
        mistakes=[
            "Cleaning test data independently from training data.",
            "Over-cleaning: removing too many rows and losing information.",
            "Not documenting cleaning decisions for reproducibility.",
        ],
        interpretation=(
            "The goal of cleaning is to maximise usable information "
            "while maintaining data integrity. Every row removed should "
            "have a documented reason."
        ),
        think_about_it=(
            "You clean the training data by removing outliers. Should "
            "you apply the same outlier thresholds to the test set? "
            "Why or why not?"
        ),
        code_link=(
            "```python\n"
            "# Basic cleaning checklist\n"
            "df.drop_duplicates(inplace=True)          # remove duplicates\n"
            "df.drop(columns=['Cabin'], inplace=True)  # drop high-missing column\n"
            "df['Age'].fillna(df['Age'].median(), inplace=True)  # fill missing\n"
            "df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n"
            "```"
        ),
        keywords=["cleaning", "clean", "fix", "correct", "prepare"],
    ),

    # ── 3 ──────────────────────────────────────────────────────────
    "missing_data": T(
        title="Handling Missing Data",
        module="preprocessing",
        what=(
            "Missing data occurs when values are absent from a dataset. "
            "The standard approaches are deletion (drop rows/columns) "
            "or imputation (fill with estimated values)."
        ),
        why=(
            "Most algorithms cannot handle NaN values. Even tree-based "
            "models like XGBoost have limitations with missing data. "
            "How you handle missing data affects model performance and bias."
        ),
        when=(
            "Always before modelling. Choose strategy based on the "
            "proportion of missing data and the mechanism (MCAR, MAR, MNAR)."
        ),
        example=(
            "Titanic 'Age' has 177 missing values (20%):\n"
            "- Drop: lose 177 rows → significant data loss\n"
            "- Mean: biased by outliers (Age is right-skewed)\n"
            "- Median: better for skewed data\n"
            "- Model-based: most accurate but complex"
        ),
        mistakes=[
            "Dropping too many rows (>20% data loss).",
            "Using mean when outliers are present (use median).",
            "Fitting imputer on entire dataset before split — data leakage!",
            "Ignoring missing data and hoping the model handles it.",
        ],
        interpretation=(
            "After imputation, verify: re-check isnull().sum() and "
            "compare df.describe() before/after. If the mean/median "
            "shifted dramatically, the imputation may have distorted "
            "the distribution."
        ),
        think_about_it=(
            "A column has 40% missing values. Should you impute or "
            "drop the column? What factors influence this decision?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.impute import SimpleImputer\n"
            "\n"
            "# Median for numerical (robust to outliers)\n"
            "imputer = SimpleImputer(strategy='median')\n"
            "df[num_cols] = imputer.fit_transform(df[num_cols])\n"
            "\n"
            "# Most frequent for categorical\n"
            "imputer = SimpleImputer(strategy='most_frequent')\n"
            "df[cat_cols] = imputer.fit_transform(df[cat_cols])\n"
            "```"
        ),
        keywords=["missing", "nan", "impute", "dropna", "fillna", "imputer"],
    ),

    # ── 4 ──────────────────────────────────────────────────────────
    "mcar_mar_mnar": T(
        title="MCAR, MAR and MNAR",
        module="preprocessing",
        what=(
            "Missing data mechanisms explain WHY data is missing:\n"
            "- MCAR (Missing Completely At Random): no pattern — "
            "pure chance.\n"
            "- MAR (Missing At Random): missingness depends on observed "
            "data.\n"
            "- MNAR (Missing Not At Random): missingness depends on "
            "the missing value itself."
        ),
        why=(
            "The mechanism determines the best imputation strategy. "
            "MCAR allows simple deletion. MAR supports model-based "
            "imputation. MNAR requires domain knowledge."
        ),
        when=(
            "Before choosing an imputation method. Understanding the "
            "mechanism helps you choose the least biased approach."
        ),
        example=(
            "Titanic Age:\n"
            "- MCAR: Age was randomly not recorded.\n"
            "- MAR: Younger passengers had less complete records.\n"
            "- MNAR: Survivors were less likely to have age recorded "
            "(records were lost during rescue)."
        ),
        mistakes=[
            "Assuming data is MCAR without investigation.",
            "Using simple mean imputation for MNAR data — introduces bias.",
            "Not considering the mechanism when choosing imputation.",
        ],
        interpretation=(
            "If you can't determine the mechanism, assume MAR and use "
            "model-based imputation (e.g., KNNImputer) which is more "
            "robust than simple mean/median."
        ),
        think_about_it=(
            "In a survey, higher-income people are less likely to report "
            "income. Is this MCAR, MAR, or MNAR? What imputation "
            "strategy is appropriate?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.impute import KNNImputer, SimpleImputer\n"
            "\n"
            "# For MAR: model-based imputation\n"
            "imputer = KNNImputer(n_neighbors=5)\n"
            "df_imputed = pd.DataFrame(\n"
            "    imputer.fit_transform(df[num_cols]),\n"
            "    columns=num_cols\n"
            ")\n"
            "\n"
            "# For MCAR: simple deletion is acceptable\n"
            "df_clean = df.dropna(subset=['Age'])\n"
            "```"
        ),
        keywords=["mcar", "mar", "mnar", "mechanism", "missingness"],
    ),

    # ── 5 ──────────────────────────────────────────────────────────
    "dropping_missing": T(
        title="Dropping Missing Values",
        module="preprocessing",
        what=(
            "Dropping removes rows or columns with missing data. "
            "dropna() removes rows; drop(columns=...) removes columns. "
            "It's the simplest approach but can lose valuable information."
        ),
        why=(
            "Sometimes deletion is the right choice: when missing data "
            "is minimal (<5%), when a column has >50% missing, or when "
            "imputation would introduce significant bias."
        ),
        when=(
            "Use when: <5% rows have missing data (safe to drop rows), "
            "or a column has >50% missing (better to drop column), "
            "or data is MCAR."
        ),
        example=(
            "```python\n"
            "# Drop rows with any missing value\n"
            "df_clean = df.dropna()\n"
            "print(f'Lost {len(df) - len(df_clean)} rows')\n"
            "\n"
            "# Drop rows where specific column is missing\n"
            "df_clean = df.dropna(subset=['Age', 'Fare'])\n"
            "\n"
            "# Drop columns with >50% missing\n"
            "threshold = len(df) * 0.5\n"
            "df_clean = df.dropna(axis=1, thresh=threshold)\n"
            "```"
        ),
        mistakes=[
            "Dropping too many rows — check how much data you lose first.",
            "Dropping columns that contain important information.",
            "Dropping before checking if the missingness has a pattern.",
        ],
        interpretation=(
            "Always report: 'Dropped X rows (Y% of data) because of "
            "missing values in columns Z.' This makes your analysis "
            "reproducible and auditable."
        ),
        think_about_it=(
            "You have 1000 rows and drop rows with any missing value, "
            "leaving only 600 rows. Is the remaining data representative? "
            "Could the 400 dropped rows have a pattern?"
        ),
        code_link=(
            "```python\n"
            "df.dropna()                          # drop any row with NaN\n"
            "df.dropna(subset=['Age'])             # drop if Age is NaN\n"
            "df.dropna(axis=1)                     # drop any column with NaN\n"
            "df.dropna(axis=1, thresh=int(0.7*len(df)))  # keep cols with 70%+ data\n"
            "```"
        ),
        keywords=["drop", "dropna", "deletion", "remove rows", "remove columns"],
    ),

    # ── 6 ──────────────────────────────────────────────────────────
    "mean_imputation": T(
        title="Mean Imputation",
        module="preprocessing",
        what=(
            "Mean imputation fills missing numerical values with the "
            "column mean. It's simple and fast but reduces variance "
            "and can distort distributions."
        ),
        why=(
            "Mean imputation preserves the column mean but reduces "
            "spread. It works when data is approximately normal and "
            "missingness is low."
        ),
        when=(
            "Use for numerical columns with roughly symmetric distribution "
            "and <10% missing data. Avoid when outliers are present "
            "(outliers pull the mean)."
        ),
        example=(
            "```python\n"
            "# Age: mean = 29.7, 177 missing values\n"
            "df['Age'].fillna(df['Age'].mean(), inplace=True)\n"
            "\n"
            "# Before: std = 14.5, After: std = 12.8\n"
            "# → Variance decreased (information lost)\n"
            "```"
        ),
        mistakes=[
            "Using mean when outliers exist (use median instead).",
            "Fitting mean on entire dataset before train/test split.",
            "Not comparing distribution before/after imputation.",
        ],
        interpretation=(
            "After mean imputation, check the histogram: the peak "
            "at the mean will be artificially tall. This reduces "
            "variance and can make confidence intervals too narrow."
        ),
        think_about_it=(
            "Income column: mean = $50,000 but most people earn $30,000 "
            "and a few earn millions. Would mean imputation be appropriate? "
            "Why?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.impute import SimpleImputer\n"
            "import numpy as np\n"
            "\n"
            "# Manual\n"
            "df['Age'].fillna(df['Age'].mean(), inplace=True)\n"
            "\n"
            "# Sklearn (preferred — works in pipelines)\n"
            "imputer = SimpleImputer(strategy='mean')\n"
            "df[['Age']] = imputer.fit_transform(df[['Age']])\n"
            "```"
        ),
        keywords=["mean", "imputation", "average", "fillna", "simple"],
    ),

    # ── 7 ──────────────────────────────────────────────────────────
    "median_imputation": T(
        title="Median Imputation",
        module="preprocessing",
        what=(
            "Median imputation fills missing values with the column median "
            "(50th percentile). It's robust to outliers because the median "
            "is not affected by extreme values."
        ),
        why=(
            "When data has outliers or is skewed, the median is a better "
            "measure of central tendency than the mean. This makes "
            "median imputation more reliable for real-world data."
        ),
        when=(
            "Use for skewed numerical data or data with outliers. "
            "This is the default strategy for most preprocessing pipelines."
        ),
        example=(
            "```python\n"
            "# Fare: mean=33.3 but median=14.45 (right-skewed)\n"
            "df['Fare'].fillna(df['Fare'].median(), inplace=True)\n"
            "\n"
            "# Median is not affected by the 512.33 max fare\n"
            "```"
        ),
        mistakes=[
            "Not checking skewness before choosing between mean and median.",
            "Applying median imputation to categorical data.",
            "Forgetting to fit the imputer only on training data.",
        ],
        interpretation=(
            "If mean and median are very different, the data is skewed. "
            "In that case, median imputation is safer. After imputation, "
            "the distribution should look similar but with no missing values."
        ),
        think_about_it=(
            "Age distribution: mean=29.7, median=28.0. These are close. "
            "Does this mean the distribution is roughly symmetric?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.impute import SimpleImputer\n"
            "\n"
            "# Check skewness first\n"
            "print(f'Mean: {df[\"Age\"].mean():.1f}, Median: {df[\"Age\"].median():.1f}')\n"
            "\n"
            "# Impute with median\n"
            "imputer = SimpleImputer(strategy='median')\n"
            "df[['Age']] = imputer.fit_transform(df[['Age']])\n"
            "```"
        ),
        keywords=["median", "imputation", "robust", "outliers", "percentile"],
    ),

    # ── 8 ──────────────────────────────────────────────────────────
    "mode_imputation": T(
        title="Mode Imputation",
        module="preprocessing",
        what=(
            "Mode imputation fills missing categorical values with the "
            "most frequent category. It's the standard approach for "
            "non-numerical columns."
        ),
        why=(
            "Categorical columns can't use mean or median. The mode "
            "(most frequent value) is the logical central tendency for "
            "categories."
        ),
        when=(
            "Use for categorical/string columns. If a category is "
            "dominant (>90%), mode imputation adds little bias. For "
            "more balanced distributions, consider KNN or model-based."
        ),
        example=(
            "```python\n"
            "# Embarked: 'S' appears 644 times, 'C' 168, 'Q' 77\n"
            "# 2 missing values → fill with 'S' (mode)\n"
            "df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n"
            "```"
        ),
        mistakes=[
            "Fitting mode on the entire dataset instead of training only.",
            "Using mode imputation when the mode category is rare (<10%).",
            "Not creating a new 'Missing' category when missingness is informative.",
        ],
        interpretation=(
            "After mode imputation, re-check value_counts(). "
            "If the mode percentage increased significantly, the "
            "imputation may have distorted the distribution."
        ),
        think_about_it=(
            "In a survey, 'Preferred Contact' has 40% Email, 35% Phone, "
            "25% Mail, and 10% missing. Should you use mode imputation "
            "or create a 'Missing' category?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.impute import SimpleImputer\n"
            "\n"
            "# Manual\n"
            "df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n"
            "\n"
            "# Sklearn (works in Pipeline)\n"
            "imputer = SimpleImputer(strategy='most_frequent')\n"
            "df[['Embarked']] = imputer.fit_transform(df[['Embarked']])\n"
            "```"
        ),
        keywords=["mode", "most_frequent", "categorical", "imputation"],
    ),

    # ── 9 ──────────────────────────────────────────────────────────
    "constant_imputation": T(
        title="Constant Imputation",
        module="preprocessing",
        what=(
            "Constant imputation fills missing values with a fixed value "
            "you specify, such as 0, -1, or 'Unknown'. It's useful when "
            "missingness itself is informative."
        ),
        why=(
            "Sometimes missing data has meaning: a missing 'Income' might "
            "mean 'unemployed'. Filling with a constant preserves this "
            "signal rather than masking it."
        ),
        when=(
            "Use when missingness is likely informative (MAR/MNAR), or "
            "when a domain-specific value is more appropriate than "
            "statistics (e.g., 0 for 'no previous purchases')."
        ),
        example=(
            "```python\n"
            "# Fill with constant\n"
            "df['Cabin'].fillna('Unknown', inplace=True)\n"
            "\n"
            "# Or create a separate indicator column first\n"
            "df['Age_missing'] = df['Age'].isnull().astype(int)\n"
            "df['Age'].fillna(-1, inplace=True)\n"
            "```"
        ),
        mistakes=[
            "Choosing an arbitrary constant without domain justification.",
            "Not creating an indicator column to flag imputed values.",
            "Using 0 when 0 is a valid and meaningful value.",
        ],
        interpretation=(
            "The indicator column (Age_missing) lets the model learn "
            "that 'was missing' is itself a useful feature. This is "
            "particularly powerful when missingness is MAR."
        ),
        think_about_it=(
            "A 'Smoking' column has missing values. The patient may or "
            "may not smoke. Would you fill with 'Unknown' or use a "
            "separate indicator? What are the trade-offs?"
        ),
        code_link=(
            "```python\n"
            "# Constant fill\n"
            "df['Cabin'].fillna('Unknown', inplace=True)\n"
            "\n"
            "# Indicator + constant (recommended)\n"
            "df['Age_was_missing'] = df['Age'].isnull().astype(int)\n"
            "df['Age'].fillna(0, inplace=True)\n"
            "\n"
            "# Sklearn constant imputer\n"
            "from sklearn.impute import SimpleImputer\n"
            "imputer = SimpleImputer(strategy='constant', fill_value='Unknown')\n"
            "```"
        ),
        keywords=["constant", "fixed", "indicator", "informative missing"],
    ),

    # ── 10 ─────────────────────────────────────────────────────────
    "duplicate_data": T(
        title="Handling Duplicate Data",
        module="preprocessing",
        what=(
            "Duplicate records are rows that are identical across all "
            "or most columns. They can occur from data entry errors, "
            "merge operations, or data collection issues."
        ),
        why=(
            "Duplicates bias models by over-representing certain patterns. "
            "They also inflate metrics if a training duplicate appears "
            "in the test set."
        ),
        when=(
            "Check and handle duplicates before train/test split. "
            "Use df.duplicated().sum() to count and "
            "df.drop_duplicates() to remove."
        ),
        example=(
            "```python\n"
            "print(f'Duplicates: {df.duplicated().sum()}')\n"
            "df = df.drop_duplicates()\n"
            "print(f'After: {len(df)} rows')\n"
            "\n"
            "# Check duplicates by specific columns\n"
            "df.duplicated(subset=['Name', 'Age']).sum()\n"
            "```"
        ),
        mistakes=[
            "Not checking for near-duplicates (same data, different formatting).",
            "Dropping all duplicates when some are legitimate records.",
            "Not deduplicating before train/test split.",
        ],
        interpretation=(
            "After dropping duplicates, check: Did we lose information? "
            "Are remaining 'duplicates' actually different records with "
            "the same values?"
        ),
        think_about_it=(
            "Two rows have identical features but different survival "
            "outcomes. Are these duplicates? What does this tell you "
            "about data quality?"
        ),
        code_link=(
            "```python\n"
            "df.duplicated().sum()                # count exact duplicates\n"
            "df.duplicated(subset=['Name']).sum()  # duplicates by Name only\n"
            "df.drop_duplicates(inplace=True)      # remove exact duplicates\n"
            "```"
        ),
        keywords=["duplicate", "dedup", "drop_duplicates", "repeated", "clean"],
    ),

    # ── 11 ─────────────────────────────────────────────────────────
    "outliers": T(
        title="Understanding Outliers",
        module="preprocessing",
        what=(
            "Outliers are data points that are significantly different "
            "from other observations. They can be genuine extreme values "
            "or data errors. Their treatment depends on context."
        ),
        why=(
            "Outliers distort mean, standard deviation, and model "
            "training. Linear models and distance-based algorithms "
            "(KNN, SVM) are especially sensitive."
        ),
        when=(
            "Detect outliers after loading, treat them before scaling "
            "and modelling. Use IQR method or Z-score for detection."
        ),
        example=(
            "```python\n"
            "# IQR method\n"
            "Q1 = df['Fare'].quantile(0.25)\n"
            "Q3 = df['Fare'].quantile(0.75)\n"
            "IQR = Q3 - Q1\n"
            "lower = Q1 - 1.5 * IQR\n"
            "upper = Q3 + 1.5 * IQR\n"
            "outliers = df[(df['Fare'] < lower) | (df['Fare'] > upper)]\n"
            "print(f'Outliers: {len(outliers)}')\n"
            "```"
        ),
        mistakes=[
            "Automatically removing all outliers — some are real data.",
            "Not distinguishing between data errors and genuine extremes.",
            "Removing outliers before train/test split.",
        ],
        interpretation=(
            "Fare has max=512.33 but median=14.45. These are genuine "
            "first-class fares, not errors. Removal would bias the model "
            "against understanding premium pricing."
        ),
        think_about_it=(
            "You find 5 houses priced at $5 million in a dataset where "
            "the median is $200K. Are these errors or luxury properties? "
            "How would you decide?"
        ),
        code_link=(
            "```python\n"
            "import numpy as np\n"
            "\n"
            "# Z-score method\n"
            "from scipy import stats\n"
            "z_scores = np.abs(stats.zscore(df[num_cols]))\n"
            "outliers = (z_scores > 3).any(axis=1)\n"
            "print(f'Outlier rows: {outliers.sum()}')\n"
            "```"
        ),
        keywords=["outlier", "extreme", "anomaly", "iqr", "zscore"],
    ),

    # ── 12 ─────────────────────────────────────────────────────────
    "iqr_method": T(
        title="IQR Method for Outlier Detection",
        module="preprocessing",
        what=(
            "The IQR (Interquartile Range) method identifies outliers "
            "as points below Q1 - 1.5*IQR or above Q3 + 1.5*IQR. "
            "IQR = Q3 - Q1 represents the middle 50% of data."
        ),
        why=(
            "IQR is robust to extreme values (unlike Z-score which "
            "uses mean and std). It works well for skewed distributions "
            "and is the basis of box plots."
        ),
        when=(
            "Use when data is skewed or you need a non-parametric method. "
            "Box plots visualize IQR-based outliers automatically."
        ),
        example=(
            "```python\n"
            "Q1 = df['Fare'].quantile(0.25)  # 7.92\n"
            "Q3 = df['Fare'].quantile(0.75)  # 31.00\n"
            "IQR = Q3 - Q1                    # 23.08\n"
            "lower = Q1 - 1.5 * IQR           # -26.70\n"
            "upper = Q3 + 1.5 * IQR           # 65.63\n"
            "# Values above 65.63 are outliers\n"
            "```"
        ),
        mistakes=[
            "Using 1.5*IQR for all contexts — some use 3*IQR for 'extreme' outliers.",
            "Applying IQR to categorical data.",
            "Not visualising outliers with a box plot before removal.",
        ],
        interpretation=(
            "The 1.5*IQR threshold catches ~0.7% of data in a normal "
            "distribution. The 3*IQR threshold catches only ~0.01% "
            "(extreme outliers). Choose based on context."
        ),
        think_about_it=(
            "A box plot shows 20 outliers for Fare. The 1.5*IQR "
            "threshold is standard but the 512 fare is a real first-class "
            "ticket. Should you remove all 20?"
        ),
        code_link=(
            "```python\n"
            "import plotly.express as px\n"
            "\n"
            "fig = px.box(df, y='Fare', title='Fare Distribution with Outliers')\n"
            "fig.show()\n"
            "\n"
            "# IQR calculation\n"
            "Q1, Q3 = df['Fare'].quantile([0.25, 0.75])\n"
            "IQR = Q3 - Q1\n"
            "mask = df['Fare'].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)\n"
            "print(f'Within IQR: {mask.sum()}, Outliers: {(~mask).sum()}')\n"
            "```"
        ),
        keywords=["iqr", "interquartile", "box plot", "quartile", "outlier"],
    ),

    # ── 13 ─────────────────────────────────────────────────────────
    "outlier_treatment": T(
        title="Outlier Treatment Methods",
        module="preprocessing",
        what=(
            "After detecting outliers, you can: (1) remove them, "
            "(2) cap/floor them (winsorise), (3) transform the data "
            "(log), or (4) keep them if they're genuine."
        ),
        why=(
            "The right treatment depends on whether outliers are errors "
            "or real. Removing all outliers from real data reduces the "
            "model's ability to handle extreme cases."
        ),
        when=(
            "After detection. Consider: domain knowledge (is it real?), "
            "proportion (<5% can be capped), and model sensitivity "
            "(linear models need more care than trees)."
        ),
        example=(
            "```python\n"
            "# 1. Cap at IQR bounds (winsorise)\n"
            "upper = Q3 + 1.5 * IQR\n"
            "df['Fare'] = df['Fare'].clip(upper=upper)\n"
            "\n"
            "# 2. Log transform (reduces skewness)\n"
            "import numpy as np\n"
            "df['log_fare'] = np.log1p(df['Fare'])\n"
            "\n"
            "# 3. Remove rows\n"
            "df = df[df['Fare'] <= upper]\n"
            "```"
        ),
        mistakes=[
            "Always removing outliers without considering if they're real.",
            "Capping without understanding the business impact.",
            "Applying outlier treatment before train/test split.",
        ],
        interpretation=(
            "Winsorising (capping) preserves the row but limits the "
            "extreme value. Log transformation reduces skewness "
            "without removing data. Removal is the most aggressive "
            "and should be last resort."
        ),
        think_about_it=(
            "House prices are right-skewed with outliers. Would you "
            "log-transform, winsorise, or remove? What are the "
            "trade-offs of each?"
        ),
        code_link=(
            "```python\n"
            "# Winsorise (cap at percentiles)\n"
            "upper = df['Fare'].quantile(0.95)\n"
            "df['Fare'] = df['Fare'].clip(upper=upper)\n"
            "\n"
            "# Log transform\n"
            "import numpy as np\n"
            "df['log_fare'] = np.log1p(df['Fare'])  # log(1+x) handles zeros\n"
            "\n"
            "# Robust scaling (reduces outlier influence)\n"
            "from sklearn.preprocessing import RobustScaler\n"
            "scaler = RobustScaler()\n"
            "df['fare_scaled'] = scaler.fit_transform(df[['Fare']])\n"
            "```"
        ),
        keywords=["treatment", "winsorise", "clip", "transform", "remove"],
    ),

    # ── 14 ─────────────────────────────────────────────────────────
    "numerical_scaling": T(
        title="Numerical Scaling Overview",
        module="preprocessing",
        what=(
            "Feature scaling transforms numerical features to a common "
            "range. Without scaling, features with larger magnitudes "
            "(e.g., Fare=100) dominate those with smaller ranges "
            "(e.g., Pclass=1-3)."
        ),
        why=(
            "Distance-based algorithms (KNN, SVM, K-Means) are directly "
            "affected by scale. Gradient descent converges faster with "
            "scaled features. Tree-based models don't need scaling."
        ),
        when=(
            "Always scale for: KNN, SVM, Logistic Regression, Neural "
            "Networks, K-Means. Don't need scaling for: Decision Trees, "
            "Random Forest, Gradient Boosting."
        ),
        example=(
            "Without scaling: Fare (0-512) dominates Pclass (1-3) in "
            "distance calculations. With StandardScaler, both have "
            "mean=0, std=1."
        ),
        mistakes=[
            "Scaling the target variable (usually not needed).",
            "Fitting the scaler on test data — data leakage!",
            "Scaling before handling missing values.",
        ],
        interpretation=(
            "After scaling, all features should be comparable in "
            "magnitude. This doesn't change the relationships between "
            "data points, only their absolute values."
        ),
        think_about_it=(
            "Why does Random Forest not need feature scaling, "
            "while KNN absolutely requires it?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler\n"
            "\n"
            "# StandardScaler: mean=0, std=1\n"
            "scaler = StandardScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train)\n"
            "X_test_scaled = scaler.transform(X_test)  # use train stats!\n"
            "```"
        ),
        keywords=["scale", "scaling", "standardize", "normalize", "range"],
    ),

    # ── 15 ─────────────────────────────────────────────────────────
    "standardization": T(
        title="Standardization (Z-Score Scaling)",
        module="preprocessing",
        what=(
            "StandardScaler transforms features to have mean=0 and "
            "standard deviation=1. Formula: z = (x - mean) / std. "
            "It doesn't bound values to a specific range."
        ),
        why=(
            "Standardisation centres data around zero and gives equal "
            "weight to all features. It's the default scaling method "
            "and works well when data is approximately normal."
        ),
        when=(
            "Use when features have different units or ranges and "
            "the data is roughly Gaussian. The default choice for "
            "most ML algorithms."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_train = scaler.fit_transform(X_train)  # fit + transform\n"
            "X_test = scaler.transform(X_test)        # transform only!\n"
            "\n"
            "# Verify: mean ≈ 0, std ≈ 1\n"
            "print(f'Mean: {X_train.mean():.6f}')  # ≈ 0\n"
            "print(f'Std:  {X_train.std():.6f}')   # ≈ 1\n"
            "```"
        ),
        mistakes=[
            "Fitting on test data — always fit on train only.",
            "Using when data has extreme outliers (use RobustScaler).",
            "Scaling the target variable y.",
        ],
        interpretation=(
            "After standardisation, a value of z=2 means the observation "
            "is 2 standard deviations above the mean. This is useful "
            "for detecting outliers and comparing feature importance."
        ),
        think_about_it=(
            "Feature A has range [0, 1000] and Feature B has range [0, 1]. "
            "After StandardScaler, what will their means and stds be?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train)\n"
            "X_test_scaled = scaler.transform(X_test)  # NO fit on test!\n"
            "```"
        ),
        keywords=["standardize", "zscore", "mean", "std", "standard"],
    ),

    # ── 16 ─────────────────────────────────────────────────────────
    "minmax_scaling": T(
        title="Min-Max Scaling",
        module="preprocessing",
        what=(
            "MinMaxScaler transforms features to a fixed range, "
            "typically [0, 1]. Formula: x_scaled = (x - min) / (max - min). "
            "It preserves the shape of the original distribution."
        ),
        why=(
            "Min-Max scaling is useful when you need bounded values "
            "(e.g., neural network inputs, image pixel values). "
            "It doesn't change the distribution shape."
        ),
        when=(
            "Use when algorithms require bounded inputs or when the "
            "data distribution is not Gaussian. Avoid with outliers — "
            "one extreme value can compress all other values."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import MinMaxScaler\n"
            "\n"
            "scaler = MinMaxScaler()  # range [0, 1]\n"
            "X_train = scaler.fit_transform(X_train)\n"
            "X_test = scaler.transform(X_test)\n"
            "\n"
            "# Custom range\n"
            "scaler = MinMaxScaler(feature_range=(-1, 1))\n"
            "```"
        ),
        mistakes=[
            "Using MinMax when outliers are present — outliers compress the scale.",
            "Fitting on test data.",
            "Assuming it handles missing values — it doesn't.",
        ],
        interpretation=(
            "After MinMax, all values are between 0 and 1 (default). "
            "The minimum becomes 0, the maximum becomes 1. Values "
            "in between are proportionally scaled."
        ),
        think_about_it=(
            "Feature A ranges [0, 100] and Feature B ranges [0, 1]. "
            "After MinMax, both will be [0, 1]. But Feature A's data "
            "points will be differently distributed. Is this a problem?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import MinMaxScaler\n"
            "\n"
            "scaler = MinMaxScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train)\n"
            "X_test_scaled = scaler.transform(X_test)\n"
            "print(f'Min: {X_train_scaled.min()}, Max: {X_train_scaled.max()}')\n"
            "```"
        ),
        keywords=["minmax", "min", "max", "range", "bounded", "normalize"],
    ),

    # ── 17 ─────────────────────────────────────────────────────────
    "robust_scaling": T(
        title="Robust Scaling",
        module="preprocessing",
        what=(
            "RobustScaler uses the median and IQR instead of mean and "
            "std. It subtracts the median and divides by IQR. It's "
            "resistant to outliers."
        ),
        why=(
            "When data has outliers, StandardScaler is distorted because "
            "outliers affect mean and std. RobustScaler ignores outliers "
            "in the scaling calculation."
        ),
        when=(
            "Use when data has significant outliers that you want to "
            "keep. Ideal for financial data, sensor data, or any "
            "domain with extreme values."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import RobustScaler\n"
            "\n"
            "scaler = RobustScaler()  # uses median and IQR\n"
            "X_train = scaler.fit_transform(X_train)\n"
            "X_test = scaler.transform(X_test)\n"
            "\n"
            "# Fare: median=14.45, IQR=23.08\n"
            "# Fare=512 → (512 - 14.45) / 23.08 ≈ 21.5\n"
            "```"
        ),
        mistakes=[
            "Not checking if outliers exist before choosing between Robust and Standard.",
            "Forgetting that RobustScaler doesn't bound values to a specific range.",
            "Fitting on test data.",
        ],
        interpretation=(
            "After robust scaling, the median is 0 and values represent "
            "distance from the median in IQR units. A value of 2 means "
            "2 IQRs above the median — clearly an outlier."
        ),
        think_about_it=(
            "Fare has outliers (max=512, median=14). StandardScaler "
            "sets mean=33, std=52. RobustScaler sets median=0, IQR=1. "
            "Which preserves more information about typical fares?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import RobustScaler\n"
            "\n"
            "scaler = RobustScaler()  # median=0, IQR=1\n"
            "X_train_scaled = scaler.fit_transform(X_train)\n"
            "X_test_scaled = scaler.transform(X_test)\n"
            "```"
        ),
        keywords=["robust", "median", "iqr", "outlier", "resistant"],
    ),

    # ── 18 ─────────────────────────────────────────────────────────
    "label_encoding": T(
        title="Label Encoding",
        module="preprocessing",
        what=(
            "Label encoding assigns a unique integer to each category: "
            "low=0, medium=1, high=2. It's simple but implies an order "
            "that may not exist."
        ),
        why=(
            "Some algorithms require numerical input. Label encoding "
            "converts categories to integers. For ordinal features, "
            "this preserves the natural order."
        ),
        when=(
            "Use for ordinal categories where order matters (low/medium/high). "
            "Don't use for nominal categories (red/blue/green) — one-hot "
            "is better there."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import LabelEncoder\n"
            "\n"
            "le = LabelEncoder()\n"
            "df['Size_encoded'] = le.fit_transform(df['Size'])\n"
            "# low=0, medium=1, high=2\n"
            "\n"
            "# To decode back:\n"
            "df['Size'] = le.inverse_transform(df['Size_encoded'])\n"
            "```"
        ),
        mistakes=[
            "Using label encoding for nominal data (implies false order).",
            "Fitting the encoder on test data.",
            "Forgetting to save the encoder for inverse transform.",
        ],
        interpretation=(
            "After label encoding, the numbers represent categories, "
            "not magnitudes. Distance between 0 and 1 isn't necessarily "
            "the same as between 1 and 2."
        ),
        think_about_it=(
            "Cities are labelled: London=0, Paris=1, Tokyo=2. Is this "
            "appropriate? What problem does it cause for algorithms "
            "that use distance?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import LabelEncoder\n"
            "\n"
            "le = LabelEncoder()\n"
            "df['sex_encoded'] = le.fit_transform(df['Sex'])\n"
            "print(dict(zip(le.classes_, le.transform(le.classes_))))\n"
            "```"
        ),
        keywords=["label", "encode", "ordinal", "integer", "categorical"],
    ),

    # ── 19 ─────────────────────────────────────────────────────────
    "one_hot_encoding": T(
        title="One-Hot Encoding",
        module="preprocessing",
        what=(
            "One-hot encoding creates a binary (0/1) column for each "
            "category. A column 'Color' with values red/blue/green "
            "becomes three columns: Color_red, Color_blue, Color_green."
        ),
        why=(
            "Most algorithms require numerical input. One-hot encoding "
            "converts categories without implying false ordinal relationships "
            "(unlike label encoding)."
        ),
        when=(
            "Use for nominal categories (no natural order): color, city, "
            "gender. Don't use for high-cardinality columns (>50 unique "
            "values) — creates too many features."
        ),
        example=(
            "```python\n"
            "# Pandas one-hot\n"
            "df = pd.get_dummies(df, columns=['Sex', 'Embarked'])\n"
            "\n"
            "# Sklearn one-hot (better for pipelines)\n"
            "from sklearn.preprocessing import OneHotEncoder\n"
            "encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n"
            "encoded = encoder.fit_transform(df[['Sex']])\n"
            "```"
        ),
        mistakes=[
            "One-hot encoding the target variable (use LabelEncoder for that).",
            "Not using handle_unknown='ignore' — test categories not in training.",
            "Creating too many columns with high-cardinality features.",
            "Including the first column (dummy variable trap) for linear models.",
        ],
        interpretation=(
            "After one-hot encoding, each new column is 0 or 1. "
            "Check: if a category appears in <1% of rows, it may be "
            "too rare for the model to learn from."
        ),
        think_about_it=(
            "A 'City' column has 500 unique values. One-hot encoding "
            "creates 500 columns. What alternative approaches exist?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import OneHotEncoder\n"
            "\n"
            "encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n"
            "encoded = encoder.fit_transform(df[['Embarked']])\n"
            "print(encoder.get_feature_names_out())\n"
            "# ['Embarked_C', 'Embarked_Q', 'Embarked_S']\n"
            "```"
        ),
        keywords=["one-hot", "dummy", "binary", "encoding", "get_dummies"],
    ),

    # ── 20 ─────────────────────────────────────────────────────────
    "ordinal_encoding": T(
        title="Ordinal Encoding",
        module="preprocessing",
        what=(
            "Ordinal encoding assigns integers to categories that have "
            "a natural order: Education (High School=0, Bachelor=1, "
            "Master=2, PhD=3). It preserves rank relationships."
        ),
        why=(
            "Unlike one-hot encoding, ordinal encoding preserves the "
            "order information, which is valuable for algorithms that "
            "can use ordinal relationships."
        ),
        when=(
            "Use when categories have a meaningful order: education level, "
            "rating (1-5 stars), satisfaction (low/medium/high). "
            "The order must be defined by domain knowledge."
        ),
        example=(
            "```python\n"
            "from sklearn.preprocessing import OrdinalEncoder\n"
            "\n"
            "education_order = [['High School', 'Bachelor', 'Master', 'PhD']]\n"
            "encoder = OrdinalEncoder(categories=education_order)\n"
            "df['edu_encoded'] = encoder.fit_transform(df[['Education']])\n"
            "# High School=0, Bachelor=1, Master=2, PhD=3\n"
            "```"
        ),
        mistakes=[
            "Using ordinal encoding for nominal categories.",
            "Defining the wrong order (e.g., assigning PhD=0).",
            "Not documenting the order mapping.",
        ],
        interpretation=(
            "The encoded integers reflect the order: 0 < 1 < 2 < 3. "
            "The distance between categories is not necessarily equal "
            "but the rank is preserved."
        ),
        think_about_it=(
            "Satisfaction levels: Very Unsatisfied, Unsatisfied, "
            "Neutral, Satisfied, Very Satisfied. How would you "
            "encode these? Is the distance between levels equal?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.preprocessing import OrdinalEncoder\n"
            "\n"
            "order = [['Low', 'Medium', 'High']]\n"
            "encoder = OrdinalEncoder(categories=order)\n"
            "df['priority_encoded'] = encoder.fit_transform(df[['Priority']])\n"
            "```"
        ),
        keywords=["ordinal", "ordered", "rank", "hierarchy", "encode"],
    ),

    # ── 21 ─────────────────────────────────────────────────────────
    "train_test_split": T(
        title="Train/Test Split",
        module="preprocessing",
        what=(
            "Splitting separates your data into a training set (to teach "
            "the model) and a test set (to evaluate it on unseen data). "
            "Typical split: 80% train, 20% test."
        ),
        why=(
            "If you evaluate a model on data it has already seen, you get "
            "an overly optimistic score. The test set simulates 'future' "
            "unseen data."
        ),
        when=(
            "Before any modelling. Always. For classification, use "
            "stratify=y to preserve class proportions in both sets."
        ),
        example=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n"
            "print(f'Train: {len(X_train)}, Test: {len(X_test)}')\n"
            "```"
        ),
        mistakes=[
            "Fitting preprocessing before splitting — data leakage!",
            "Not using stratify for imbalanced classification problems.",
            "Using a test size that's too small (<5%) — unreliable evaluation.",
            "Not setting random_state — results change every run.",
        ],
        interpretation=(
            "The train and test sets should have similar distributions. "
            "Check by comparing y_train.value_counts() and "
            "y_test.value_counts(). Big differences mean the split "
            "went wrong."
        ),
        think_about_it=(
            "If you have only 100 samples, is a 80/20 split reliable? "
            "What technique would give more stable estimates?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y,\n"
            "    test_size=0.2,        # 80/20 split\n"
            "    random_state=42,      # reproducible\n"
            "    stratify=y            # preserve class balance\n"
            ")\n"
            "```"
        ),
        keywords=["split", "train", "test", "validation", "stratify", "holdout"],
    ),

    # ── 22 ─────────────────────────────────────────────────────────
    "data_leakage": T(
        title="Data Leakage",
        module="preprocessing",
        what=(
            "Data leakage occurs when information from the test set "
            "accidentally leaks into the training process. It makes "
            "models appear more accurate than they really are."
        ),
        why=(
            "Leakage gives false confidence. A model with 99% accuracy "
            "due to leakage may only achieve 70% on real data. "
            "It's the most common mistake in ML."
        ),
        when=(
            "Prevent at every step: split first, then preprocess. "
            "Fit scalers/imputers on train only, transform test with "
            "train parameters."
        ),
        example=(
            "Common leakage sources:\n"
            "1. Filling missing values with mean of entire dataset\n"
            "2. Scaling before train/test split\n"
            "3. Feature selection using target variable on full data\n"
            "4. Time series: using future data to predict past"
        ),
        mistakes=[
            "Fitting imputer/scaler on entire dataset before split.",
            "Using test data for feature selection or hyperparameter tuning.",
            "Not using Pipeline to automate correct fit/transform sequence.",
        ],
        interpretation=(
            "If your model's test score is much higher than expected, "
            "leakage is the most likely cause. Always check: Was any "
            "preprocessing fitted on test data?"
        ),
        think_about_it=(
            "You compute the mean of 'Age' across the entire dataset "
            "and use it to fill missing values in both train and test. "
            "Is this leakage? Why or why not?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "# WRONG: fit on entire dataset\n"
            "imputer = SimpleImputer(strategy='mean')\n"
            "X = imputer.fit_transform(X)  # ← leakage!\n"
            "\n"
            "# CORRECT: fit on train only\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "imputer = SimpleImputer(strategy='mean')\n"
            "X_train = imputer.fit_transform(X_train)  # fit + transform\n"
            "X_test = imputer.transform(X_test)        # transform only!\n"
            "```"
        ),
        keywords=["leakage", "data leakage", "leak", "information", "test"],
    ),

    # ── 23 ─────────────────────────────────────────────────────────
    "pipelines": T(
        title="Sklearn Pipelines",
        module="preprocessing",
        what=(
            "A Pipeline chains preprocessing steps and a model into "
            "a single object. When you call fit(), each step fits and "
            "transforms in sequence. This prevents data leakage."
        ),
        why=(
            "Pipelines ensure the correct fit/transform order, make code "
            "cleaner, and prevent the most common data leakage mistakes. "
            "They're the standard best practice in production ML."
        ),
        when=(
            "Use pipelines whenever you have preprocessing + modelling. "
            "They're especially important with cross-validation, where "
            "each fold needs its own fit."
        ),
        example=(
            "```python\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('model', LogisticRegression())\n"
            "])\n"
            "\n"
            "pipe.fit(X_train, y_train)\n"
            "score = pipe.score(X_test, y_test)\n"
            "```"
        ),
        mistakes=[
            "Not using pipelines and manually fitting each step.",
            "Fitting preprocessing outside the pipeline.",
            "Forgetting that pipeline.score() uses the last step's score method.",
        ],
        interpretation=(
            "When you call pipe.fit(X_train, y_train), step 1 fits "
            "and transforms X_train, step 2 fits and transforms the "
            "output, and step 3 fits the model. When you call "
            "pipe.transform(X_test), only transform is called at each step."
        ),
        think_about_it=(
            "You have a pipeline with imputer → scaler → model. "
            "When you call pipe.predict(X_new), does it impute and scale "
            "X_new using parameters learned from training data?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "\n"
            "pipe = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('model', RandomForestClassifier(n_estimators=100))\n"
            "])\n"
            "\n"
            "pipe.fit(X_train, y_train)\n"
            "y_pred = pipe.predict(X_test)\n"
            "```"
        ),
        keywords=["pipeline", "chain", "sklearn", "workflow", "leakage prevention"],
    ),

    # ── 24 ─────────────────────────────────────────────────────────
    "column_transformer": T(
        title="ColumnTransformer",
        module="preprocessing",
        what=(
            "ColumnTransformer applies different preprocessing to "
            "different column types. Numerical columns get scaling; "
            "categorical columns get encoding. It manages both "
            "simultaneously."
        ),
        why=(
            "Real datasets have mixed column types. A single transformer "
            "can't handle both numerical scaling and categorical encoding. "
            "ColumnTransformer makes mixed-type preprocessing clean."
        ),
        when=(
            "Use whenever you have both numerical and categorical features. "
            "It's essential for building proper preprocessing pipelines."
        ),
        example=(
            "```python\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
            "from sklearn.impute import SimpleImputer\n"
            "\n"
            "preprocessor = ColumnTransformer([\n"
            "    ('num', Pipeline([\n"
            "        ('imputer', SimpleImputer(strategy='median')),\n"
            "        ('scaler', StandardScaler())\n"
            "    ]), num_cols),\n"
            "    ('cat', Pipeline([\n"
            "        ('imputer', SimpleImputer(strategy='most_frequent')),\n"
            "        ('encoder', OneHotEncoder(handle_unknown='ignore'))\n"
            "    ]), cat_cols),\n"
            "])\n"
            "```"
        ),
        mistakes=[
            "Applying the same preprocessing to all columns regardless of type.",
            "Not specifying remainder='drop' or 'passthrough'.",
            "Forgetting that OneHotEncoder creates sparse output by default.",
        ],
        interpretation=(
            "ColumnTransformer processes numerical and categorical "
            "columns independently, then concatenates the results. "
            "This ensures each column type gets appropriate treatment."
        ),
        think_about_it=(
            "You have 5 numerical columns and 3 categorical columns. "
            "After ColumnTransformer with StandardScaler and OneHotEncoder, "
            "how many output columns will you have?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.compose import ColumnTransformer\n"
            "\n"
            "preprocessor = ColumnTransformer([\n"
            "    ('num', StandardScaler(), ['Age', 'Fare']),\n"
            "    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sex', 'Embarked']),\n"
            "], remainder='drop')\n"
            "\n"
            "X_processed = preprocessor.fit_transform(X_train)\n"
            "```"
        ),
        keywords=["column", "transformer", "compose", "mixed", "types"],
    ),

    # ── 25 ─────────────────────────────────────────────────────────
    "preprocessing_best_practices": T(
        title="Preprocessing Best Practices",
        module="preprocessing",
        what=(
            "Best practices ensure preprocessing is correct, "
            "reproducible, and doesn't introduce data leakage. "
            "They cover the order of operations, documentation, "
            "and pipeline usage."
        ),
        why=(
            "Following best practices prevents the most common ML "
            "failures: data leakage, incorrect preprocessing, "
            "and non-reproducible results."
        ),
        when=(
            "Every time you preprocess data. Make these habits automatic."
        ),
        example=(
            "Best practices checklist:\n"
            "1. Split data FIRST (train/test)\n"
            "2. Fit preprocessing on train only\n"
            "3. Use Pipeline + ColumnTransformer\n"
            "4. Handle missing values before scaling\n"
            "5. Use stratify for classification splits\n"
            "6. Set random_state for reproducibility\n"
            "7. Document every decision"
        ),
        mistakes=[
            "Preprocessing before splitting — most common mistake.",
            "Not using Pipeline — manual preprocessing is error-prone.",
            "Changing preprocessing based on test set results.",
        ],
        interpretation=(
            "The golden rule: fit on train, transform on both. "
            "If any preprocessing step accesses test data statistics, "
            "you have leakage."
        ),
        think_about_it=(
            "You try several imputation strategies and pick the one "
            "that gives the best test accuracy. Is this valid? What "
            "should you do instead?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "# Full pipeline: preprocessing + model\n"
            "full_pipeline = Pipeline([\n"
            "    ('preprocessor', preprocessor),\n"
            "    ('model', RandomForestClassifier())\n"
            "])\n"
            "\n"
            "# Cross-validation handles train/test splitting internally\n"
            "scores = cross_val_score(full_pipeline, X, y, cv=5)\n"
            "print(f'CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})')\n"
            "```"
        ),
        keywords=["best practices", "checklist", "golden rules", "pipeline", "leakage"],
    ),
}
