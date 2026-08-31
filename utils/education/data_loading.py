"""
Data Loading — learning topics for the Dataset Explorer module.
"""
from utils.education.base import T

TOPICS = {
    # ── 1. Introduction to Datasets ────────────────────────────────
    "introduction_to_datasets": T(
        title="Introduction to Datasets",
        module="dataset_explorer",
        what=(
            "A dataset is a structured collection of data, usually organised "
            "as rows (records) and columns (attributes). Every data science "
            "project begins with understanding the dataset."
        ),
        why=(
            "If you don't understand your data, every analysis and model "
            "built on it will be unreliable. Knowing what each column "
            "represents is the foundation of good data science."
        ),
        when=(
            "Always the very first step. Before any cleaning, analysis, "
            "or modelling, you must understand what you're working with."
        ),
        example=(
            "Consider a Titanic dataset: each row is a passenger, each "
            "column is a feature (Name, Age, Fare, Survived). The 'Survived' "
            "column is the target we want to predict."
            "\n\n```python\nimport pandas as pd\n"
            "df = pd.read_csv('titanic.csv')\n"
            "print(f'Shape: {df.shape}')  # e.g. (891, 12)\n"
            "print(df.dtypes)\n```"
        ),
        mistakes=[
            "Skipping data understanding and jumping straight to modelling.",
            "Assuming column meanings without reading a data dictionary.",
            "Ignoring that some columns are identifiers, not features.",
        ],
        interpretation=(
            "A good dataset understanding includes: how many rows and "
            "columns, what each column means, what the target is, and "
            "what type of problem you're solving (classification vs regression)."
        ),
        think_about_it=(
            "If a dataset has 50 columns, should you use all of them for "
            "modelling? What factors determine which columns are useful?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "df = pd.read_csv('titanic.csv')\n"
            "print(f'Shape: {df.shape}')  # (891, 12)\n"
            "print(df.columns.tolist())\n"
            "```"
        ),
        keywords=["dataset", "introduction", "basics", "structure"],
    ),

    # ── 2. Structured vs Unstructured Data ─────────────────────────
    "structured_vs_unstructured": T(
        title="Structured vs Unstructured Data",
        module="dataset_explorer",
        what=(
            "Structured data fits neatly into tables (rows and columns). "
            "Unstructured data has no predefined format — text, images, "
            "audio, video. Semi-structured data (JSON, XML) is in between."
        ),
        why=(
            "The type of data determines what tools you use. Pandas and "
            "SQL handle structured data. NLP handles text. Computer vision "
            "handles images. Most introductory data science focuses on "
            "structured data."
        ),
        when=(
            "Identify the data type before choosing your approach. "
            "If data is in CSV/Excel/SQL, it's structured. If it's "
            "free-text or images, you need specialised tools."
        ),
        example=(
            "Structured: a spreadsheet of house prices with columns "
            "for size, location, bedrooms.\n"
            "Unstructured: customer review text, product images.\n"
            "Semi-structured: API responses in JSON format."
        ),
        mistakes=[
            "Treating unstructured data as structured (e.g., using mean on text).",
            "Not converting semi-structured data (JSON) into flat tables before analysis.",
            "Ignoring the volume — unstructured data is 80% of all enterprise data.",
        ],
        interpretation=(
            "For BS Data Science, most coursework uses structured data. "
            "But real-world projects often require converting unstructured "
            "data into structured features (e.g., word counts from text)."
        ),
        think_about_it=(
            "A hospital has patient records (structured), doctor notes "
            "(unstructured text), and X-ray images (unstructured). "
            "How would you combine all three for a disease prediction model?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "import json\n"
            "\n"
            "# Structured: CSV\n"
            "df = pd.read_csv('data.csv')\n"
            "\n"
            "# Semi-structured: JSON\n"
            "with open('data.json') as f:\n"
            "    data = json.load(f)\n"
            "df = pd.json_normalize(data)\n"
            "```"
        ),
        keywords=["structured", "unstructured", "semi-structured", "format"],
    ),

    # ── 3. CSV Files ───────────────────────────────────────────────
    "csv_files": T(
        title="CSV Files",
        module="dataset_explorer",
        what=(
            "CSV (Comma-Separated Values) is the most common format for "
            "tabular data. Each line is a row, and values are separated by "
            "commas (or other delimiters like tabs or semicolons)."
        ),
        why=(
            "CSV is universal — every tool (Excel, Python, R, SQL) can read "
            "it. It's lightweight and human-readable, making it the default "
            "format for sharing datasets."
        ),
        when=(
            "Use CSV for most tabular data exchange. For large datasets "
            "(>1GB), consider Parquet for speed and compression."
        ),
        example=(
            "A CSV file looks like:\n"
            "```\n"
            "Name,Age,City\n"
            "Alice,25,London\n"
            "Bob,30,Paris\n"
            "```"
        ),
        mistakes=[
            "Not specifying the correct delimiter (comma vs tab vs semicolon).",
            "Forgetting encoding='latin-1' for files with special characters.",
            "Not handling quoted fields that contain commas.",
            "Loading huge CSVs entirely into memory instead of chunking.",
        ],
        interpretation=(
            "After loading a CSV, check: Are column names correct? "
            "Are numbers actually numeric? Are there extra/missing columns?"
        ),
        think_about_it=(
            "A CSV file has commas inside a column value, e.g., "
            "'New York, NY'. How does pandas handle this? What parameter "
            "controls it?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "df = pd.read_csv('data.csv')\n"
            "df = pd.read_csv('data.csv', sep=';')       # semicolon-separated\n"
            "df = pd.read_csv('data.csv', encoding='latin-1')  # special chars\n"
            "df = pd.read_csv('data.csv', nrows=1000)     # first 1000 rows only\n"
            "```"
        ),
        keywords=["csv", "comma", "separated", "delimiter", "read_csv"],
    ),

    # ── 4. Excel Files ─────────────────────────────────────────────
    "excel_files": T(
        title="Excel Files",
        module="dataset_explorer",
        what=(
            "Excel files (.xlsx, .xls) store data in worksheets with "
            "formatting, formulas, and multiple sheets. They are common "
            "in business environments."
        ),
        why=(
            "Many non-technical stakeholders share data as Excel files. "
            "Being able to load them programmatically avoids manual "
            "copy-paste and ensures reproducibility."
        ),
        when=(
            "Use pd.read_excel() when the source is an Excel file. "
            "Specify the sheet name if the workbook has multiple sheets."
        ),
        example=(
            "An Excel file might have sheets 'Sales_2023' and "
            "'Sales_2024', each with different column structures."
        ),
        mistakes=[
            "Loading the wrong sheet (default is the first sheet).",
            "Not installing openpyxl (required for .xlsx files).",
            "Merged cells causing unexpected column structures.",
            "Excel formulas showing cached values, not live calculations.",
        ],
        interpretation=(
            "After loading Excel data, verify the shape and columns match "
            "what you see in Excel. Check for merged cells and hidden rows."
        ),
        think_about_it=(
            "An Excel file has a header row but also two rows of "
            "summaries at the bottom. How do you skip those rows?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "df = pd.read_excel('data.xlsx')\n"
            "df = pd.read_excel('data.xlsx', sheet_name='Sales')  # specific sheet\n"
            "df = pd.read_excel('data.xlsx', skiprows=2)  # skip first 2 rows\n"
            "\n"
            "# List all sheet names\n"
            "xls = pd.ExcelFile('data.xlsx')\n"
            "print(xls.sheet_names)\n"
            "```"
        ),
        keywords=["excel", "xlsx", "read_excel", "sheet"],
    ),

    # ── 5. Loading Data with Pandas ────────────────────────────────
    "loading_data_with_pandas": T(
        title="Loading Data with Pandas",
        module="dataset_explorer",
        what=(
            "Pandas provides read_csv(), read_excel(), read_json(), "
            "read_sql(), and many more functions to load data from "
            "various sources into DataFrames."
        ),
        why=(
            "Pandas is the standard data loading tool in Python. "
            "Learning its read functions and their parameters is "
            "essential for any data science workflow."
        ),
        when=(
            "Use the appropriate read_* function based on your data "
            "source. CSV for text files, Excel for spreadsheets, "
            "JSON for web APIs, SQL for databases."
        ),
        example=(
            "Loading Titanic from a URL and from a local file produces "
            "the same DataFrame."
        ),
        mistakes=[
            "Not checking the encoding parameter for non-ASCII characters.",
            "Forgetting that dates may be loaded as strings (use parse_dates).",
            "Not specifying index_col when a column should be the index.",
            "Loading the entire huge file when you only need a sample.",
        ],
        interpretation=(
            "Always run df.shape, df.head(), and df.info() immediately "
            "after loading. This gives you a quick sanity check."
        ),
        think_about_it=(
            "You load a CSV and the 'Price' column is type 'object' "
            "instead of 'float64'. What's the most likely cause and "
            "how would you fix it?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "# CSV\n"
            "df = pd.read_csv('data.csv')\n"
            "\n"
            "# Excel\n"
            "df = pd.read_excel('data.xlsx', sheet_name='Sheet1')\n"
            "\n"
            "# JSON\n"
            "df = pd.read_json('data.json')\n"
            "\n"
            "# From URL\n"
            "url = 'https://example.com/data.csv'\n"
            "df = pd.read_csv(url)\n"
            "```"
        ),
        keywords=["pandas", "read_csv", "read_excel", "read_json", "load"],
    ),

    # ── 6. Dataset Shape ───────────────────────────────────────────
    "dataset_shape": T(
        title="Dataset Shape",
        module="dataset_explorer",
        what=(
            "The shape of a dataset is its dimensions: number of rows "
            "(samples/observations) and number of columns (features/variables). "
            "Accessed via df.shape which returns (rows, columns)."
        ),
        why=(
            "Shape tells you the scale of your data. Very small datasets "
            "may not support complex models. Very large datasets may "
            "require special handling (chunking, sampling)."
        ),
        when=(
            "Check shape immediately after loading data. It helps "
            "identify if columns were dropped or rows were lost during "
            "loading or cleaning."
        ),
        example=(
            "```python\n"
            "df.shape  # (891, 12) → 891 passengers, 12 features\n"
            "```"
        ),
        mistakes=[
            "Ignoring when shape changes unexpectedly during cleaning.",
            "Confusing shape[0] (rows) with shape[1] (columns).",
            "Not checking that row count matches your expectations.",
        ],
        interpretation=(
            "shape[0] = number of observations. shape[1] = number of "
            "variables. After merging datasets, check shape to ensure "
            "no unexpected row multiplication occurred."
        ),
        think_about_it=(
            "You merge two DataFrames on 'ID' and the result has more "
            "rows than either original. What happened?"
        ),
        code_link=(
            "```python\n"
            "print(df.shape)           # (891, 12)\n"
            "print(len(df))            # 891 (rows only)\n"
            "print(len(df.columns))    # 12 (columns only)\n"
            "```"
        ),
        keywords=["shape", "rows", "columns", "dimensions", "size"],
    ),

    # ── 7. Rows and Columns ────────────────────────────────────────
    "rows_and_columns": T(
        title="Understanding Rows and Columns",
        module="dataset_explorer",
        what=(
            "Each row is one observation (e.g., one passenger). "
            "Each column is one attribute or feature of that observation "
            "(e.g., Age, Fare, Pclass). Understanding this is fundamental."
        ),
        why=(
            "Rows = samples. Columns = features + target. Knowing this "
            "helps you understand sklearn's API: X (features, columns) "
            "and y (target, one column)."
        ),
        when=(
            "When selecting features, splitting data, or understanding "
            "model input/output. X has shape (n_samples, n_features)."
        ),
        example=(
            "```python\n"
            "df.head()     # shows first 5 rows\n"
            "df.columns    # Index(['PassengerId', 'Survived', ...])\n"
            "df['Age']     # select one column (Series)\n"
            "df[['Age','Fare']]  # select multiple columns (DataFrame)\n"
            "```"
        ),
        mistakes=[
            "Selecting rows when you meant columns or vice versa.",
            "Confusing df.loc[] (label-based) with df.iloc[] (position-based).",
            "Dropping the target column accidentally during feature selection.",
        ],
        interpretation=(
            "In sklearn: X = DataFrame with feature columns, "
            "y = Series with the target column. The number of rows "
            "must match between X and y."
        ),
        think_about_it=(
            "You want to select rows where Age > 30 AND the passenger "
            "is female. Write the pandas expression."
        ),
        code_link=(
            "```python\n"
            "# Select columns\n"
            "df['Age']                    # single column\n"
            "df[['Age', 'Fare']]          # multiple columns\n"
            "\n"
            "# Select rows\n"
            "df[df['Age'] > 30]           # conditional\n"
            "df.iloc[0:5]                 # by position\n"
            "df.loc[df['Sex'] == 'female'] # by label\n"
            "```"
        ),
        keywords=["rows", "columns", "select", "loc", "iloc", "index"],
    ),

    # ── 8. Data Types ──────────────────────────────────────────────
    "data_types": T(
        title="Understanding Data Types",
        module="dataset_explorer",
        what=(
            "Data types define what kind of values a column holds: "
            "numerical (int, float), categorical (strings/categories), "
            "boolean (True/False), or datetime."
        ),
        why=(
            "Different algorithms expect different data types. "
            "Numerical features can be scaled; categorical features must "
            "be encoded. Using the wrong type silently produces wrong results."
        ),
        when=(
            "Check data types immediately after loading. The dtypes attribute "
            "and df.info() give you a quick overview."
        ),
        example=(
            "```python\n"
            "print(df.dtypes)\n"
            "# sepal_length    float64\n"
            "# species          object  <-- this is categorical\n"
            "\n"
            "# Convert string to category\n"
            "df['species'] = df['species'].astype('category')\n"
            "```"
        ),
        mistakes=[
            "Treating a categorical column as numerical (e.g., zip codes).",
            "Not converting strings that represent numbers (e.g., '$10.50').",
            "Ignoring datetime columns that could be split into year/month/day.",
        ],
        interpretation=(
            "float64/int64 = numerical (can do math on them). "
            "object/category = categorical (need encoding before modelling). "
            "bool = binary (often the target for classification)."
        ),
        think_about_it=(
            "A column 'Gender' has values 'M', 'F', 'Other'. Should this "
            "be treated as categorical or numerical? Why?"
        ),
        code_link=(
            "```python\n"
            "df.dtypes                          # check types\n"
            "df.info()                          # summary with non-null counts\n"
            "df.select_dtypes('number')         # numerical columns only\n"
            "df.select_dtypes('object')         # string columns only\n"
            "```"
        ),
        keywords=["dtype", "type", "numerical", "categorical", "object", "category"],
    ),

    # ── 9. Numerical vs Categorical Data ───────────────────────────
    "numerical_vs_categorical": T(
        title="Numerical vs Categorical Data",
        module="dataset_explorer",
        what=(
            "Numerical data represents measurable quantities (age, price, "
            "temperature). Categorical data represents groups or labels "
            "(color, gender, city). The distinction drives preprocessing "
            "and model choice."
        ),
        why=(
            "Numerical features can be scaled and used directly by most "
            "algorithms. Categorical features require encoding. Mixing "
            "them up leads to incorrect models."
        ),
        when=(
            "Separate numerical and categorical columns before "
            "preprocessing. Use df.select_dtypes() or manual inspection."
        ),
        example=(
            "Titanic dataset:\n"
            "Numerical: Age, Fare, SibSp, Parch\n"
            "Categorical: Sex, Embarked, Pclass (looks numeric but is categorical!)\n"
            "Text: Name, Ticket, Cabin (usually identifiers, not features)"
        ),
        mistakes=[
            "Treating Pclass (1, 2, 3) as numerical — it's ordinal categorical.",
            "Using PassengerId as a feature — it's an identifier, not informative.",
            "Not distinguishing between nominal (no order) and ordinal (ordered) categories.",
        ],
        interpretation=(
            "Pclass has values 1, 2, 3. It's categorical (ordinal) because "
            "the numbers represent categories (1st, 2nd, 3rd class), not "
            "a measurable quantity. The difference between 1 and 2 isn't "
            "necessarily the same as between 2 and 3."
        ),
        think_about_it=(
            "Temperature in Celsius (20, 25, 30) and star rating "
            "(1-5 stars). Both are numerical. Are they the same type of "
            "numerical data?"
        ),
        code_link=(
            "```python\n"
            "# Separate column types\n"
            "num_cols = df.select_dtypes(include='number').columns.tolist()\n"
            "cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()\n"
            "print(f'Numerical: {num_cols}')\n"
            "print(f'Categorical: {cat_cols}')\n"
            "```"
        ),
        keywords=["numerical", "categorical", "continuous", "discrete", "types"],
    ),

    # ── 10. Inspecting Data ────────────────────────────────────────
    "inspecting_data": T(
        title="Inspecting Data with head, tail, sample",
        module="dataset_explorer",
        what=(
            "head() shows the first rows, tail() shows the last rows, "
            "and sample() shows random rows. These are your first tools "
            "for understanding what the data actually looks like."
        ),
        why=(
            "Summary statistics alone can be misleading. Looking at "
            "actual data helps spot formatting issues, outliers, and "
            "unexpected values that numbers won't reveal."
        ),
        when=(
            "Immediately after loading data. Use head() to check structure, "
            "sample() to check variety, tail() to check for trailing junk rows."
        ),
        example=(
            "```python\n"
            "df.head(3)      # first 3 rows\n"
            "df.tail(3)      # last 3 rows\n"
            "df.sample(5)    # 5 random rows\n"
            "```"
        ),
        mistakes=[
            "Only looking at head() — data issues may appear only at the end.",
            "Not scrolling through enough columns when the dataset is wide.",
            "Ignoring trailing rows that might be summary/footer data.",
        ],
        interpretation=(
            "Look for: correct column names, appropriate values, no "
            "obvious encoding issues (e.g., 'Ã©' instead of 'é'), "
            "and no unexpected None/NaN patterns."
        ),
        think_about_it=(
            "You load a CSV and the last row contains 'Total: 500'. "
            "What should you do with it?"
        ),
        code_link=(
            "```python\n"
            "df.head(10)         # first 10 rows\n"
            "df.tail(5)          # last 5 rows\n"
            "df.sample(10)       # 10 random rows\n"
            "df.sample(10, random_state=42)  # reproducible random\n"
            "```"
        ),
        keywords=["head", "tail", "sample", "preview", "inspect"],
    ),

    # ── 11. Data Information with info() ───────────────────────────
    "data_info": T(
        title="Data Information with info()",
        module="dataset_explorer",
        what=(
            "df.info() prints a concise summary: column names, non-null "
            "counts, data types, and memory usage. It's the quickest way "
            "to see missing data and type mismatches."
        ),
        why=(
            "info() reveals two critical issues in one call: (1) columns "
            "with many missing values and (2) columns with unexpected "
            "data types. Both must be fixed before modelling."
        ),
        when=(
            "Run df.info() as the second step after loading data "
            "(after df.head())."
        ),
        example=(
            "```\n"
            "<class 'pandas.core.frame.DataFrame'>\n"
            "RangeIndex: 891 entries, 0 to 890\n"
            "Data columns (total 12 columns):\n"
            " #   Column    Non-Null Count  Dtype  \n"
            "---  ------    --------------  -----  \n"
            " 0   PassengerId  891 non-null   int64  \n"
            " 1   Age          714 non-null   float64  ← 177 missing!\n"
            "```"
        ),
        mistakes=[
            "Ignoring columns with many missing values in info() output.",
            "Not noticing 'object' dtype when you expect 'float64' or 'int64'.",
            "Skipping info() on large datasets — it still works quickly.",
        ],
        interpretation=(
            "Non-null < total rows means missing data. 'object' dtype "
            "on a number column means there are non-numeric values. "
            "High memory usage may indicate you need to optimise dtypes."
        ),
        think_about_it=(
            "A column has 891 non-null values out of 891 rows, but "
            "info() shows it as 'object' type. It should be numeric. "
            "How would you investigate?"
        ),
        code_link=(
            "```python\n"
            "df.info()                    # full summary\n"
            "df.info(memory_usage=False)  # without memory line\n"
            "\n"
            "# Count missing values per column\n"
            "print(df.isnull().sum())\n"
            "```"
        ),
        keywords=["info", "summary", "memory", "non-null", "dtypes"],
    ),

    # ── 12. Basic Statistical Summary ──────────────────────────────
    "descriptive_statistics": T(
        title="Descriptive Statistics",
        module="dataset_explorer",
        what=(
            "df.describe() computes count, mean, std, min, 25%, 50%, 75%, "
            "and max for numerical columns. It gives a quick statistical "
            "snapshot of your data."
        ),
        why=(
            "Statistics reveal the distribution and range of your data. "
            "Mean vs median shows skewness. Min/max reveals outliers. "
            "Std shows spread."
        ),
        when=(
            "After loading and inspecting data. Use describe() for "
            "numerical columns and describe(include='object') for "
            "categorical columns."
        ),
        example=(
            "```python\n"
            "df.describe()  # numerical summary\n"
            "# count    mean     std    min    25%    50%    75%    max\n"
            "# 714    29.699  14.526  0.42  20.12  28.00  38.00  80.00\n"
            "\n"
            "df.describe(include='all')  # includes categorical\n"
            "```"
        ),
        mistakes=[
            "Interpreting describe() on categorical columns (mean makes no sense).",
            "Not noticing when count < total rows (indicates missing values).",
            "Ignoring large gaps between mean and median (indicates skewness).",
        ],
        interpretation=(
            "Mean ≈ Median suggests symmetric distribution. "
            "Mean >> Median suggests right skew (outliers pulling mean up). "
            "Std close to 0 means values are nearly identical."
        ),
        think_about_it=(
            "Age column: mean=29.7, median=28.0, max=80.0, min=0.42. "
            "What can you infer about the distribution? Are there likely "
            "outliers?"
        ),
        code_link=(
            "```python\n"
            "df.describe()                       # numerical\n"
            "df.describe(include='object')       # categorical\n"
            "df['Fare'].describe()               # single column\n"
            "```"
        ),
        keywords=["describe", "statistics", "mean", "median", "std", "summary"],
    ),

    # ── 13. Identifying Missing Values ─────────────────────────────
    "identifying_missing": T(
        title="Identifying Missing Values",
        module="dataset_explorer",
        what=(
            "Missing values (NaN, None, empty cells) occur when data "
            "wasn't recorded, merged, or available. Every dataset needs "
            "a missing-value audit before modelling."
        ),
        why=(
            "Most ML models cannot handle missing data. Even if a model "
            "runs, missing values introduce bias and reduce accuracy."
        ),
        when=(
            "Check immediately after loading. Use df.isnull().sum() for "
            "counts and df.isnull().mean() for proportions."
        ),
        example=(
            "```python\n"
            "df.isnull().sum()          # count per column\n"
            "df.isnull().mean() * 100   # percentage per column\n"
            "\n"
            "# Rows with any missing value\n"
            "df[df.isnull().any(axis=1)]\n"
            "```"
        ),
        mistakes=[
            "Assuming missing values are always NaN — check for placeholders like 0, -1, or 'N/A'.",
            "Only checking columns, not rows — some rows may have many missing fields.",
            "Not checking the proportion — a column with 90% missing is likely unusable.",
        ],
        interpretation=(
            "If a column has <5% missing, imputation is usually safe. "
            "5-25% needs careful imputation. >50% missing means consider "
            "dropping the column entirely."
        ),
        think_about_it=(
            "The 'Cabin' column in Titanic has 77% missing values. "
            "Should you impute or drop it? What if cabin location "
            "actually affects survival?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "df.isnull().sum()              # count NaN per column\n"
            "df.isnull().mean() * 100       # % missing per column\n"
            "df.isnull().any(axis=1).sum()  # rows with any NaN\n"
            "```"
        ),
        keywords=["missing", "nan", "null", "isnull", "na", "missing values"],
    ),

    # ── 14. Identifying Duplicate Records ──────────────────────────
    "duplicate_records": T(
        title="Identifying Duplicate Records",
        module="dataset_explorer",
        what=(
            "Duplicate rows are identical (or near-identical) records "
            "that appear more than once. They can bias models by "
            "over-representing certain patterns."
        ),
        why=(
            "Duplicates inflate apparent dataset size and can cause "
            "data leakage if a duplicate of a training sample appears "
            "in the test set."
        ),
        when=(
            "Check after loading. Use df.duplicated() to detect and "
            "df.drop_duplicates() to remove. Decide whether exact "
            "duplicates or subset duplicates matter."
        ),
        example=(
            "```python\n"
            "df.duplicated().sum()  # count of duplicate rows\n"
            "\n"
            "# Show duplicates\n"
            "df[df.duplicated(keep=False)]\n"
            "\n"
            "# Remove duplicates\n"
            "df_clean = df.drop_duplicates()\n"
            "```"
        ),
        mistakes=[
            "Dropping all duplicates when some are legitimately different records.",
            "Not checking for near-duplicates (same data, slight formatting differences).",
            "Forgetting that train/test split should happen after deduplication.",
        ],
        interpretation=(
            "Some 'duplicates' are legitimate (e.g., two people with the "
            "same attributes). Use domain knowledge to decide. Always "
            "check the count before and after dropping."
        ),
        think_about_it=(
            "Two rows have identical features but different target values. "
            "Are these duplicates? What does this imply about data quality?"
        ),
        code_link=(
            "```python\n"
            "df.duplicated().sum()                 # count duplicates\n"
            "df.duplicated(subset=['Name']).sum()  # duplicate by Name only\n"
            "df.drop_duplicates(inplace=True)      # remove duplicates\n"
            "print(f'Before: {len_before}, After: {len(df)}')\n"
            "```"
        ),
        keywords=["duplicate", "dedup", "drop_duplicates", "repeated"],
    ),

    # ── 15. Data Quality Checks ────────────────────────────────────
    "data_quality": T(
        title="Data Quality Assessment",
        module="dataset_explorer",
        what=(
            "Data quality assessment is a systematic check of your dataset "
            "for completeness, accuracy, consistency, and validity. "
            "It combines missing values, duplicates, type mismatches, "
            "and range checks."
        ),
        why=(
            "Garbage in, garbage out. Poor data quality leads to "
            "misleading analysis and unreliable models. A quality check "
            "before modelling saves hours of debugging later."
        ),
        when=(
            "After loading data and before any analysis. Run a quality "
            "audit that checks: missing values, duplicates, data types, "
            "value ranges, and unique value counts."
        ),
        example=(
            "```python\n"
            "quality_report = {\n"
            "    'rows': len(df),\n"
            "    'columns': len(df.columns),\n"
            "    'missing_pct': df.isnull().mean().to_dict(),\n"
            "    'duplicates': df.duplicated().sum(),\n"
            "    'dtypes': df.dtypes.to_dict(),\n"
            "}\n"
            "```"
        ),
        mistakes=[
            "Skipping quality assessment on 'trusted' data sources.",
            "Only checking missing values — consistency and validity matter too.",
            "Not recording the quality report for reproducibility.",
        ],
        interpretation=(
            "A good quality report answers: Are there missing values? "
            "Duplicates? Columns with wrong types? Values outside "
            "expected ranges? Columns with only one unique value?"
        ),
        think_about_it=(
            "You find that 'Age' has values from -5 to 200. These are "
            "technically not missing, but they're invalid. How do you "
            "handle values that exist but are wrong?"
        ),
        code_link=(
            "```python\n"
            "def quality_check(df):\n"
            "    print(f'Rows: {len(df)}, Cols: {len(df.columns)}')\n"
            "    print(f'Duplicates: {df.duplicated().sum()}')\n"
            "    print('\\nMissing values (%):')\n"
            "    print((df.isnull().mean() * 100).round(1))\n"
            "    print('\\nUnique values:')\n"
            "    for col in df.columns:\n"
            "        print(f'  {col}: {df[col].nunique()}')\n"
            "quality_check(df)\n"
            "```"
        ),
        keywords=["quality", "audit", "check", "completeness", "validity"],
    ),

    # ── 16. Unique Values ──────────────────────────────────────────
    "unique_values": T(
        title="Understanding Unique Values",
        module="dataset_explorer",
        what=(
            "Unique value counts tell you how many distinct values a "
            "column has. This helps identify constant columns (1 unique "
            "value), ID columns (all unique), and cardinality of "
            "categorical features."
        ),
        why=(
            "A column with only one unique value provides no information "
            "for prediction. A column where every row is unique is likely "
            "an identifier. Knowing cardinality guides encoding choices."
        ),
        when=(
            "After loading data. Use df.nunique() for counts and "
            "df[col].value_counts() for distribution of categorical columns."
        ),
        example=(
            "```python\n"
            "df.nunique()  # unique count per column\n"
            "# PassengerId    891  (all unique — ID column)\n"
            "# Survived         2  (binary target)\n"
            "# Pclass           3  (ordinal categorical)\n"
            "# Name           891  (all unique — identifier)\n"
            "```"
        ),
        mistakes=[
            "Keeping columns where nunique() == 1 (constant columns).",
            "Using ID columns as features (they don't generalise to new data).",
            "Not checking nunique() before one-hot encoding (high cardinality = many columns).",
        ],
        interpretation=(
            "nunique() == 1 → constant column, drop it. "
            "nunique() == nrows → likely an ID, drop it. "
            "nunique() < 10 → probably categorical. "
            "nunique() > 50 → high cardinality, consider target encoding."
        ),
        think_about_it=(
            "A 'Name' column has 891 unique values in a dataset with "
            "891 rows. Why shouldn't you use it as a feature?"
        ),
        code_link=(
            "```python\n"
            "df.nunique()                       # unique count per column\n"
            "df['Pclass'].value_counts()        # distribution of one column\n"
            "df['Pclass'].value_counts(normalize=True)  # as percentages\n"
            "\n"
            "# Drop constant columns\n"
            "constant_cols = [c for c in df.columns if df[c].nunique() <= 1]\n"
            "df.drop(columns=constant_cols, inplace=True)\n"
            "```"
        ),
        keywords=["unique", "nunique", "value_counts", "cardinality", "distinct"],
    ),

    # ── 17. Data Validation ────────────────────────────────────────
    "data_validation": T(
        title="Data Validation",
        module="dataset_explorer",
        what=(
            "Data validation checks that values conform to expected "
            "rules: types, ranges, formats, and relationships between "
            "columns. It catches errors that basic inspection misses."
        ),
        why=(
            "Invalid data silently corrupts analysis. A negative age, "
            "a future birthdate, or a fare of 10 million could be "
            "real or erroneous — you need to check."
        ),
        when=(
            "After initial loading. Validate column ranges (Age: 0-120), "
            "consistency (Parch <= SibSp for children), and formats "
            "(valid date strings)."
        ),
        example=(
            "```python\n"
            "# Check for impossible values\n"
            "print(df[df['Age'] < 0])     # negative ages?\n"
            "print(df[df['Age'] > 120])   # impossible ages?\n"
            "print(df[df['Fare'] < 0])    # negative fares?\n"
            "\n"
            "# Check value distributions\n"
            "print(df['Sex'].value_counts())  # only 'male' and 'female'?\n"
            "```"
        ),
        mistakes=[
            "Trusting that data from a database is always valid.",
            "Not validating after transformations (e.g., log of negative number).",
            "Ignoring domain-specific constraints (e.g., room number can't exceed building floors).",
        ],
        interpretation=(
            "Validation errors fall into three categories: (1) data entry "
            "errors (typos), (2) system errors (wrong format), and "
            "(3) genuinely unusual but valid values. Handle each differently."
        ),
        think_about_it=(
            "A passenger has Age=200 and Fare=0. One is clearly an error, "
            "the other might be valid (free ticket). How do you decide "
            "which to fix?"
        ),
        code_link=(
            "```python\n"
            "# Range checks\n"
            "assert df['Age'].between(0, 120).all(), 'Invalid ages found'\n"
            "assert df['Fare'].ge(0).all(), 'Negative fares found'\n"
            "\n"
            "# Type checks\n"
            "print(df.dtypes)\n"
            "\n"
            "# Consistency checks\n"
            "inconsistent = df[df['Pclass'] == 4]  # should be 1, 2, or 3\n"
            "print(f'Invalid Pclass: {len(inconsistent)} rows')\n"
            "```"
        ),
        keywords=["validation", "range", "check", "consistency", "rules"],
    ),

    # ── 18. Loading Large Datasets ─────────────────────────────────
    "loading_large_datasets": T(
        title="Loading Large Datasets",
        module="dataset_explorer",
        what=(
            "Large datasets (>1GB) may not fit in memory. Techniques "
            "include chunked reading, dtype optimisation, column selection "
            "at load time, and using efficient formats like Parquet."
        ),
        why=(
            "Running out of memory during loading crashes your kernel. "
            "Optimised loading also speeds up iteration during EDA and "
            "feature engineering."
        ),
        when=(
            "When df.info() shows high memory usage, or when you know "
            "the dataset is large. Always optimise before loading if you "
            "know the file size exceeds your available RAM."
        ),
        example=(
            "```python\n"
            "# Load only needed columns\n"
            "df = pd.read_csv('big.csv', usecols=['Age', 'Fare', 'Survived'])\n"
            "\n"
            "# Optimise dtypes\n"
            "df = pd.read_csv('big.csv', dtype={'PassengerId': 'int32'})\n"
            "\n"
            "# Chunked reading\n"
            "chunks = pd.read_csv('big.csv', chunksize=10000)\n"
            "for chunk in chunks:\n"
            "    process(chunk)\n"
            "```"
        ),
        mistakes=[
            "Loading all columns when you only need 3.",
            "Using default int64 when int32 or category would save memory.",
            "Not using Parquet for repeated reads of the same large dataset.",
        ],
        interpretation=(
            "Memory usage = rows x columns x bytes_per_value. "
            "1M rows x 50 columns x 8 bytes = 400MB. "
            "Using category dtype for low-cardinality columns can "
            "reduce this by 90%."
        ),
        think_about_it=(
            "Your laptop has 8GB RAM. A CSV is 3GB. Can you load "
            "it entirely? What are your options?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "# Selective loading\n"
            "df = pd.read_csv('big.csv', usecols=['col1', 'col2'])\n"
            "\n"
            "# Optimise memory\n"
            "df['category_col'] = df['category_col'].astype('category')\n"
            "\n"
            "# Parquet (fast, compressed)\n"
            "df.to_parquet('data.parquet')   # save\n"
            "df = pd.read_parquet('data.parquet')  # load\n"
            "```"
        ),
        keywords=["large", "memory", "chunked", "parquet", "optimisation"],
    ),

    # ── 19. Train/Test Data Concept ────────────────────────────────
    "train_test_concept": T(
        title="Train/Test Data Concept",
        module="dataset_explorer",
        what=(
            "Before modelling, data must be split into a training set "
            "(to teach the model) and a test set (to evaluate it on "
            "unseen data). This prevents overfitting and gives a realistic "
            "performance estimate."
        ),
        why=(
            "If you evaluate on training data, the model has already "
            "memorised the answers. The test set simulates 'future' data "
            "the model has never seen."
        ),
        when=(
            "Before any preprocessing or modelling. The split must happen "
            "first so that scaling, imputation, and encoding are fitted "
            "only on training data."
        ),
        example=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n"
            "```"
        ),
        mistakes=[
            "Fitting preprocessing before splitting — data leakage!",
            "Using the test set for any decision (feature selection, hyperparameter tuning).",
            "Not using stratify for classification — imbalanced classes may not be represented.",
        ],
        interpretation=(
            "Typical split: 80% train, 20% test. For small datasets, "
            "use cross-validation instead of a single split. The test "
            "set should only be touched once — at the very end."
        ),
        think_about_it=(
            "You find the best model by testing 10 different algorithms "
            "on the test set and picking the highest score. Is this "
            "problematic? Why?"
        ),
        code_link=(
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y,\n"
            "    test_size=0.2,       # 80/20 split\n"
            "    random_state=42,     # reproducible\n"
            "    stratify=y           # preserve class balance\n"
            ")\n"
            "```"
        ),
        keywords=["train", "test", "split", "holdout", "validation", "leakage"],
    ),

    # ── 20. Common Data Loading Mistakes ───────────────────────────
    "common_loading_mistakes": T(
        title="Common Data Loading Mistakes",
        module="dataset_explorer",
        what=(
            "Data loading errors are subtle and can corrupt your entire "
            "analysis pipeline. The most common mistakes involve encoding, "
            "separator detection, header handling, and type inference."
        ),
        why=(
            "A silent loading error means every subsequent step — EDA, "
            "preprocessing, modelling — operates on wrong data. Catching "
            "these early saves hours of debugging."
        ),
        when=(
            "Always verify after loading: check shape, dtypes, head(), "
            "and a few value_counts(). These quick checks catch 90% of "
            "loading errors."
        ),
        example=(
            "Common errors:\n"
            "1. CSV with semicolons loaded with comma separator → 1 column\n"
            "2. Dates loaded as strings → can't do datetime operations\n"
            "3. Numbers with commas (1,000) loaded as strings\n"
            "4. Excel file loaded with wrong sheet → wrong data\n"
            "5. Index column loaded as a regular column"
        ),
        mistakes=[
            "Not verifying data after loading — assuming it's correct.",
            "Using pd.read_csv without checking separator (comma vs semicolon).",
            "Not setting parse_dates for date columns.",
            "Forgetting header=None when CSV has no header row.",
        ],
        interpretation=(
            "Quick post-load checklist:\n"
            "1. df.shape — matches expected?\n"
            "2. df.head() — looks right?\n"
            "3. df.dtypes — correct types?\n"
            "4. df.isnull().sum() — expected missing pattern?"
        ),
        think_about_it=(
            "You load a CSV and see only 1 column with values like "
            "'Name,Age,City'. What went wrong and how do you fix it?"
        ),
        code_link=(
            "```python\n"
            "import pandas as pd\n"
            "\n"
            "# Common fixes:\n"
            "df = pd.read_csv('data.csv', encoding='latin-1')  # encoding\n"
            "df = pd.read_csv('data.csv', sep=';')             # separator\n"
            "df = pd.read_csv('data.csv', header=0)             # header row\n"
            "df = pd.read_csv('data.csv', parse_dates=['date']) # dates\n"
            "df = pd.read_csv('data.csv', na_values=['N/A', '']) # custom NaN\n"
            "\n"
            "# Always verify:\n"
            "print(df.shape)\n"
            "print(df.dtypes)\n"
            "print(df.head())\n"
            "```"
        ),
        keywords=["mistakes", "encoding", "separator", "header", "common errors"],
    ),
}
