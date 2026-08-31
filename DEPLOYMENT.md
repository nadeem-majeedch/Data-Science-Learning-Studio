# Deployment Guide

## Streamlit Community Cloud

### Prerequisites

- A GitHub repository containing the project.
- A [Streamlit Community Cloud](https://streamlit.io/cloud) account (free tier available).

### Deployment Steps

1. **Push your repository to GitHub.**

2. **Go to [share.streamlit.io](https://share.streamlit.io).**

3. **Click "New app".**

4. **Configure:**
   - **Repository:** Select your GitHub repository.
   - **Branch:** `main` (or your deployment branch).
   - **Main file path:** `app.py`

5. **Click "Deploy".**

Streamlit Cloud will automatically:
- Detect Python version from your repository.
- Install packages from `requirements.txt`.
- Start the application.

### Required Secrets

**No Streamlit secrets are required.**

This application does not use `st.secrets`, environment variables, or external API keys. All functionality runs locally within the application.

### What Is Included in the Repository

| File / Directory | Purpose |
|---|---|
| `app.py` | Streamlit entry point (home page) |
| `pages/` | All module pages (13 pages) |
| `utils/` | Shared utility modules |
| `learning/` | Learning curriculum content |
| `datasets/` | Sample datasets (iris, titanic, breast_cancer, wine_quality, california_housing) |
| `tests/` | Test suite (not used at runtime) |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Streamlit server configuration |

### Runtime Notes

- **SQLite database** (`experiments.db`): Created at runtime to store experiment records. This file is ephemeral on Streamlit Cloud — data resets when the app restarts.
- **File uploads**: Users can upload CSV/Excel files through the Dataset Explorer. These are held in memory only and are not persisted.
- **Sample datasets**: Included in the `datasets/` directory and accessible from the Dataset Explorer.

### Troubleshooting

| Issue | Solution |
|---|---|
| App fails to start | Check the logs for missing dependencies. Ensure `requirements.txt` is complete. |
| Import errors | Verify all Python files are present in the repository. |
| Port already in use | Streamlit Cloud handles port assignment automatically. |
| Memory errors | California Housing dataset is ~20K rows — should work within free tier limits. |

### Python Version

This application requires **Python 3.9+**. Streamlit Community Cloud defaults to a compatible version.

### Manual Local Deployment

```bash
# Clone the repository
git clone <your-repo-url>
cd Data-Science-Lab

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will open at `http://localhost:8501`.
