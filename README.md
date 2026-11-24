# InsightX — AI-Powered Business Intelligence

A working local analytics application for CSV and Excel datasets. The app uses real uploaded data for profiling, quality checks, cleaning previews, KPI calculations, charts, business insights, natural-language questions, and exports.

## Requirements
- Python 3.10+
- pip

## Run locally
```bash
# Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Quick start
1. Click **Load demo data** or upload a `.csv`, `.xlsx`, or `.xls` file.
2. Review the dataset profile and quality checks.
3. Open **Data cleaning** to select safe transformations and preview the result before applying.
4. Explore the dashboard and business insights. Filters update KPI cards and charts.
5. Ask questions in **Ask your data**.
6. Export cleaned CSV/Excel or a PDF summary.

## What is implemented
- CSV and Excel upload, including worksheet selection via `?sheet=...` for Excel.
- Dataset profiling: shape, inferred types, nulls, unique values, duplicates, numeric summaries.
- Quality checks: missing cells, duplicate rows, whitespace, negative quantities, invalid dates and IQR outliers.
- Cleaning preview and approval-based application. Original uploaded data remains available until a cleaning action is applied; the UI offers reset to original.
- Dataset-driven KPIs when suitable columns exist: revenue, profit, orders, units, AOV, margin, customers, growth.
- Dynamic category, region, product and date filters.
- Plotly charts built from the filtered dataset.
- Evidence-based deterministic insights and a basic natural-language analyst.
- CSV, Excel and PDF exports.
- Generated demo dataset with 1,200 rows and a separate deliberately messy test dataset.

## Important limitations
- This is a local single-user starter. It does not include accounts, PostgreSQL persistence, cloud deployment, or multi-user isolation.
- The "AI analyst" is a deterministic data-question assistant, not an external LLM. This avoids sending company data to a third party. An LLM can be added later behind a server-side API key and explicit consent.
- Column detection is heuristic. Always verify detected mappings and KPI definitions for your business.
- `.xls` requires `xlrd`; install it separately if you need legacy Excel files: `pip install xlrd`.
- Large-file performance depends on your machine. The app keeps uploaded datasets in server memory.

## Privacy
Files are processed by the local FastAPI server. No dataset is transmitted to an AI provider by this starter. Stop the server to clear in-memory sessions.
