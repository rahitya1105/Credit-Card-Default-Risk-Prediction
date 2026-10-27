"""Download and validate the original UCI dataset. No synthetic fallback."""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "credit_card_clients.csv"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip"
TARGET = "default_next_month"
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
REQUIRED = ["ID", "LIMIT_BAL", *PAY_STATUS, *BILLS, *PAYMENTS, TARGET]


def validate_data(df):
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if df[REQUIRED].isna().any().any():
        raise ValueError("Missing values found in required dataset columns.")
    if df["ID"].duplicated().any():
        raise ValueError("Customer IDs must be unique.")
    if not set(df[TARGET].unique()).issubset({0, 1}) or df[TARGET].nunique() != 2:
        raise ValueError("The target must contain both 0 and 1.")
    if (df["LIMIT_BAL"] <= 0).any():
        raise ValueError("Credit limits must be positive.")
    if (df[PAYMENTS] < 0).any().any():
        raise ValueError("Payment amounts cannot be negative.")
    return df


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urlopen(DOWNLOAD_URL, timeout=90) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        spreadsheets = [n for n in archive.namelist() if n.lower().endswith(".xls")]
        if len(spreadsheets) != 1:
            raise ValueError("Expected one Excel file in the official UCI archive.")
        source_bytes = archive.read(spreadsheets[0])
    # First spreadsheet row contains X1...X23; second contains actual field names.
    df = pd.read_excel(BytesIO(source_bytes), header=1, engine="xlrd")
    df.columns = [str(c).strip() for c in df.columns]
    outcome = [c for c in df.columns if c.lower().startswith("default")]
    if len(outcome) != 1:
        raise ValueError("Could not identify the original default outcome column.")
    df = df.rename(columns={outcome[0]: TARGET})
    validate_data(df)
    if len(df) != 30000:
        raise ValueError(f"Expected the original 30,000 records, received {len(df):,}.")
    df.to_csv(CSV_PATH, index=False)
    (DATA_DIR / "source_metadata.json").write_text(json.dumps({
        "source": SOURCE_PAGE,
        "download": DOWNLOAD_URL,
        "citation": "Yeh, I. (2009). Default of Credit Card Clients. UCI. DOI:10.24432/C55S3H",
        "license": "CC BY 4.0",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha256(archive_bytes).hexdigest(),
        "csv_sha256": sha256(CSV_PATH.read_bytes()).hexdigest(),
        "rows": len(df),
        "period": "April–September 2005 predictors; next-month default outcome",
        "currency": "New Taiwan dollar (NT$)",
    }, indent=2))
    return df


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run python train.py --download first.")
    return validate_data(pd.read_csv(CSV_PATH))


if __name__ == "__main__":
    frame = download_data()
    print(f"Saved {len(frame):,} real historical records to {CSV_PATH}")
