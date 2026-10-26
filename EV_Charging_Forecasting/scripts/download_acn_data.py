"""
ACN-Data Session Downloader
===========================
This script downloads authentic EV charging session records from the Caltech ACN-Data REST API.

API Documentation: https://ev.caltech.edu/dataset
Authentication: HTTP Basic Auth where Username = API_TOKEN, Password = "" (empty).

Usage:
    python scripts/download_acn_data.py --site caltech --start 2020-01-01 --end 2020-03-31
    python scripts/download_acn_data.py --token YOUR_TOKEN --site caltech --start 2020-01-01 --end 2020-03-31
"""

import os
import sys
import json
import argparse
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import pandas as pd


def get_project_root() -> Path:
    """Returns the root directory of the EV_Charging_Forecasting project."""
    return Path(__file__).resolve().parent.parent


def download_acn_sessions(api_token: str, site: str, start_date: str, end_date: str) -> list:
    """
    Queries the Caltech ACN-Data REST API for EV charging sessions between start_date and end_date.
    
    Args:
        api_token: The personal API token from ev.caltech.edu/dataset.
        site: One of 'caltech', 'jpl', or 'office001'.
        start_date: Start date string formatted as YYYY-MM-DD.
        end_date: End date string formatted as YYYY-MM-DD.

    Returns:
        A list of dictionary records containing charging sessions.
    """
    base_url = f"https://ev.caltech.edu/api/v1/sessions/{site}/{start_date}/{end_date}"
    print(f"[*] Requesting data from: {base_url}")
    print(f"[*] Site: {site} | Range: {start_date} to {end_date}")

    auth = HTTPBasicAuth(api_token, "")
    try:
        response = requests.get(base_url, auth=auth, timeout=60)
    except requests.RequestException as e:
        print(f"[!] Network error occurred while connecting to ACN-Data: {e}")
        sys.exit(1)

    if response.status_code == 401:
        print("[!] Authentication failed (HTTP 401).")
        print("    Please check that your ACN_DATA_API_TOKEN is valid.")
        print("    You can generate a free token at: https://ev.caltech.edu/dataset")
        sys.exit(1)
    elif response.status_code != 200:
        print(f"[!] API request failed with status code {response.status_code}:")
        print(f"    {response.text}")
        sys.exit(1)

    data = response.json()
    sessions = data.get("_items", [])
    print(f"[+] Successfully retrieved {len(sessions)} charging sessions from Caltech ACN-Data.")
    return sessions


def save_sessions(sessions: list, raw_dir: Path) -> tuple[Path, Path]:
    """
    Saves retrieved sessions to raw JSON and tabular CSV files.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / "acn_sessions.json"
    csv_path = raw_dir / "acn_sessions.csv"

    # Save complete JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)
    print(f"[+] Saved raw JSON to: {json_path}")

    # Normalize to pandas DataFrame and save CSV
    if sessions:
        df = pd.json_normalize(sessions)
        df.to_csv(csv_path, index=False)
        print(f"[+] Saved flattened CSV to: {csv_path}")
    else:
        print("[!] No session items found in date range; empty CSV not created.")

    return json_path, csv_path


def main():
    root = get_project_root()
    load_dotenv(dotenv_path=root / ".env")

    parser = argparse.ArgumentParser(description="Download Caltech ACN-Data EV Charging Sessions")
    parser.add_argument("--token", default=os.getenv("ACN_DATA_API_TOKEN"), help="ACN-Data API Token")
    parser.add_argument("--site", default=os.getenv("ACN_SITE", "caltech"), choices=["caltech", "jpl", "office001"], help="ACN Site")
    parser.add_argument("--start", default=os.getenv("ACN_START_DATE", "2020-01-01"), help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=os.getenv("ACN_END_DATE", "2020-03-31"), help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    token = args.token
    if not token or token == "your_caltech_acn_api_token_here":
        print("\n" + "=" * 70)
        print(" [!] ACN-Data API Token is required.")
        print("=" * 70)
        print("To download the authentic dataset:")
        print("1. Visit: https://ev.caltech.edu/dataset")
        print("2. Create a free account or log in.")
        print("3. Copy your API token.")
        print("4. Provide it via either:")
        print("   - .env file: ACN_DATA_API_TOKEN=your_token")
        print("   - Command line: python scripts/download_acn_data.py --token YOUR_TOKEN")
        print("=" * 70 + "\n")
        sys.exit(1)

    raw_dir = root / "data" / "raw"
    sessions = download_acn_sessions(
        api_token=token,
        site=args.site,
        start_date=args.start,
        end_date=args.end,
    )
    save_sessions(sessions, raw_dir)
    print("\n[✔] Data acquisition completed successfully! You can now run notebooks/01_data_acquisition_and_inspection.ipynb.\n")


if __name__ == "__main__":
    main()
