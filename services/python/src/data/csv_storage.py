from pathlib import Path
from typing import Any

import pandas as pd


class CsvStorage:
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    def _ensure_dir(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def save_hourly(self, asset: str, date: str, rows: list[dict[str, Any]]) -> None:
        file_path = Path(self.base_path) / "hourly" / asset / f"{date}.csv"
        self._ensure_dir(file_path)
        new_df = pd.DataFrame(rows)
        if file_path.exists():
            existing_df = pd.read_csv(file_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined.to_csv(file_path, index=False)
        else:
            new_df.to_csv(file_path, index=False)

    def save_daily(self, asset: str, date: str, rows: list[dict[str, Any]]) -> None:
        file_path = Path(self.base_path) / "daily" / asset / f"{date}.csv"
        self._ensure_dir(file_path)
        pd.DataFrame(rows).to_csv(file_path, index=False)

    def save_fundamentals(self, asset: str, date: str, data: dict[str, Any]) -> None:
        file_path = Path(self.base_path) / "fundamentals" / asset / f"{date}.csv"
        self._ensure_dir(file_path)
        pd.DataFrame([data]).to_csv(file_path, index=False)

    def read_hourly(self, asset: str, date: str) -> pd.DataFrame:
        file_path = Path(self.base_path) / "hourly" / asset / f"{date}.csv"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_csv(file_path)

    def read_daily(self, asset: str, date: str) -> pd.DataFrame:
        file_path = Path(self.base_path) / "daily" / asset / f"{date}.csv"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_csv(file_path)

    def read_fundamentals(self, asset: str, date: str) -> pd.DataFrame:
        file_path = Path(self.base_path) / "fundamentals" / asset / f"{date}.csv"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_csv(file_path)
