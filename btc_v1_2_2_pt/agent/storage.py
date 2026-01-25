import csv
import os
from typing import Dict


class CSVStorage:
    def __init__(self, log_dir: str):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

    def _append(self, filename: str, row: Dict) -> None:
        path = os.path.join(self.log_dir, filename)
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def log_price(self, row: Dict) -> None:
        self._append("prices.csv", row)

    def log_signal(self, row: Dict) -> None:
        self._append("signals.csv", row)

    def log_trade(self, row: Dict) -> None:
        self._append("trades.csv", row)

    def log_equity(self, row: Dict) -> None:
        self._append("equity.csv", row)
