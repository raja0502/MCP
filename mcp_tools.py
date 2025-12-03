# mcp_tools.py
from typing import Any, Dict
from services import db_service
import pandas as pd

def register_tools(mcp):

    @mcp.tool()
    def profile_table(table_name: str, limit: int | None = None) -> Dict[str, Any]:
        """
        Simple profiler: row_count, nulls, distinct, min/max for numeric.
        """
        df = db_service.read_table(table_name, limit=limit)
        profile: Dict[str, Any] = {}
        for col in df.columns:
            s = df[col]
            info = {
                "dtype": str(s.dtype),
                "nulls": int(s.isna().sum()),
                "distinct": int(s.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(s):
                info["min"] = float(s.min()) if s.count() else None
                info["max"] = float(s.max()) if s.count() else None
            profile[col] = info

        return {
            "table": table_name,
            "row_count": len(df),
            "profile": profile,
        }

    @mcp.tool()
    def check_negative_balance(table_name: str = "customers") -> Dict[str, Any]:
        """
        Example DQ rule tool: find negative balances.
        """
        df = db_service.execute_sql(
            f"SELECT * FROM {table_name} WHERE balance < 0"
        )
        return {
            "table": table_name,
            "issue": "negative_balance",
            "count": len(df),
            "sample": df.to_dict(orient="records"),
        }
