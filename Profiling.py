"""
Enterprise-grade Data Profiling Service
Maps all metrics to DQ dimensions (Accuracy, Completeness, Consistency, 
Timeliness, Integrity, Validity).
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from enum import Enum
import json

DB_PATH = "near_production.db"

class DQDimension(Enum):
    COMPLETENESS = "Completeness"
    ACCURACY = "Accuracy"
    CONSISTENCY = "Consistency"
    TIMELINESS = "Timeliness"
    INTEGRITY = "Integrity"
    VALIDITY = "Validity"

class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    MIXED = "mixed"

# ============ 1. DATA LOADING ============

def read_table(table_name: str, limit: int | None = None) -> pd.DataFrame:
    """Load table from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ============ 2. INFER DATA TYPES ============

def infer_data_type(series: pd.Series) -> DataType:
    """Infer the semantic data type of a column."""
    if pd.api.types.is_numeric_dtype(series):
        return DataType.NUMERIC
    elif pd.api.types.is_datetime64_any_dtype(series):
        return DataType.DATETIME
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return DataType.CATEGORICAL
    else:
        return DataType.MIXED

# ============ 3. CORE PROFILING METRICS ============

def profile_numeric_column(series: pd.Series) -> Dict[str, Any]:
    """Profile a numeric column: min, max, mean, std, percentiles, skewness, kurtosis."""
    s = series.dropna().astype("float64")
    if len(s) == 0:
        return {}
    
    profile = {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "variance": float(s.var()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
    }
    
    # Outlier detection (IQR method)
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    outlier_count = ((s < (Q1 - 1.5 * IQR)) | (s > (Q3 + 1.5 * IQR))).sum()
    profile["outlier_count"] = int(outlier_count)
    profile["outlier_pct"] = float(outlier_count / len(s) * 100) if len(s) > 0 else 0.0
    
    return profile

def profile_categorical_column(series: pd.Series) -> Dict[str, Any]:
    """Profile a categorical column: distinct, top values, cardinality band."""
    s = series.dropna()
    if len(s) == 0:
        return {}
    
    distinct = series.nunique(dropna=True)
    total = len(series)
    
    # Cardinality band
    if distinct == 0:
        cardinality = "none"
    elif distinct == 1:
        cardinality = "constant"
    elif distinct < 20:
        cardinality = "low"
    elif distinct < 0.5 * total:
        cardinality = "medium"
    else:
        cardinality = "high"
    
    # Top values
    vc = series.value_counts(dropna=True).head(10)
    top_values = {str(k): int(v) for k, v in vc.items()}
    
    # Entropy (measure of diversity)
    counts = series.value_counts(normalize=True, dropna=True)
    entropy = float(-np.sum(counts * np.log2(counts + 1e-10)))
    
    profile = {
        "distinct_count": int(distinct),
        "cardinality": cardinality,
        "top_values": top_values,
        "entropy": entropy,
    }
    
    return profile

def profile_datetime_column(series: pd.Series) -> Dict[str, Any]:
    """Profile a datetime column: range, frequency, recency."""
    s = series.dropna()
    if len(s) == 0:
        return {}
    
    s_dt = pd.to_datetime(s)
    min_dt = s_dt.min()
    max_dt = s_dt.max()
    
    profile = {
        "min_date": min_dt.isoformat(),
        "max_date": max_dt.isoformat(),
        "range_days": int((max_dt - min_dt).days),
        "unique_dates": int(s_dt.nunique()),
        "days_since_latest": int((datetime.now() - max_dt).days),
    }
    
    return profile

# ============ 4. DQ DIMENSION SCORING ============

def compute_completeness_score(series: pd.Series, row_count: int) -> float:
    """
    Completeness: % of non-null values.
    Range: 0-100. Higher is better.
    """
    nulls = series.isna().sum()
    if row_count == 0:
        return 100.0
    return 100.0 * (1 - nulls / row_count)

def compute_accuracy_score(series: pd.Series, data_type: DataType) -> float:
    """
    Accuracy: based on outliers, invalid types, and distribution.
    For numeric: penalize for extreme outliers.
    For categorical: penalize for high cardinality with sparse values.
    Range: 0-100. Higher is better.
    """
    if data_type == DataType.NUMERIC:
        s = series.dropna().astype("float64")
        if len(s) == 0:
            return 100.0
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((s < (Q1 - 1.5 * IQR)) | (s > (Q3 + 1.5 * IQR))).sum()
        outlier_pct = outliers / len(s) * 100
        # Max 5% outliers = 100%, more = lower score
        accuracy = max(0.0, 100.0 - (outlier_pct - 5.0) * 2)
        return min(100.0, accuracy)
    elif data_type == DataType.CATEGORICAL:
        distinct = series.nunique(dropna=True)
        total = len(series.dropna())
        if total == 0 or distinct == 0:
            return 100.0
        avg_freq = total / distinct
        if avg_freq >= 10:
            return 100.0
        elif avg_freq >= 2:
            return 80.0
        else:
            return 50.0  # Very sparse categorical
    else:
        return 100.0

def compute_consistency_score(series: pd.Series, data_type: DataType) -> float:
    """
    Consistency: measure of uniformity and alignment.
    For categorical: penalize if many unique values with low frequency.
    For numeric: penalize for high variance.
    Range: 0-100. Higher is better.
    """
    if data_type == DataType.CATEGORICAL:
        vc = series.value_counts(normalize=True, dropna=True)
        if len(vc) == 0:
            return 100.0
        entropy = -np.sum(vc * np.log2(vc + 1e-10))
        max_entropy = np.log2(len(series.unique()))
        if max_entropy == 0:
            return 100.0
        normalized_entropy = entropy / max_entropy
        # Lower entropy = more consistent
        return 100.0 * (1 - normalized_entropy)
    elif data_type == DataType.NUMERIC:
        s = series.dropna().astype("float64")
        if len(s) == 0 or s.mean() == 0:
            return 100.0
        cv = (s.std() / abs(s.mean())) if s.mean() != 0 else 0
        # Coefficient of variation: penalize high variance
        consistency = 100.0 / (1 + cv)
        return min(100.0, consistency)
    else:
        return 100.0

def compute_timeliness_score(series: pd.Series, data_type: DataType) -> float:
    """
    Timeliness: measure of recency and no future dates.
    For datetime: penalize if max date is in future or too old.
    Range: 0-100. Higher is better.
    """
    if data_type == DataType.DATETIME:
        s = series.dropna()
        if len(s) == 0:
            return 100.0
        s_dt = pd.to_datetime(s)
        max_dt = s_dt.max()
        now = datetime.now()
        
        future_count = (s_dt > now).sum()
        future_pct = future_count / len(s) * 100
        
        days_since_max = (now - max_dt).days
        
        timeliness = 100.0
        timeliness -= future_pct * 5  # Penalize future dates heavily
        timeliness -= min(50.0, days_since_max / 10)  # Penalize old data
        
        return max(0.0, min(100.0, timeliness))
    else:
        return 100.0

def compute_integrity_score(series: pd.Series, data_type: DataType) -> float:
    """
    Integrity: measure of foreign key validity, cross-field consistency.
    Here, simplified: check for negative IDs, invalid references.
    Range: 0-100. Higher is better.
    """
    if data_type == DataType.NUMERIC:
        s = series.dropna().astype("float64")
        if len(s) == 0:
            return 100.0
        # Check for obvious broken data (negative when shouldn't be)
        if (s < 0).any():
            neg_pct = (s < 0).sum() / len(s) * 100
            return max(0.0, 100.0 - neg_pct * 10)
    return 100.0

def compute_validity_score(series: pd.Series, data_type: DataType) -> float:
    """
    Validity: measure of values conforming to expected format/domain.
    For categorical: penalize for unexpected values.
    For numeric: penalize for out-of-range or invalid values.
    Range: 0-100. Higher is better.
    """
    if data_type == DataType.NUMERIC:
        s = series.dropna().astype("float64")
        if len(s) == 0:
            return 100.0
        # Check for extreme outliers (beyond 99.9th percentile)
        p999 = s.quantile(0.999)
        p001 = s.quantile(0.001)
        extreme = ((s > p999 * 100) | (s < p001 / 100)).sum()
        extreme_pct = extreme / len(s) * 100
        return max(0.0, 100.0 - extreme_pct * 5)
    elif data_type == DataType.CATEGORICAL:
        # Check for "invalid" keywords in categorical values
        s_str = series.astype(str).str.lower()
        invalid_keywords = ["invalid", "null", "na", "nan", "unknown", "n/a"]
        invalid_count = s_str.isin(invalid_keywords).sum()
        invalid_pct = invalid_count / len(series) * 100
        return max(0.0, 100.0 - invalid_pct * 10)
    else:
        return 100.0

# ============ 5. MAIN PROFILING FUNCTION ============

def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profile entire dataframe.
    Returns: {
        "table_metadata": {...},
        "columns": {
            "col_name": {
                "dtype": ...,
                "data_type": ...,
                "metrics": {...},
                "dq_scores": {
                    "completeness": ...,
                    "accuracy": ...,
                    ...
                }
            }
        }
    }
    """
    row_count = len(df)
    profile = {
        "table_metadata": {
            "row_count": row_count,
            "column_count": len(df.columns),
            "profiled_at": datetime.now().isoformat(),
        },
        "columns": {}
    }
    
    for col in df.columns:
        s = df[col]
        data_type = infer_data_type(s)
        
        col_prof = {
            "dtype": str(s.dtype),
            "data_type": data_type.value,
            "metrics": {}
        }
        
        # Compute type-specific metrics
        if data_type == DataType.NUMERIC:
            col_prof["metrics"].update(profile_numeric_column(s))
        elif data_type == DataType.CATEGORICAL:
            col_prof["metrics"].update(profile_categorical_column(s))
        elif data_type == DataType.DATETIME:
            col_prof["metrics"].update(profile_datetime_column(s))
        
        # Compute DQ dimension scores
        col_prof["dq_scores"] = {
            "completeness": compute_completeness_score(s, row_count),
            "accuracy": compute_accuracy_score(s, data_type),
            "consistency": compute_consistency_score(s, data_type),
            "timeliness": compute_timeliness_score(s, data_type),
            "integrity": compute_integrity_score(s, data_type),
            "validity": compute_validity_score(s, data_type),
        }
        
        # Overall column score (average of all dimensions)
        col_prof["overall_dq_score"] = float(
            np.mean(list(col_prof["dq_scores"].values()))
        )
        
        profile["columns"][col] = col_prof
    
    return profile

# ============ 6. SAVE PROFILE TO SQLITE ============

def save_profile_to_sqlite(table_name: str, profile: Dict[str, Any]):
    """Save profiling results to dq_profile table."""
    conn = sqlite3.connect(DB_PATH)
    
    records = []
    for col_name, col_data in profile["columns"].items():
        record = {
            "table_name": table_name,
            "column_name": col_name,
            "profiled_at": profile["table_metadata"]["profiled_at"],
            "dtype": col_data["dtype"],
            "data_type": col_data["data_type"],
            "row_count": profile["table_metadata"]["row_count"],
            "metrics_json": json.dumps(col_data["metrics"]),
            "dq_scores_json": json.dumps(col_data["dq_scores"]),
            "overall_dq_score": col_data["overall_dq_score"],
        }
        records.append(record)
    
    df_records = pd.DataFrame(records)
    df_records.to_sql("dq_profile", conn, if_exists="append", index=False)
    conn.close()
    
    print(f"Saved {len(records)} column profiles to dq_profile table")

# ============ 7. COMPUTE TABLE-LEVEL SCORE ============

def compute_table_dq_score(profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute table-level DQ score per dimension.
    Returns: {
        "completeness": avg_score,
        "accuracy": avg_score,
        ...
        "overall": avg of all dimensions
    }
    """
    scores_per_dimension = {dim.value: [] for dim in DQDimension}
    
    for col_data in profile["columns"].values():
        for dim_name, score in col_data["dq_scores"].items():
            scores_per_dimension[dim_name].append(score)
    
    table_scores = {
        dim: float(np.mean(scores)) if scores else 100.0
        for dim, scores in scores_per_dimension.items()
    }
    
    table_scores["overall"] = float(np.mean(list(table_scores.values())))
    
    return table_scores

if __name__ == "__main__":
    print("=" * 80)
    print("ENTERPRISE DATA PROFILING SERVICE")
    print("=" * 80)
    
    # Profile each table
    for table_name in ["customers", "accounts", "transactions"]:
        print(f"\nProfiling {table_name}...")
        df = read_table(table_name)
        profile = profile_dataframe(df)
        
        # Save to SQLite
        save_profile_to_sqlite(table_name, profile)
        
        # Compute table-level scores
        table_scores = compute_table_dq_score(profile)
        
        print(f"\n{table_name.upper()} - DQ Scores per Dimension:")
        for dim, score in table_scores.items():
            print(f"  {dim:20s}: {score:6.2f}%")
        
        # Show column-level insights
        print(f"\nTop issues in {table_name}:")
        col_scores = {
            col: col_data["overall_dq_score"]
            for col, col_data in profile["columns"].items()
        }
        worst_cols = sorted(col_scores.items(), key=lambda x: x[1])[:3]
        for col, score in worst_cols:
            print(f"  {col:20s}: {score:6.2f}%")
    
    print("\n" + "=" * 80)
    print("Profiling complete. Results saved to dq_profile table.")
    print("=" * 80)
