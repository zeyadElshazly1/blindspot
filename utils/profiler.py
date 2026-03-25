import pandas as pd
import numpy as np
from scipy import stats

def profile_dataset(df):
    profile = []
    
    for col in df.columns:
        col_data = df[col]
        missing = col_data.isnull().sum()
        missing_pct = round(missing / len(df) * 100, 1)
        unique = col_data.nunique()
        unique_pct = round(unique / len(df) * 100, 1)
        dtype = str(col_data.dtype)

        col_profile = {
            "column": col,
            "dtype": dtype,
            "missing": missing,
            "missing_pct": missing_pct,
            "unique": unique,
            "unique_pct": unique_pct,
        }

        if col_data.dtype in [np.float64, np.int64, float, int]:
            clean = col_data.dropna()
            col_profile.update({
                "type": "numeric",
                "mean": round(float(clean.mean()), 2),
                "median": round(float(clean.median()), 2),
                "std": round(float(clean.std()), 2),
                "min": round(float(clean.min()), 2),
                "max": round(float(clean.max()), 2),
                "q25": round(float(clean.quantile(0.25)), 2),
                "q75": round(float(clean.quantile(0.75)), 2),
                "skewness": round(float(clean.skew()), 2),
                "outliers": int((np.abs(stats.zscore(clean)) > 3).sum()),
                "zeros": int((clean == 0).sum()),
            })
        elif col_data.dtype == 'datetime64[ns]':
            clean = col_data.dropna()
            col_profile.update({
                "type": "datetime",
                "min": str(clean.min()),
                "max": str(clean.max()),
                "range_days": (clean.max() - clean.min()).days,
            })
        else:
            top_values = col_data.value_counts().head(5).to_dict()
            col_profile.update({
                "type": "categorical",
                "top_values": top_values,
                "most_common": str(col_data.mode()[0]) if len(col_data.mode()) > 0 else "N/A",
                "most_common_pct": round(col_data.value_counts().iloc[0] / len(df) * 100, 1) if len(col_data.value_counts()) > 0 else 0,
            })

        # Data quality flags
        flags = []
        if missing_pct > 30:
            flags.append("high missing data")
        if unique_pct == 100 and col_data.dtype == object:
            flags.append("possible ID column")
        if col_data.dtype in [np.float64, np.int64] and col_profile.get("outliers", 0) > 0:
            flags.append(f"{col_profile['outliers']} outliers")
        if col_data.dtype in [np.float64, np.int64] and abs(col_profile.get("skewness", 0)) > 2:
            flags.append("highly skewed")
        if unique == 1:
            flags.append("constant column")

        col_profile["flags"] = flags
        profile.append(col_profile)

    return profile