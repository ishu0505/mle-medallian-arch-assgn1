# utils/silver_financials.py

import os
import re
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_date, regexp_replace, when, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# Configuration & rules
column_type_map = {
    "Customer_ID":              StringType(),
    "Annual_Income":            FloatType(),
    "Monthly_Inhand_Salary":    FloatType(),
    "Num_Bank_Accounts":        IntegerType(),
    "Num_Credit_Card":          IntegerType(),
    "Interest_Rate":            IntegerType(),
    "Num_of_Loan":              IntegerType(),
    "Type_of_Loan":             StringType(),
    "Delay_from_due_date":      IntegerType(),
    "Num_of_Delayed_Payment":   IntegerType(),
    "Changed_Credit_Limit":     FloatType(),
    "Num_Credit_Inquiries":     IntegerType(),
    "Credit_Mix":               StringType(),
    "Outstanding_Debt":         FloatType(),
    "Credit_Utilization_Ratio": FloatType(),
    "Credit_History_Age":       StringType(),  # raw at Silver
    "Payment_of_Min_Amount":    StringType(),  # raw at Silver
    "Total_EMI_per_month":      FloatType(),
    "Amount_invested_monthly":  FloatType(),
    "Payment_Behaviour":        StringType(),
    "Monthly_Balance":          FloatType(),
    "snapshot_date":            DateType(),
}
INTEREST_MIN = 0
INTEREST_MAX = 100
PAYBEH_PATTERN = r"^[A-Za-z0-9 _]+$"

def clean_numerics(df: DataFrame) -> DataFrame:
    cols = [
        "Annual_Income", "Monthly_Inhand_Salary", "Changed_Credit_Limit",
        "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
    ]
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, regexp_replace(col(c), r"[^0-9\.\-]", ""))
    return df

def enforce_interest_rate(df: DataFrame) -> DataFrame:
    if "Interest_Rate" in df.columns:
        df = df.withColumn(
            "Interest_Rate",
            when(
                (col("Interest_Rate").cast(IntegerType()) < INTEREST_MIN) |
                (col("Interest_Rate").cast(IntegerType()) > INTEREST_MAX),
                lit(0)
            ).otherwise(col("Interest_Rate").cast(IntegerType()))
        )
    return df

def clean_payment_behaviour(df: DataFrame) -> DataFrame:
    if "Payment_Behaviour" in df.columns:
        df = df.withColumn(
            "Payment_Behaviour",
            when(col("Payment_Behaviour").rlike(PAYBEH_PATTERN), col("Payment_Behaviour")).otherwise(lit(None))
        )
    return df

def cast_columns(df: DataFrame) -> DataFrame:
    for c, dtype in column_type_map.items():
        if c not in df.columns:
            continue
        if isinstance(dtype, DateType):
            df = df.withColumn(c, to_date(col(c), "yyyy-MM-dd"))
        else:
            df = df.withColumn(c, col(c).cast(dtype))
    return df

def process_silver_financials(
    spark: SparkSession,
    csv_path: str,
    silver_base: str
) -> None:
    """
    Bronze→Silver for financials:
      - Load Bronze CSV
      - Clean numerics, enforce interest rate, clean payment behaviour
      - Cast all columns
      - Extract date from filename (bronze_feature_financials_YYYY_MM_DD.csv)
      - Write to silver_base/financials/silver_financials_<YYYY_MM_DD>.parquet
    """
    df = spark.read.csv(csv_path, header=True, inferSchema=False)
    df = clean_numerics(df)
    df = enforce_interest_rate(df)
    df = clean_payment_behaviour(df)
    df = cast_columns(df)

    fname = os.path.basename(csv_path)
    m = re.search(r"(\d{4}_\d{2}_\d{2})", fname)
    if not m:
        raise ValueError(f"Cannot parse date from filename {fname}")
    date_str = m.group(1)

    out_dir = os.path.join(silver_base, "financials", f"silver_financials_{date_str}.parquet")
    df.repartition(1).write.mode("overwrite").parquet(out_dir)
    print(f"Wrote Financials → {out_dir}")
