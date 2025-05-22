# utils/silver_attributes.py

import os
import re
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_date, regexp_replace, when, lit
from pyspark.sql.types import StringType, IntegerType, DateType

# Configuration & rules
column_type_map = {
    "Customer_ID":   StringType(),
    "Name":          StringType(),
    "Age":           IntegerType(),
    "SSN":           StringType(),
    "Occupation":    StringType(),
    "snapshot_date": DateType(),
}
AGE_MIN = 15
AGE_MAX = 100
SSN_PATTERN = r"^\d{3}-\d{2}-\d{4}$"
OCCUPATION_PATTERN = r"^[A-Za-z ]+$"

def clean_ssn(df: DataFrame) -> DataFrame:
    df = df.withColumn("SSN", regexp_replace(col("SSN"), r"[^0-9\-]", ""))
    return df.withColumn(
        "SSN",
        when(col("SSN").rlike(SSN_PATTERN), col("SSN")).otherwise(lit(None))
    )

def enforce_age(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "Age",
        when(
            (col("Age").cast(IntegerType()) < AGE_MIN) |
            (col("Age").cast(IntegerType()) > AGE_MAX),
            lit(0)
        ).otherwise(col("Age").cast(IntegerType()))
    )

def clean_occupation(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "Occupation",
        when(col("Occupation").rlike(OCCUPATION_PATTERN), col("Occupation")).otherwise(lit(None))
    )

def cast_columns(df: DataFrame) -> DataFrame:
    for c, dtype in column_type_map.items():
        if c not in df.columns:
            continue
        if isinstance(dtype, DateType):
            df = df.withColumn(c, to_date(col(c), "yyyy-MM-dd"))
        else:
            df = df.withColumn(c, col(c).cast(dtype))
    return df

def process_silver_attributes(
    spark: SparkSession,
    csv_path: str,
    silver_base: str
) -> None:
    """
    Bronze→Silver for attributes:
      - Load Bronze CSV
      - Clean SSN, Age, Occupation
      - Cast all columns
      - Extract date from filename (bronze_feature_attributes_YYYY_MM_DD.csv)
      - Write to silver_base/attributes/silver_attributes_<YYYY_MM_DD>.parquet
    """
    df = spark.read.csv(csv_path, header=True, inferSchema=False)
    df = clean_ssn(df)
    df = enforce_age(df)
    df = clean_occupation(df)
    df = cast_columns(df)

    fname = os.path.basename(csv_path)
    m = re.search(r"(\d{4}_\d{2}_\d{2})", fname)
    if not m:
        raise ValueError(f"Cannot parse date from filename {fname}")
    date_str = m.group(1)

    out_dir = os.path.join(silver_base, "attributes", f"silver_attributes_{date_str}.parquet")
    df.repartition(1).write.mode("overwrite").parquet(out_dir)
    print(f"Wrote Attributes → {out_dir}")


