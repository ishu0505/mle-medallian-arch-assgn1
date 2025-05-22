# utils/silver_clickstream.py

import os
import re
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_date, when, lit, floor
from pyspark.sql.types import IntegerType, StringType, DateType

# Configuration & rules
column_type_map = {
    **{f"fe_{i}": IntegerType() for i in range(1,21)},
    "Customer_ID":   StringType(),
    "snapshot_date": DateType(),
}

def cast_feature_columns(df: DataFrame) -> DataFrame:
    for c in column_type_map:
        if c.startswith("fe_") and c in df.columns:
            df = df.withColumn(c, col(c).cast(IntegerType()))
    return df

def cast_meta_columns(df: DataFrame) -> DataFrame:
    if "Customer_ID" in df.columns:
        df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    if "snapshot_date" in df.columns:
        df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))
    return df

def validate_features(df: DataFrame) -> None:
    nulls = {c: df.filter(col(c).isNull()).count() for c in column_type_map if c.startswith("fe_")}
    floats= {c: df.filter(col(c).isNotNull() & (col(c)!=floor(col(c)))).count() for c in column_type_map if c.startswith("fe_")}
    print("Feature nulls:", nulls)
    print("Non-integer floats:", floats)

def process_silver_clickstream(
    spark: SparkSession,
    csv_path: str,
    silver_base: str
) -> None:
    """
    Bronze→Silver for clickstream:
      - Load Bronze CSV
      - Cast features & meta
      - Validate feature integrity
      - Extract date from filename (bronze_feature_clickstream_YYYY_MM_DD.csv)
      - Write to silver_base/clickstream/silver_clickstream_<YYYY_MM_DD>.parquet
    """
    df = spark.read.csv(csv_path, header=True, inferSchema=False)
    df = cast_feature_columns(df)
    df = cast_meta_columns(df)
    validate_features(df)

    fname = os.path.basename(csv_path)
    m = re.search(r"(\d{4}_\d{2}_\d{2})", fname)
    if not m:
        raise ValueError(f"Cannot parse date from filename {fname}")
    date_str = m.group(1)

    out_dir = os.path.join(silver_base, "clickstream", f"silver_clickstream_{date_str}.parquet")
    df.repartition(1).write.mode("overwrite").parquet(out_dir)
    print(f"Wrote Clickstream → {out_dir}")
