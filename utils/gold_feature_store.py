# utils/gold_feature_store.py
import os, re
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType, FloatType

# ------------------------------------------------------------------ #
#  Financial helpers                                                 #
# ------------------------------------------------------------------ #
def _one_hot_loans(df: DataFrame) -> DataFrame:
    df = df.withColumn(
        "loan_list",
        F.split(F.regexp_replace("Type_of_Loan", r"\s+and\s+", ","), ",\s*")
    )
    loans = (
        df.select(F.explode("loan_list").alias("raw"))
          .select(F.trim("raw").alias("loan"))
          .filter("loan != ''")
          .distinct()
          .rdd.flatMap(lambda r: r)
          .collect()
    )
    for loan in loans:
        safe = "loan_" + re.sub(r"[^A-Za-z0-9]", "_", loan).lower()
        df = df.withColumn(safe, F.array_contains("loan_list", loan).cast("int"))
    return df.drop("loan_list")

def _credit_history_to_months(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("years_part",
            F.regexp_extract("Credit_History_Age", r"(\d+)\s*Years", 1).cast(IntegerType())
        )
        .withColumn("months_part",
            F.regexp_extract("Credit_History_Age", r"(\d+)\s*Months", 1).cast(IntegerType())
        )
        .fillna({"years_part": 0, "months_part": 0})
        .withColumn("Credit_History_Months",
            F.col("years_part")*12 + F.col("months_part")
        )
        .drop("years_part", "months_part")
    )

def _prepare_financials(df_fin: DataFrame) -> DataFrame:
    df_fin = _one_hot_loans(df_fin)
    df_fin = _credit_history_to_months(df_fin)
    return df_fin

# ------------------------------------------------------------------ #
#  Click-stream helper                                               #
# ------------------------------------------------------------------ #
def _dedup_click(df_cs: DataFrame) -> DataFrame:
    aggs = [F.mean(F.col(f"fe_{i}")).alias(f"fe_{i}") for i in range(1, 21)]
    return df_cs.groupBy("Customer_ID", "snapshot_date").agg(*aggs)

# ------------------------------------------------------------------ #
#  Build Gold for a single date                                      #
# ------------------------------------------------------------------ #
def build_gold_for_date(
    spark: SparkSession,
    silver_base: str,
    gold_root: str,
    date_str: str,
) -> DataFrame:
    """
    Build Gold feature store anchored on Financials for one snapshot date.
    Parquet is written to:  <gold_root>/feature_store/gold_feature_financial_<date>.parquet
    """
    fin_path  = f"{silver_base}/financials/silver_financials_{date_str}.parquet"
    attr_path = f"{silver_base}/attributes/silver_attributes_{date_str}.parquet"
    cs_path   = f"{silver_base}/clickstream/silver_clickstream_{date_str}.parquet"

    # 1. Load mandatory Financials
    fin = _prepare_financials(spark.read.parquet(fin_path))

    # 2. Optional left-join tables
    attrs  = spark.read.parquet(attr_path) if os.path.exists(attr_path) else None
    cs_raw = spark.read.parquet(cs_path)   if os.path.exists(cs_path)   else None
    clicks = _dedup_click(cs_raw) if cs_raw else None

    gold = fin
    if attrs:
        gold = gold.join(attrs, ["Customer_ID", "snapshot_date"], "left")
    if clicks:
        gold = gold.join(clicks, ["Customer_ID", "snapshot_date"], "left")

    # 3. Impute defaults
    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    gold = gold.fillna(0, subset=fe_cols) \
               .fillna({"Age": 0, "Credit_History_Months": 0})

    # 4. Example engineered metric
    gold = gold.withColumn(
        "Debt_to_Income_Ratio",
        F.when(
            F.col("Annual_Income") > 0,
            F.col("Outstanding_Debt") / F.col("Annual_Income")
        ).otherwise(F.lit(None).cast(FloatType()))
    )

    # 5. Write to feature_store directory
    out_dir = f"{gold_root}/feature_store/gold_feature_financial_{date_str}.parquet"
    gold.repartition(1).write.mode("overwrite").parquet(out_dir)
    print(f"✅  Gold financial feature store written → {out_dir}")
    return gold


