import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table



# make sure utils/ is importable
sys.path.append("utils")
from utils.silver_attributes import process_silver_attributes
from utils.silver_financials import process_silver_financials
from utils.silver_clickstream import process_silver_clickstream



# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()




# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"



# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)


# create bronze datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)


# create bronze datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)


folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()


BRONZE = "/app/datamart/bronze"
SILVER = "/app/datamart/silver"




spark = SparkSession.builder.appName("BronzeToSilverOrchestrator").getOrCreate()

# Attributes
for fname in sorted(os.listdir(f"{BRONZE}/attributes")):
    # print(fname)
    if fname.endswith(".csv") and "bronze_features_attributes" in fname:
        print('hello')
        path = f"{BRONZE}/attributes/{fname}"
        print("→ Attr:", fname)
        print('attri', path , SILVER)
        process_silver_attributes(spark, path, SILVER)

# Financials
for fname in sorted(os.listdir(f"{BRONZE}/financials")):
    if fname.endswith(".csv") and "bronze_features_financials" in fname: 
        path = f"{BRONZE}/financials/{fname}"
        print("→ Fin:", fname)
        print('fin', path , SILVER)

        process_silver_financials(spark, path, SILVER)

# Clickstream
for fname in sorted(os.listdir(f"{BRONZE}/clickstream")):
    if fname.endswith(".csv") and "bronze_feature_clickstream" in fname:
        path = f"{BRONZE}/clickstream/{fname}"
        print("→ Click:", fname)
        process_silver_clickstream(spark, path, SILVER)

spark.stop()
print("✅ Bronze→Silver pipeline finished.")


import os, re, sys, pandas as pd
from pyspark.sql import SparkSession

# Spark session
spark = SparkSession.builder.appName("GoldFinancialOrchestrator").getOrCreate()

# utils import
sys.path.append("utils")
from utils.gold_feature_store import build_gold_for_date

silver_base = "/app/datamart/silver"
gold_root   = "/app/datamart/gold"

# Collect snapshot dates present in Silver Financials
fin_dates = {
    re.search(r"(\d{4}_\d{2}_\d{2})", f).group(1)
    for f in os.listdir(f"{silver_base}/financials")
    if f.startswith("silver_financials_") and f.endswith(".parquet")
}

print("Financial snapshot dates:", sorted(fin_dates))

# Build Gold for each date
for d in sorted(fin_dates):
    build_gold_for_date(
        spark=spark,
        silver_base=silver_base,
        gold_root=gold_root,
        date_str=d
    )

# ---- Preview latest snapshot ----
latest_folder = sorted(
    f for f in os.listdir(f"{gold_root}/feature_store")
    if f.startswith("gold_feature_financial_") and f.endswith(".parquet")
)[-1]

sample_path = f"{gold_root}/feature_store/{latest_folder}"
df_preview  = spark.read.parquet(sample_path)

row_cnt = df_preview.count()
print(f"\nRows in latest Gold snapshot ({latest_folder}): {row_cnt}")

pd.set_option("display.max_columns", None)
# display(df_preview.limit(5).toPandas())

spark.stop()


    