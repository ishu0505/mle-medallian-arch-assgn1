

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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType



# utils/bronze_generic.py
def process_bronze_table(snapshot_date_str: str,
                         input_csv_path: str,
                         bronze_dir: str,
                         spark):
    """
    Reads input_csv_path, filters to snapshot_date_str, and writes out
    a bronze CSV in bronze_dir.
    """
    # parse date
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # read & filter
    df = (
        spark.read
             .csv(input_csv_path, header=True, inferSchema=True)
             .filter(F.col("snapshot_date") == snapshot_date)
    )
    print(f"{snapshot_date_str} row count:", df.count())

    # ensure output directory exists
    os.makedirs(bronze_dir, exist_ok=True)

    # build output filename
    file_name = f"bronze_{os.path.splitext(os.path.basename(input_csv_path))[0]}_{snapshot_date_str.replace('-', '_')}.csv"
    out_path = os.path.join(bronze_dir, file_name)

    # write out
    df.toPandas().to_csv(out_path, index=False)
    print("Saved to:", out_path)

    return df


def backfill_bronze_dates(dates: list[str],
                          csv_paths: list[str],
                          bronze_dirs: list[str],
                          spark):
    """
    For each (input_csv, bronze_dir) pair, and for each date in dates,
    call process_bronze_table. Returns a dict keyed by (csv, date).
    """
    if len(csv_paths) != len(bronze_dirs):
        raise ValueError("csv_paths and bronze_dirs must be same length")

    results = {}
    for inp_csv, out_dir in zip(csv_paths, bronze_dirs):
        # for each date, process and store the resulting DataFrame
        for d in dates:
            df = process_bronze_table(d, inp_csv, out_dir, spark)
            results[(inp_csv, d)] = df

    return results