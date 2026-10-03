from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import subprocess

print("Starting Spark Session..")

spark = SparkSession.builder \
    .appName("CrimeFlowPipeline") \
    .config("spark.eventLog.enabled", "false") \
    .getOrCreate()

print("Spark Session Created Successfully")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "banking-crime-prevention"
DATASET = "BankingCrimeFlow"
BUCKET = " banking-crime-prevent-project-sample"

# GCS FILE PATHS
txn_path = f"gs://{BUCKET}/raw_files/Datafiles_transactions_raw.csv"
cust_kyc_path = f"gs://{BUCKET}/raw_files/Datafiles_customers_kyc.json"
branch_path = f"gs://{BUCKET}/raw_files/Datafiles_branch_lookup.csv"


# ============================================================
# READ TRANSACTION FILE
# ============================================================

print("Reading Transaction File..")

txn_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(txn_path)

print("TRANSACTION FILE LOADED")

txn_df.show(5)
txn_df.printSchema()


# ============================================================
# READ CUSTOMER KYC FILE
# ============================================================

print("Reading Customer File..")

customer_df = spark.read \
    .option("multiline", "true") \
    .json(cust_kyc_path)

print("CUSTOMER FILE LOADED")

customer_df.show(5)
customer_df.printSchema()


# ============================================================
# READ BRANCH FILE
# ============================================================

print("Reading Branch File..")

branch_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(branch_path)

print("BRANCH FILE LOADED")

branch_df.show(5)
branch_df.printSchema()


# ============================================================
# VALID RECORDS
# ============================================================

valid_df = txn_df.filter(
    F.col("customer_id").isNotNull()
)

print("VALID RECORDS")

valid_df.show(5)


# ============================================================
# INVALID RECORDS
# ============================================================

invalid_df = txn_df.filter(
    F.col("customer_id").isNull()
).withColumn(
    "failure_reason",
    F.lit("NULL_CUSTOMER_ID")
)

print("INVALID RECORDS")

invalid_df.show(5)


# ============================================================
# WRITE INVALID RECORDS TO QUARANTINE
# ============================================================

invalid_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(
        f"gs://{BUCKET}/quarantine/invalid_records/"
    )

print("INVALID RECORDS WRITTEN")


# ============================================================
# STAGE 2 — CLEANSE & STANDARDISE
# ============================================================

# ------------------------------------------------------------
# CLEAN AMOUNT
# ------------------------------------------------------------

clean_df = valid_df.withColumn(
    "amount",
    F.regexp_replace(
        F.col("amount"),
        "£|,",
        ""
    )
)

print("AMOUNT SYMBOLS CLEANED")

clean_df.show(5)


# Convert amount to double

clean_df = clean_df.withColumn(
    "amount",
    F.col("amount").cast("double")
)

clean_df.printSchema()


# ------------------------------------------------------------
# REFUND FLAG
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "is_refund",
    F.when(
        F.col("amount") < 0,
        True
    ).otherwise(False)
)

print("REFUND FLAG CREATED")

clean_df.show(5)


# ------------------------------------------------------------
# CLEAN CURRENCY
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "currency_unit",
    F.when(
        F.upper(F.trim(F.col("currency_unit"))) != "GBP",
        "GBP"
    ).otherwise(
        F.upper(F.trim(F.col("currency_unit")))
    )
)

print("CURRENCY CLEANED")

clean_df.show(10)


# ------------------------------------------------------------
# CLEAN CHANNEL
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "channel",
    F.upper(
        F.trim(
            F.col("channel")
        )
    )
)

print("CHANNEL CLEANED")

clean_df.show(10)


# ------------------------------------------------------------
# CLEAN TRANSACTION DATE
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "transaction_date",
    F.coalesce(
        F.to_date(
            F.col("transaction_date"),
            "dd/MM/yyyy"
        ),
        F.to_date(
            F.col("transaction_date"),
            "yyyy-MM-dd"
        )
    )
)

print("DATE CLEANSING COMPLETED")

clean_df.show(10)


# ============================================================
# DEDUPLICATION
# ============================================================

window_spec = Window \
    .partitionBy("transaction_id") \
    .orderBy(
        F.col("transaction_date")
    )

clean_df = clean_df.withColumn(
    "row_num",
    F.row_number().over(window_spec)
)

print("DUPLICATE RECORDS")

clean_df.filter(
    F.col("row_num") > 1
).show()


# Keep first record

clean_df = clean_df.filter(
    F.col("row_num") == 1
).drop("row_num")

print("Duplicates are removed")


# ============================================================
# CUSTOMER KYC CLEANING
# ============================================================

customer_df = customer_df.withColumn(
    "pep_flag",
    F.coalesce(
        F.col("pep_flag"),
        F.lit(False)
    )
)

customer_df = customer_df.withColumn(
    "risk_rating",
    F.initcap(
        F.col("risk_rating")
    )
)

print("CUSTOMER KYC CLEANING COMPLETED")

customer_df.show(10)


# ============================================================
# STAGE 3 — ENRICHMENT
# ============================================================

# ------------------------------------------------------------
# CUSTOMER / PEP ENRICHMENT
# ------------------------------------------------------------

final_df = clean_df.join(
    customer_df.select(
        "customer_id",
        "full_name",
        "risk_rating",
        "pep_flag"
    ),
    on="customer_id",
    how="left"
)

# Create PEP transaction flag

final_df = final_df.withColumn(
    "is_pep_transaction",
    F.when(
        F.col("pep_flag") == True,
        True
    ).otherwise(False)
)

print("PEP ENRICHMENT COMPLETED")

final_df.show(10)


# ============================================================
# CUSTOMER TRANSACTION WINDOW
# ============================================================

cust_window = Window.partitionBy(
    "customer_id"
)


# Transaction count

final_df = final_df.withColumn(
    "txn_count",
    F.count("transaction_id").over(
        cust_window
    )
)


# Total transaction amount

final_df = final_df.withColumn(
    "total_amount",
    F.sum("amount").over(
        cust_window
    )
)

print("TOTAL AMOUNT CALCULATED")
print("TOTAL TRANSACTION COUNT CALCULATED")

final_df.show(10)


# ============================================================
# VELOCITY RISK BAND
# ============================================================

final_df = final_df.withColumn(
    "velocity_risk_band",
    F.when(
        (
            F.col("txn_count") > 10
        ) |
        (
            F.col("total_amount") > 50000
        ),
        "HIGH"
    ).otherwise(
        "LOW"
    )
)

print("VELOCITY RISK COMPLETED")

final_df.show(5)


# ============================================================
# BRANCH LOOKUP JOIN
# ============================================================

final_df = final_df.join(
    branch_df,
    on="branch_code",
    how="left"
)

print("BRANCH JOIN COMPLETED")

final_df.show(10)


# ============================================================
# WRITE CLEANSED PARQUET
# ============================================================

parquet_output_path = (
    f"gs://{BUCKET}/cleansed/final_transactions/"
)

final_df.write \
    .mode("overwrite") \
    .parquet(parquet_output_path)

print("PARQUET WRITTEN")


# ============================================================
# PARQUET DATA VALIDATION
# ============================================================

print("Reading Final Parquet File..")

parquet_df = spark.read \
    .parquet(parquet_output_path)

parquet_df.show()

record_count = parquet_df.count()

print(
    f"Total records: {record_count}"
)


# ============================================================
# WRITE RECORD COUNT TO AUDIT
# ============================================================

count_df = spark.createDataFrame(
    [
        (record_count,)
    ],
    [
        "record_count"
    ]
)

count_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(
        f"gs://{BUCKET}/audit/"
    )

print("RECORD COUNT WRITTEN")


# ============================================================
# READ AUDIT RECORD COUNT
# ============================================================

cnt_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(
        f"gs://{BUCKET}/audit/"
    )

cnt_df.show()


# ============================================================
# STAGE 4 — LOAD TO BIGQUERY
# ============================================================

# ============================================================
# FACT TRANSACTIONS TABLE
# ============================================================

bqpath = (
    f"{PROJECT_ID}:{DATASET}.fact_transactions"
)

gcs_parquetpath = (
    f"gs://{BUCKET}/cleansed/final_transactions/*.parquet"
)

print("BigQuery Target:")
print(bqpath)

print("GCS Parquet Source:")
print(gcs_parquetpath)


# ------------------------------------------------------------
# CHECK WHETHER BQ COMMAND IS AVAILABLE
# ------------------------------------------------------------

print("Checking bq command...")

try:

    bq_check = subprocess.run(
        ["bq", "--version"],
        check=True,
        capture_output=True,
        text=True
    )

    print(bq_check.stdout)

except FileNotFoundError:

    raise Exception(
        "bq command not found. "
        "Please ensure Google Cloud SDK / bq CLI "
        "is installed and available in PATH."
    )


# ------------------------------------------------------------
# LOAD PARQUET INTO BIGQUERY
# ------------------------------------------------------------

cmd = [
    "bq",
    "load",
    "--replace",
    "--source_format=PARQUET",
    bqpath,
    gcs_parquetpath
]

print("Loading FACT TRANSACTIONS into BigQuery...")

result = subprocess.run(
    cmd,
    check=True,
    capture_output=True,
    text=True
)

print(result.stdout)

if result.stderr:
    print(result.stderr)

print("FACT TABLE LOADED")


# ============================================================
# DIM CUSTOMER TABLE
# ============================================================

TEMP_BUCKET = BUCKET

dim_customer_df = customer_df.select(
    "customer_id",
    "full_name",
    "risk_rating",
    "pep_flag",
    "country_code",
    "kyc_status",
    "onboarded_date"
).dropDuplicates()


dim_customer_df.write \
    .format("bigquery") \
    .option(
        "temporaryGcsBucket",
        TEMP_BUCKET
    ) \
    .option(
        "table",
        f"{PROJECT_ID}:{DATASET}.dim_customer"
    ) \
    .mode("overwrite") \
    .save()

print("DIM CUSTOMER LOADED")


# ============================================================
# HIGH RISK MART
# ============================================================

mart_df = final_df.filter(
    (
        F.col("is_pep_transaction") == True
    ) |
    (
        F.col("velocity_risk_band") == "HIGH"
    )
)

print("HIGH RISK RECORDS")

mart_df.show(10)


mart_df.write \
    .format("bigquery") \
    .option(
        "temporaryGcsBucket",
        TEMP_BUCKET
    ) \
    .option(
        "table",
        f"{PROJECT_ID}:{DATASET}.mart_high_risk_activity"
    ) \
    .mode("overwrite") \
    .save()

print("MART TABLE LOADED")


# ============================================================
# PIPELINE QUALITY LOG
# ============================================================

valid_count = valid_df.count()
invalid_count = invalid_df.count()

quality_log_df = spark.createDataFrame(
    [
        (
            "SUCCESS",
            valid_count,
            invalid_count
        )
    ],
    [
        "pipeline_status",
        "valid_records",
        "invalid_records"
    ]
)


quality_log_df.write \
    .format("bigquery") \
    .option(
        "temporaryGcsBucket",
        TEMP_BUCKET
    ) \
    .option(
        "table",
        f"{PROJECT_ID}:{DATASET}.pipeline_quality_log"
    ) \
    .mode("overwrite") \
    .save()

print("QUALITY LOG TABLE LOADED")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("==========================================")
print("PIPELINE COMPLETED SUCCESSFULLY")
print("==========================================")