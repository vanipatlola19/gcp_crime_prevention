#Stage 1 — Ingest & Validate
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

print('Starting Spark Session..')

spark = SparkSession.builder \
        .appName('CrimeFlowPipeline') \
        .config('spark.eventLog.enabled','false') \
        .getOrCreate()
print('Spark Session Created Successfully')

#CONFIG
PROJECT_ID = 'banking-crimeprevention'
DATASET = 'BankingCrimeFlow'
BUCKET = 'banking_crime-prevent-project'

#GCS FILE PATHS
txn_path = ('gs://banking_crime-prevent-project/raw_files/Datafiles_transactions_raw.csv')
cust_kyc_path = ('gs://banking_crime-prevent-project/raw_files/Datafiles_customers_kyc.json')
branch_path = ('gs://banking_crime-prevent-project/raw_files/Datafiles_branch_lookup.csv')

#READ TRANSACTION FILE
print('Reading Transaction File..')
txn_df = spark.read \
        .option('header','true') \
        .option('inferschema','true') \
        .csv(txn_path)

print('TRANSACTION FILE LOADED')
txn_df.show(5)
txn_df.printSchema()

# READ CUSTOMER FILE
print('Reading Customer File..')
customer_df = spark.read \
            .option('multiline','true') \
            .json(cust_kyc_path)

print('CUSTOMER FILE LOADED')
customer_df.show(5)
customer_df.printSchema()

# READ BRANCH FILE
print('Reading Branch File..')
branch_df = spark.read \
            .option('header','true') \
            .option('inferschema','true') \
            .csv(branch_path)
print('BRANCH FILE LOADED')
branch_df.show(5)
branch_df.printSchema()

#VALID RECORDS
valid_df = txn_df.filter(F.col('customer_id').isNotNull())
valid_df.show(5)

#INVALID RECORDS
invalid_df = txn_df.filter(F.col('customer_id').isNull()) \
                    .withColumn('failure_reason',F.lit('NULL_CUSTOMER_ID'))
invalid_df.show(5)

#WRITE INVALID RECORDS
invalid_df.write.mode('overwrite').option('header','true').csv(f"gs://{BUCKET}/quarantine/invalid_records/")
print('INVALID RECORDS WRITTEN')

#Stage 2 — Cleanse & Standardise
#CLEAN AMOUNT
clean_df = valid_df.withColumn('amount',F.regexp_replace(F.col('amount'),'£|,',''))
clean_df.show(5)
print('INVALID SYMBOLS FROM AMOUNT COLUMN ARE REMOVED')
#clean_df.printSchema()

clean_df = clean_df.withColumn('amount',F.col('amount').cast('double'))
clean_df.printSchema()

#clean_df = clean_df.withColumn('amount_gbp',abs(F.col('amount')))
clean_df.show(5)

clean_df = clean_df.withColumn('is_refund',F.when(F.col('amount')<0, True).otherwise(False))
clean_df.show(5)
clean_df.filter(F.col('is_refund')=='true').show()

#CLEAN CURRENCY
clean_df=clean_df.withColumn('currency_unit',
                    F.when(F.upper(F.trim(F.col('currency_unit')))!='GBP','GBP').otherwise(F.upper(F.trim(F.col('currency_unit'))))
                   )
clean_df.show(10)

#CLEAN CHANNEL
clean_df=clean_df.withColumn('channel',F.upper(F.trim(F.col('channel'))))
clean_df.show(10)

#CLEAN DATE
clean_df=clean_df.withColumn('transaction_date',
                    F.coalesce(
                        F.to_date(F.col('transaction_date'),'dd/MM/yyyy'),
                        F.to_date(F.col('transaction_date'),'yyyy-MM-dd')
                    ))
clean_df.show(10)
print('DATE CLEANSING COMPLETED')

#DEDUPLICATION(REMOVE DUPLICATES)
window_spec = Window.partitionBy('transaction_id').orderBy(F.col('transaction_date'))
clean_df = clean_df.withColumn('row_num',F.row_number().over(window_spec))
clean_df.filter(F.col('row_num') >1 ).show()

clean_df = clean_df.filter(F.col('row_num') == 1).drop('row_num')
print('Duplicates are removed')

#CUSTOMER_KYC CLEANING
customer_df=customer_df.withColumn('pep_flag',F.coalesce(F.col('pep_flag'),F.lit(False)))
customer_df=customer_df.withColumn('risk_rating',F.initcap(F.col('risk_rating')))
customer_df.show(10)

#Stage 3 — Enrich
#PEP ENRICHMENT
final_df = clean_df.join(
customer_df.select('customer_id','full_name','risk_rating','pep_flag'),on='customer_id',how='left')

final_df=final_df.withColumn('is_pep_transaction',F.when(F.col('pep_flag') == True, True).otherwise(False))
final_df.show(10)

#WINDOW FUNCTION
cust_window = Window.partitionBy('customer_id')
final_df = final_df.withColumn('txn_count',F.count('transaction_id').over(cust_window))

final_df = final_df.withColumn('total_amount',F.sum('amount').over(cust_window))
final_df.show(10)
print('total amount calculated')
print('total transaction count calculated')

#VELOCITY RISK BAND
final_df = final_df.withColumn('velocity_risk_band',
                    F.when(
                            (F.col('txn_count') > 10) |
                            (F.col('total_amount') > 50000), 'HIGH')
                            .otherwise('LOW')
                   )
final_df.show(5)
print('VELOCITY RISK COMPLETED')

#JOIN BRANCH LOOKUP
final_df = final_df.join(branch_df,on='branch_code',how='left')
final_df.show(10)
print('BRANCH JOIN COMPLETED')

#WRITE CLEANSED PARQUET
final_df.write.mode('overwrite') \
        .parquet(f"gs://{BUCKET}/cleansed/final_transactions/")
print('PARQUET WRITTEN')

#PARQUET FILE DATA VALIDATION
print('Reading Final Parquet File..')
parquet_df = spark.read \
        .parquet(f"gs://{BUCKET}/cleansed/final_transactions/")
parquet_df.show()

record_count = parquet_df.count()
print(f"Total records: {record_count}")

count_df=spark.createDataFrame(
        [(record_count,)],
        ['record_count'])
count_df.write \
    .mode('overwrite') \
    .option('header','true') \
    .csv(f"gs://{BUCKET}/audit/")
print('record count written')

cnt_df = spark.read \
        .option('header','true') \
        .option('inferschema','true') \
        .csv(f"gs://{BUCKET}/audit/")
cnt_df.show()

#Stage 4 — Load to BigQuery
#Step 1: Load fact table
#FACT TRANSACTIONS TABLE
bqpath = f"{PROJECT_ID}:{DATASET}.fact_transactions"
gcs_parquetpath = f"gs://{BUCKET}/cleansed/final_transactions/*.parquet"
print(bqpath)
!bq load \
    --replace \
      --source_format=PARQUET \
      {bqpath} \
    {gcs_parquetpath}
print("FACT TABLE LOADED")

#DIM CUSTOMER TABLE
TEMP_BUCKET = "banking_crime-prevent-project"   # Bucket name only, no gs://

dim_customer_df = customer_df.select(
                'customer_id',
                'full_name',
                'risk_rating',
                'pep_flag',
                'country_code',
                'kyc_status',
                'onboarded_date').dropDuplicates()
dim_customer_df.write.format('bigquery') \
                .option("temporaryGcsBucket", TEMP_BUCKET) \
                .option('table',f"{PROJECT_ID}:{DATASET}.dim_customer") \
                .mode('overwrite').save()
print('DIM CUSTOMER LOADED')

#HIGH RISK MART
mart_df = final_df.filter(
    (F.col('is_pep_transaction') == True) |
    (F.col('velocity_risk_band') == 'High')
)

mart_df.write.format('bigquery') \
                .option("temporaryGcsBucket", TEMP_BUCKET) \
                .option('table',f"{PROJECT_ID}:{DATASET}.mart_high_risk_activity") \
                .mode('overwrite').save()
print('MART TABLE LOADED')

#QUALITY LOG TABLE
quality_log_df = spark.createDataFrame(
    [
        (
            "SUCCESS",
            valid_df.count(),
            invalid_df.count()
        )
    ],
    [
        'pipeline_status',
        'valid_records',
        'invalid_records'
    ]
)

quality_log_df.write.format('bigquery') \
                .option("temporaryGcsBucket", TEMP_BUCKET) \
                .option('table',f"{PROJECT_ID}:{DATASET}.pipeline_quality_log") \
                .mode('overwrite').save()
print('QUALITY LOG TABLE LOADED')
print('PIPELINE COMPLETED SUCCESSFULLY')