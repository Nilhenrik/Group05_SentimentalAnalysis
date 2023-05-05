import re
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.base import *
from delta import configure_spark_with_delta_pip, DeltaTable
import sparknlp
from typing import List
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from delta.tables import DeltaTable
import random
from pyspark.sql.functions import expr

def remove_forwarded_content(text):
    if not isinstance(text, str):
        return text

    forwarded_pattern = re.compile(r"(\-{2,}\s*Forwarded by)|(^\s*From:.*\n?.*on.*$)|(^\s*Sent from my.*$)", re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(forwarded_pattern, "", text)
    return cleaned_text

def binary_sentiment(sentiment: List[str]) -> int:
    if not sentiment:
        return 0
    if sentiment[0] == "positive":
        return 1
    elif sentiment[0] == "negative":
        return -1
    else:
        return 0

def basic_df_example(spark: SparkSession) -> None:
    schema = StructType() \
                        .add("id", StringType())\
                        .add("day",StringType())\
                        .add("month",StringType())\
                        .add("text", StringType())

    delta_table_schema = StructType() \
                        .add("id", StringType())\
                        .add("day",StringType())\
                        .add("season",StringType())\
                        .add("words", StringType())\
                        .add("characters", StringType())\
                        .add("sentiment", StringType())\
    
    df = spark.read.option("delimiter", ";").csv("hdfs:///src/output1/part*",schema=schema)
    #df = df.union(df)
    df = df.filter(col('day').isin(['mon', 'tue', 'wed', 'thu', 'fri','sat','sun']))
    season = spark.createDataFrame([
        ('jan', 'winter'), ('feb', 'winter'), ('mar', 'spring'), ('apr', 'spring'),
        ('may', 'spring'), ('jun', 'summer'), ('jul', 'summer'), ('aug', 'summer'),
        ('sep', 'autumm'), ('oct', 'autumm'), ('nov', 'autumm'), ('dec', 'winter'),]
        , ['month', 'season'])
    
    remove_forwarded_content_udf = udf(remove_forwarded_content, StringType())
    lower_udf = udf(lambda x: x.lower(), StringType())

    df = df.withColumn("text", remove_forwarded_content_udf(col("text")))
    df = df.withColumn("month", lower_udf(col("month")))

    # Removes the rows with empty text and id column
    df = df.filter((col("text") != "") | (col("id") != ""))

    df = df.dropDuplicates(['id'])
   
    # Define the Spark NLP pipeline with ViveknSentimentModel
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized")

    vivekn_sentiment = ViveknSentimentModel.pretrained() \
        .setInputCols(["document", "normalized"]) \
        .setOutputCol("sentiment")

    finisher = Finisher() \
        .setInputCols(["sentiment"]) \
        .setOutputCols(["final_sentiment"])

    pipeline = Pipeline().setStages([document_assembler, tokenizer, normalizer, vivekn_sentiment, finisher])

    # Fit and transform the data using the pipeline
    pipeline_model = pipeline.fit(df)
    result = pipeline_model.transform(df)
    binary_sentiment_udf = udf(binary_sentiment, IntegerType())
    result = result.withColumn("binary_sentiment", binary_sentiment_udf(col("final_sentiment")))
    result = result.withColumn('words', size(split(col('text'), '\s+')))
    result = result.withColumn('characters', length(col('text')))

    # Join result and season, then groupby
    result = result.join(season, result.month == season.month, 'full')
    partResult = (result.groupBy(spark_partition_id()).count().orderBy(desc("count")))
    partResult.show()
    delta_table = DeltaTable.createIfNotExists(spark).location("/src/saved").addColumns(delta_table_schema).execute()
    result.createOrReplaceTempView("result_view")
    delta_table.alias("delta_table").merge(
        result.alias("result_view"),
        "delta_table.id = result_view.id"
    ).whenMatchedUpdate(
        set={"sentiment": "result_view.binary_sentiment",}
    ).whenNotMatchedInsert(
        values={"id": "result_view.id",
                "day": "result_view.day",
                "season": "result_view.season",
                "words": "result_view.words",
                "characters": "result_view.characters",
                "sentiment": "result_view.binary_sentiment"}
    ).execute()
    delta_df = delta_table.toDF()

    grouped = delta_df.groupby('day', 'season').agg(
        count('day').alias('count'),
        sum('words').alias('sum_words'),
        sum('characters').alias('sum_characters'),
        sum('sentiment').alias('sum_sentiment')
    )
    grouped.sort("day", "season").show(grouped.count())
    return
    
    

if __name__ == "__main__":
    # The rest of your SparkSession code and basic_df_example() call
    builder = SparkSession.builder \
    .appName("Non Opt - 2 cores - 6 instances")\
    .master("yarn")\
    .config("spark.executor.instances", "6")\
    .config("spark.executor.cores", "2")\
    .config("spark.executor.memory", "2G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")\
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")


    spark = configure_spark_with_delta_pip(builder, ["com.johnsnowlabs.nlp:spark-nlp_2.12:4.4.0"]).getOrCreate()
    # $example off:init_session$

    basic_df_example(spark)

    spark.stop()
