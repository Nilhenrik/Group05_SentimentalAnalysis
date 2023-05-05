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

def basic_df_example(spark: SparkSession) -> None:
    schema = StructType() \
                        .add("id", StringType())\
                        .add("day",StringType())\
                        .add("month",StringType())\
                        .add("text", StringType())

    delta_table_schema = StructType() \
                    .add("id", StringType())\
                    .add("day", StringType())\
                    .add("season", StringType())\
                    .add("words", StringType())\
                    .add("characters", StringType())\
                    .add("sentiment", StringType())
    
    df = spark.read.option("delimiter", ";").csv("hdfs:///src/output1/part*",schema=schema)
    #df = df.union(df)
    df = df.filter(col('day').isin(['mon', 'tue', 'wed', 'thu', 'fri','sat','sun']))
    season = spark.createDataFrame([
        ('jan', 'winter'), ('feb', 'winter'), ('mar', 'spring'), ('apr', 'spring'),
        ('may', 'spring'), ('jun', 'summer'), ('jul', 'summer'), ('aug', 'summer'),
        ('sep', 'autumm'), ('oct', 'autumm'), ('nov', 'autumm'), ('des', 'winter'),]
        , ['month', 'season'])
    
    # Non udf
    #generate_salt_udf = udf(generate_salt, StringType())
    forwarded_pattern = r"(\-{2,}\s*Forwarded by)|(^\s*From:.*\n?.*on.*$)|(^\s*Sent from my.*$)"
    df = df.withColumn("text", 
                   split(regexp_replace("text", forwarded_pattern, ""), "\n\n")[0])
    df = df.withColumn("text", regexp_replace(col("text"), "^<html>.*", ""))
    df = df.withColumn("text", regexp_replace(col("text"), "[\t]", ""))
    df = df.withColumn("month", lower(col('month')))
    
    # Removes the rows with empty text column
    #df = df.filter(col("text") != "" & col("id") != "")
    df = df.filter((col("text") != "") | (col("id") != ""))
    
    df = df.dropDuplicates(['id'])
    num_partitions = 24
    df = df.repartition(num_partitions)
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
     # Set the desired number of partitions
    # Add the salt column after transforming the data using the pipeline
    #result = result.withColumn("salt", generate_salt_udf(lit(3)))
    result = result.withColumn("binary_sentiment", 
                            when(col("final_sentiment")[0] == "positive", 1)
                            .when(col("final_sentiment")[0] == "negative", -1)
                            .otherwise(0))

    result = result.withColumn('words', size(split(col('text'), '\s+')))
    result = result.withColumn('characters', length(col('text')))
    
    # Join result and season, then groupby
    result = result.join(season, result.month == season.month, 'full')

    # Remove the salt column before grouping
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
    .appName("Serialization with partition Opt - 2 cores - 6 instances")\
    .master("yarn")\
    .config("spark.executor.instances", "6")\
    .config("spark.executor.memory", "2G")\
    .config("spark.executor.cores", "2")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")\
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

    spark = configure_spark_with_delta_pip(builder, ["com.johnsnowlabs.nlp:spark-nlp_2.12:4.4.0"]).getOrCreate()

    basic_df_example(spark)

    spark.stop()
