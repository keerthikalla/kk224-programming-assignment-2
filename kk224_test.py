import random
import sys

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName("test").getOrCreate()
spark.sparkContext.setLogLevel("Error")


df = spark.read.format("csv").load("s3://winepredbucket/ValidationDataset.csv, header=True, sep=";")


df = df.toDF("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "label")

df = df \
        .withColumn("fixed_acidity", col("fixed_acidity").cast(DoubleType())) \
        .withColumn("volatile_acidity", col("volatile_acidity").cast(DoubleType())) \
        .withColumn("citric_acid", col("citric_acid").cast(DoubleType())) \
        .withColumn("residual_sugar", col("residual_sugar").cast(DoubleType())) \
        .withColumn("chlorides", col("chlorides").cast(DoubleType())) \
        .withColumn("free_sulfur_dioxide", col("free_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("total_sulfur_dioxide", col("total_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("density", col("density").cast(DoubleType())) \
        .withColumn("pH", col("pH").cast(DoubleType())) \
        .withColumn("sulphates", col("sulphates").cast(DoubleType())) \
        .withColumn("alcohol", col("alcohol").cast(DoubleType())) \
        .withColumn("label", col("label").cast(IntegerType()))


features = df.columns
features = features[:-1]

VectorAssembler = VectorAssembler(inputCols=val.columns[1:-1], outputCol='features')
df_tr = VectorAssembler.transform(val)
df_tr = df_tr.select(['features', 'label']


# rdd converted dataset
dataset = to_labeled_point(sc, features, label)

va = VectorAssembler(inputCols=features, outputCol="features")
df_va = va.transform(df)
df_va = df_va.select(["features", "label"])
df = df_va

RFModel = RandomForestModel.load(sc, "s3://winepredbucket/Trainingmodel.model")

print("model loaded successfully")
predictions = RFModel.predict(dataset.map(lambda x: x.features))

labelsAndPredictions = dataset.map(lambda lp: lp.label).zip(predictions)

labelsAndPredictions_df = labelsAndPredictions.toDF()
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show()
labelpred_df = labelpred.toPandas()

F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1- score: ", F1score)
print(confusion_matrix(labelpred_df['label'], labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'], labelpred_df['Prediction']))
print("Accuracy", accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))


