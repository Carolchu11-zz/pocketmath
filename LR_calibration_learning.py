# Calibration with training set and testing set

import re
from math import log
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import IsotonicRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SQLContext
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *

sc = SparkContext("local", "LR calibration")
sqlContext = SQLContext(sc)


def replace(value):
    if value == 'WIN':
        return 0
    else:
        return 1


def unique(value):
    u = re.sub('-[\d\w]+', '', value)
    v = list(set(u.split(',')))
    return v


def data_prepare(datapath, schema):
    f = sqlContext.read.csv(datapath, header='false', mode="DROPMALFORMED", schema=schema)
    uf = UserDefinedFunction(lambda x: replace(x), IntegerType())
    f = f.select(*[uf(column).alias('event') if column == 'event' else column for column in f.columns])

    # Subtract sublist of column 'order_iab-cat'
    uf1 = UserDefinedFunction(lambda x: unique(x), ArrayType(StringType()))
    f = f.select(*[uf1(column).alias('order_iab_cat') if column == 'order_iab_cat' else column for column in f.columns])

    # Replace null value by 'NA'
    f = f.fillna('NA')

    # Use ChiSqSelector for selecting features
    # selector = ChiSqSelector(numTopFeatures=20000, outputCol="selectedFeatures")
    # f = selector.fit(f).transform(f)
    # f = f.select(["label", "selectedFeatures"])

    return f


def split(data):
    win_sample = data.where(data['event'] == 0).sample(withReplacement=False, fraction=0.1)
    click_sample = data.where(data['event'] == 1)

    f = win_sample.unionAll(click_sample)

    # Convert column 'order_iab_cat' to one-hot encoding column
    a = f.select('order_iab_cat').rdd.map(lambda r: r[0]).collect()
    b = [item for sublist in a for item in sublist]

    order_iab_cats = list(set(b))
    order_iab_cats_expr = [
        F.when(F.array_contains(F.col("order_iab_cat"), order_iab_cat), 1).otherwise(0).alias("e_" + order_iab_cat) for
        order_iab_cat in order_iab_cats]

    f = f.select('event', 'io_id', 'creative_id', 'creative_version', 'exchange_id', 'app_bundle', 'norm_device_make',
                 'norm_device_model', 'norm_device_os', 'location_id', 'geo_country', *order_iab_cats_expr)

    # Convert all columns except 'order_iab_cat' to one-hot encoding column
    column_vec_in = ['io_id', 'creative_id', 'creative_version', 'exchange_id', 'app_bundle',
                     'norm_device_make', 'norm_device_model', 'norm_device_os', 'location_id', 'geo_country']
    column_vec_out = ['io_id_catVec', 'creative_id_catVec', 'creative_version_catVec', 'exchange_id_catVec',
                      'app_bundle_catVec',
                      'norm_device_make_catVec', 'norm_device_model_catVec', 'norm_device_os_catVec',
                      'location_id_catVec', 'geo_country_catVec']

    indexers = [StringIndexer(inputCol=x, outputCol=x + '_tmp')
                for x in column_vec_in]

    encoders = [OneHotEncoder(dropLast=False, inputCol=x + "_tmp", outputCol=y)
                for x, y in zip(column_vec_in, column_vec_out)]

    tmp = [[i, j] for i, j in zip(indexers, encoders)]
    tmp = [i for sublist in tmp for i in sublist]

    # Concatenate all feature columns into featuresCol(Vector)
    order_iab_cats_ = ['e_' + item for item in order_iab_cats]
    vectorAsCols = ['io_id_catVec', 'creative_id_catVec', 'creative_version_catVec'] + order_iab_cats_ + [
        'exchange_id_catVec', 'app_bundle_catVec', 'norm_device_make_catVec',
        'norm_device_model_catVec', 'norm_device_os_catVec', 'location_id_catVec', 'geo_country_catVec']
    vectorAssembler = VectorAssembler(inputCols=vectorAsCols, outputCol="features")
    labelIndexer = StringIndexer(inputCol='event', outputCol="label")
    tmp += [vectorAssembler, labelIndexer]
    pipeline = Pipeline(stages=tmp)

    f = pipeline.fit(f).transform(f)
    f.cache()

    win = f.where(f['label'] == 0)
    click = f.where(f['label'] == 1)

    # Split dataset into two part, training set for training model, validate set for training calibration model
    train_win, test_win = win.randomSplit([4.0, 1.0], seed=None)
    train_click, test_click = click.randomSplit([4.0, 1.0], seed=None)
    train, test = train_win.unionAll(train_click), test_win.unionAll(test_click)

    return train, test


def model_prepare(train):
    # Logistic regression with no calibration
    lr = LogisticRegression()

    # Cross Validation and return cvModel
    paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).addGrid(lr.elasticNetParam,
                                                                             [0.0, 0.25, 0.5, 0.75, 1.0]).build()
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator().setEstimator(lr).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(4)

    cvModel = cv.fit(train)

    return cvModel


def computeLogLoss(p, y):
    """Calculates the value of log loss for a given probabilty and label.

    Note:
        log(0) is undefined, so when p is 0 we need to add a small value (epsilon) to it
        and when p is 1 we need to subtract a small value (epsilon) from it.

    Args:
        p (float): A probabilty between 0 and 1.
        y (int): A label.  Takes on the values 0 and 1.

    Returns:
        float: The log loss value.
    """
    epsilon = 10e-12
    if p == 0:
        p += epsilon
    elif p == 1:
        p -= epsilon
    if y == 1:
        return -log(p)
    else:
        return -log(1 - p)


def test_model(model, test):
    evaluator = BinaryClassificationEvaluator()

    # Compute ROC area under curve, and log loss before calibration
    transformed = model.transform(test)
    roc_auc = evaluator.evaluate(transformed, {evaluator.metricName: "areaUnderROC"})

    logloss = transformed.select(['label', 'probability']).rdd.map(
        lambda row: computeLogLoss(row.probability[1], row.label)).mean()

    return transformed, roc_auc, logloss


# Training isotonic regression model by using logistic regression model and training set
def calibration(model, transformed, train):
    # Apply trained logistic regression model in training set and get transform
    transformed_train = model.transform(train)

    # Logistic regression with isotonic calibration
    # Create dataframe by selecting label from transformed_train and and probability as features to train isotonic regression model
    uf = UserDefinedFunction(lambda x: Vectors.dense(x[1]), VectorUDT())

    df = transformed_train.selectExpr("label as label", "probability as features")
    df = df.select(*[uf(column).alias('features') if column == 'features' else column for column in df.columns])
    lr_isotonic = IsotonicRegression()
    model_isotonic = lr_isotonic.fit(df)

    # Compute ROC area under curve and log loss after isotonic calibration
    # Create training set for isotonic regression model by selecting probability as features
    df1 = transformed.selectExpr("label as label", "probability as features")
    df1 = df1.select(*[uf(column).alias('features') if column == 'features' else column for column in df1.columns])
    transformed_isotonic = model_isotonic.transform(df1)

    predictionAndLabels = transformed_isotonic.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))
    metrics = BinaryClassificationMetrics(predictionAndLabels)
    roc_auc_isotonic = metrics.areaUnderROC

    logloss_isotonic = transformed_isotonic.select(['label', 'prediction']).rdd.map(
        lambda row: computeLogLoss(row.prediction, row.label)).mean()

    return transformed_isotonic, roc_auc_isotonic, logloss_isotonic


def binScore(value):
    value *= 10000
    value = int(value)

    if value <= 100:
        return value / 5 * 5

    elif value <= 200:
        return value / 10 * 10

    elif value <= 500:
        return value / 20 * 20

    else:
        return value / 100 * 100


def bucket(transformed):
    uf = UserDefinedFunction(lambda x: binScore(x[1]), IntegerType())
    df = transformed.select(
        *[uf(column).alias('probability') if column == 'probability' else column for column in transformed.columns])
    df = df.selectExpr("label as label", "probability as pred_CTR")
    df = df.groupBy("pred_CTR").agg(F.sum("label").alias("#click"),
                                    (F.count("label") - F.sum("label")).alias("#impr"))
    predictionBins = df.withColumn("real_CTR", df["#click"] / (df["#click"] + df["#impr"]))
    uf1 = UserDefinedFunction(lambda x: int(x * 10000), IntegerType())
    predictionBins = predictionBins.select(
        *[uf1(column).alias('real_CTR') if column == 'real_CTR' else column for column in predictionBins.columns])

    return predictionBins


def bucket_isotonic(transformed):
    uf = UserDefinedFunction(lambda x: binScore(x), IntegerType())
    df = transformed.select(
        *[uf(column).alias('prediction') if column == 'prediction' else column for column in transformed.columns])
    df = df.selectExpr("label as label", "prediction as pred_CTR")
    df = df.groupBy("pred_CTR").agg(F.sum("label").alias("#click"),
                                    (F.count("label") - F.sum("label")).alias("#impr"))
    predictionBins = df.withColumn("real_CTR", df["#click"] / (df["#click"] + df["#impr"]))
    uf1 = UserDefinedFunction(lambda x: int(x * 10000), IntegerType())
    predictionBins = predictionBins.select(
        *[uf1(column).alias('real_CTR') if column == 'real_CTR' else column for column in predictionBins.columns])

    return predictionBins


if __name__ == '__main__':
    schema = StructType([
        StructField("event", StringType()),
        StructField("io_id", StringType()),
        StructField("creative_id", StringType()),
        StructField("creative_version", StringType()),
        StructField("order_iab_cat", StringType()),
        StructField("exchange_id", StringType()),
        StructField("app_bundle", StringType()),
        StructField("norm_device_make", StringType()),
        StructField("norm_device_model", StringType()),
        StructField("norm_device_os", StringType()),
        StructField("location_id", StringType()),
        StructField("geo_country", StringType()),
    ])

    f = data_prepare('/Users/carol/Documents/data/from_hermes_raw/part-00000', schema)
    # f = data_prepare('/Users/carol/Downloads/part-00000.csv', schema)
    train, test = split(f)
    cvModel = model_prepare(train)
    transformed, roc_auc, logloss = test_model(cvModel, test)
    ctr_pre = bucket(transformed)

    print("ROC area before calibration: %f" % roc_auc)
    print("Log loss before calibration: %f" % logloss)

    # Show bucket probability before calibration
    print ctr_pre.orderBy("pred_CTR").show()

    transformed_isotonic, roc_auc_isotonic, logloss_isotonic = calibration(cvModel, transformed, train)
    ctr_pre_isotonic = bucket_isotonic(transformed_isotonic)
    print("ROC area after calibration: %f" % roc_auc_isotonic)
    print("Log loss after calibration: %f" % logloss_isotonic)

    # Show bucket probability after calibration
    print ctr_pre_isotonic.orderBy("pred_CTR").show()