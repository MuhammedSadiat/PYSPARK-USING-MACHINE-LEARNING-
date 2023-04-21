This is a Python code that uses PySpark to train a random forest classifier on a dataset of Amazon product reviews.
Here are step-by-step instructions on how to use this application:

Firstly, Install PySpark: PySpark is a Python library that allows you to run Apache Spark on your local machine. You can install PySpark using pip by running the following command in your terminal or command prompt:pip install pyspark

Secondly, Import the necessary libraries: Before you can use PySpark in your Python code, you need to import the necessary libraries. Use the following code at the beginning of your Python script:
import logging
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

Thirdly,configure logging: This step is optional. If you want to suppress log messages from Py4j, add the following code:
logger = logging.getLogger("py4j")
logger.setLevel(logging.ERROR)

Fourthly,load dataset: In this step, you load the dataset into a PySpark dataframe. Use the following code:
df = spark.read.format("csv").option("header", "false").option("delimiter", "\t").load("amazon_cells_labelled.txt")
The "csv" format option specifies that the input file is a CSV file. The "header" option specifies that the input file does not have a header row. The "delimiter" option specifies that the fields in the input file are separated by a tab character.

Fifth,rename the columns and cast the labels to integer type: In this step, you rename the columns of the PySpark dataframe and cast the labels to integer type. Add the following code:
df = df.toDF("text", "label")
df = df.withColumn("label", col("label").cast("int"))

Sixth,tokenize the text data: In this step, you tokenize the text data by splitting each review into a list of words. Use the following code:
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df = tokenizer.transform(df)

Seventh,remove stop words from the text data: In this step, you remove stop words from the text data. Stop words are words that are commonly used in a language but do not carry much meaning, such as "the" and "and". Use the following code:
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="wordsfiltred")
df = stopwords_remover.transform(df)

Eighth,compute the term frequency of the filtered words: In this step, you compute the term frequency of the filtered words. Term frequency is the number of times a word appears in a document. Use the following code:
hashing_tf = HashingTF(inputCol="wordsfiltred", outputCol="rawfeatures")
df = hashing_tf.transform(df)

Nineth,compute the inverse document frequency of the raw features: In this step, you compute the inverse document frequency of the raw features. Inverse document frequency is a measure of how much information a word provides, based on how often it appears in a collection of documents. Use the following code:
idf = IDF(inputCol="rawfeatures", outputCol="features")
idf_model = idf.fit(df)
df = idf_model.transform(df)

Tenth,split the data into training and testing sets: In this step, you split the data into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate the performance of the model. Use the following code:

(training_data, testing_data) = df.randomSplit([0.8, 0.2], seed=42)
The randomSplit() method randomly splits the data into training and testing sets, with a 80:20 ratio. The seed parameter is set to 42 to ensure reproducibility of the results.

Eleventh,train a random forest classifier: In this step, you train a random forest classifier on the training data. Use the following code:

random_forest = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=4)
rforest_model = random_forest.fit(training_data)
The RandomForestClassifier() method creates an instance of a random forest classifier with 100 trees and a maximum depth of 4. The fit() method trains the model on the training data.

Lastly,evaluate the performance of the random forest classifier on the testing data: In this step, you evaluate the performance of the random forest classifier on the testing data. Use the following code:
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(rforest_model.transform(testing_data))
print("Accuracy: {:.1f}%".format(accuracy * 100))

The MulticlassClassificationEvaluator() method creates an instance of a multi-class classification evaluator with the label column set to "label", the prediction column set to "prediction", and the metric set to "accuracy". The evaluate() method computes the accuracy of the model on the testing data. Finally, the accuracy is printed to the console. I got an accuracy of 57.4%.

TO EXECUTE THE CODE: Justcopy and paste it into a Python script file, and run the file using a Python interpreter. Make sure that the input dataset file "amazon_cells_labelled.txt" is in the same directory as the Python script file.

I used Jupyterhub to execute mine.