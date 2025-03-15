from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

from pyspark.sql.functions import size, sum as sum_

# Initialize Spark session
spark = SparkSession.builder.appName("TFIDFExample").getOrCreate()

# Sample data: a DataFrame with document IDs and text
data = [
    (0, "spark is a unified analytics engine"),
    (1, "it provides high-level APIs"),
    (2, "spark is fast and general-purpose cluster computing")
]

data = [
    (0, "spark is a unified analytics engine"),
    (1, "it provides high-level APIs"),
    (2, "spark is fast and general-purpose cluster computing"),
    (3, "new_word_1 new_word_2 new_word_3 new_word_4 new_word_5")
]

df = spark.createDataFrame(data, ["id", "text"])

# Tokenize the text column into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df)

print(type(wordsData))
print(wordsData.columns)
wordsData.show(truncate=False)

# Convert words to term frequency features using HashingTF
#hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
#hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")


# Calculate total word count by summing the size of each array in "words"
total_words = wordsData.select(sum_(size("words")).alias("total_words"))
total_words.show()
total_words_count = total_words.collect()[0].total_words
print(type(total_words_count))


hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=int(total_words_count))

featurizedData = hashingTF.transform(wordsData)

# Compute the IDF (Inverse Document Frequency) values and transform the data
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

print(rescaledData.columns)
rescaledData.show(truncate=False)

# Show the resulting TF-IDF features
rescaledData.select("id", "features").show(truncate=False)

spark.stop()
