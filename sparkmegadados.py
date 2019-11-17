import pandas as pd
import pyspark
from pyspark.sql import SQLContext
import os
from nltk.corpus import stopwords
import nltk
import re as re
from pyspark.ml.feature import CountVectorizer , IDF
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import SparkSession
from pyspark import SparkContext

nltk.download('stopwords')
sc =SparkContext()

spark = SparkSession.builder.master("local[*]").appName("Learning_Spark").getOrCreate()

sqlContext = SQLContext(sc)

# reading the data
data = sqlContext.read.format("csv").options(header='true', inferschema='true').load(os.path.realpath("lyrics.csv"))

reviews = data.rdd.map(lambda x : x['Review Text']).filter(lambda x: x is not None)
StopWords = stopwords.words("english")

tokens = reviews.map( lambda document: document.strip().lower()).map( lambda document: re.split(" ", document)).map( lambda word: [x for x in word if x.isalpha()]).map( lambda word: [x for x in word if len(x) > 3] ).map( lambda word: [x for x in word if x not in StopWords]).zipWithIndex()

exit()

df_txts = sqlContext.createDataFrame(tokens, ["list_of_words",'index'])
# TF
cv = CountVectorizer(inputCol="list_of_words", outputCol="raw_features", vocabSize=5000, minDF=10.0)
cvmodel = cv.fit(df_txts)
result_cv = cvmodel.transform(df_txts)
# IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(result_cv)
result_tfidf = idfModel.transform(result_cv)

num_topics = 10
max_iterations = 100
lda_model = LDA.train(result_tfidf[['index','features']].map(list), k=num_topics, maxIterations=max_iterations)

wordNumbers = 5
topicIndices = spark.parallelize(lda_model.describeTopics\
(maxTermsPerTopic = wordNumbers))

def topic_render(topic):
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result
topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()
for topic in range(len(topics_final)):
    print ("Topic" + str(topic) + ":")
    for term in topics_final[topic]:
        print (term)
    print ('\n')
