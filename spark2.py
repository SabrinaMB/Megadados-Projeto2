import findspark
findspark.init("[spark install location]")

import pyspark
import string
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, Tokenizer, RegexTokenizer, StopWordsRemover

sc = pyspark.SparkContext(appName = "LDA_app")

#Function to load lines in a CSV file, and remove some special characters
def parseLine(line):
    line = line.encode('ascii',errors='ignore')
    line_split = line.replace('"','').replace('.','')\
    .replace('(','').replace(')','').replace('!','').split(';')
    return line_split

sqlContext = SQLContext(sc)

#load dataset, a local CSV file, and load this as a SparkSQL dataframe without external csv libraries.

dataset_location = 'lyrics.csv'
sqlContext = SQLContext(sc)

data_set = sc.textFile(dataset_location)

labels = data_set.first().replace('"','').split(';')

#create a schema
fields = [StructField(field_name, StringType(), True) for field_name in labels]
schema = StructType(fields)

#get everything but the header:
header = data_set.take(1)[0]
data_set = data_set.filter(lambda line: line != header)

#parse dataset
data_set = data_set.map(parseLine)

#create dataframe
data_df = sqlContext.createDataFrame(data_set, schema)


#Tokenize the text in the text column
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsDataFrame = tokenizer.transform(data_df)


#remove 20 most occuring documents, documents with non numeric characters, and documents with <= 3 characters
cv_tmp = CountVectorizer(inputCol="words", outputCol="tmp_vectors")
cv_tmp_model = cv_tmp.fit(wordsDataFrame)


top20 = list(cv_tmp_model.vocabulary[0:20])
more_then_3_charachters = [word for word in cv_tmp_model.vocabulary if len(word) <= 3]
contains_digits = [word for word in cv_tmp_model.vocabulary if any(char.isdigit() for char in word)]

stopwords = []  #Add additional stopwords in this list

#Combine the three stopwords
stopwords = stopwords + top20 + more_then_3_charachters + contains_digits

#Remove stopwords from the tokenized list
remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords = stopwords)
wordsDataFrame = remover.transform(wordsDataFrame)

#Create a new CountVectorizer model without the stopwords
cv = CountVectorizer(inputCol="filtered", outputCol="vectors")
cvmodel = cv.fit(wordsDataFrame)
df_vect = cvmodel.transform(wordsDataFrame)

#transform the dataframe to a format that can be used as input for LDA.train. LDA train expects a RDD with lists,
#where the list consists of a uid and (sparse) Vector
def parseVectors(line):
    return [int(line[2]), line[0]]


sparsevector = df_vect.select('vectors', 'text', 'id').map(parseVectors)

#Train the LDA model
model = LDA.train(sparsevector, k=5, seed=1)

#Print the topics in the model
topics = model.describeTopics(maxTermsPerTopic = 15)
for x, topic in enumerate(topics):
    print 'topic nr: ' + str(x)
    words = topic[0]
    weights = topic[1]
    for n in range(len(words)):
        print cvmodel.vocabulary[words[n]] + ' ' + str(weights[n])
