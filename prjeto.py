# import kaggle
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
import pandas as pd
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, Tokenizer, RegexTokenizer, StopWordsRemover



# spark = SparkSession.builder \
#     .master("local[*]") \
#     .appName("Learning_Spark") \
#     .getOrCreate()

sc = pyspark.SparkContext(appName = "LDA_app")

sqlContext = SQLContext(sc)

pdDF = pd.read_csv('Megadados-Projeto2/lyrics.csv')

mySchema = StructType([ StructField("index", LongType(), True)\
                       ,StructField("song", StringType(), True)\
                       ,StructField("year", IntegerType(), True)\
                       ,StructField("artist", StringType(), True)\
                       ,StructField("genre", StringType(), True)\
                       ,StructField("lyrics", StringType(), True)])

df = sqlContext.createDataFrame(pdDF,schema=mySchema)

tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
wordsDataFrame = tokenizer.transform(df)


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






























