from pyspark.sql import SparkSession
from datetime import datetime
import argparse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from operator import itemgetter 
from itertools import chain
import operator
import itertools
from sklearn.metrics import recall_score
import time
import numpy as np

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL CSV to Parquet File Conversion") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Change the directory where the HashEmbedding csv file is present
df = spark.read.csv("/home/diwesh/hashEmbeds.csv", header=False, sep=",")
df.write.partitionBy("_c2").parquet("hashEmbeds.parquet")
print("DONE")

# Search for the top-k data from the hash partitioned parquet file.
hashed_file = spark.read.parquet("hashEmbeds.parquet")
hashed_file.createOrReplaceTempView("parquetFile")
all_value = spark.sql("SELECT * FROM parquetFile")
print("This is all the values : {}\n".format(all_value))
all_value_count = spark.sql("SELECT COUNT(*) FROM parquetFile")
hashcodes = spark.sql("SELECT DISTINCT _c2 FROM parquetFile")
print("Distinct hash values : \n")


recall_value_list = []
new_list = []
for item in [x["_c3"] for x in all_value.rdd.collect()]:
    item = item.split(',')
    c=0
    for i in item:
        i = float(i)
        item[c] = i
        c=c+1
    new_list.append(item)
new_np_array = np.array(new_list)
print("np array of embeddings with all the values in parquet file : \n{}\n".format(new_np_array))
n=25
K = 25
complete_bucket_time = []
hash_bucket_time = []
index = np.random.choice(new_np_array.shape[0], n, replace=False)
key_np_array = new_np_array[index]
print("Key np array : {}\n".format(key_np_array))

for arr_items in key_np_array:
    one_key_list = []
    one_key_list.append(arr_items)
    
    needle_time_complete = []
    for cs in range(50):
        start_time = time.time()
        similarities = cosine_similarity(one_key_list, new_np_array)
        needle_time_complete.append((time.time() - start_time))
    complete_bucket_time.append((sum(needle_time_complete) / len(needle_time_complete)))
    
    sim_list = []
    for j in similarities:
        for k in j:
            p = []
            p.append(k)
            sim_list.append(p)
    sim_np_array = np.array(sim_list)
    output_array = sorted(list(chain.from_iterable(sim_np_array)), reverse=True)[0:K]
    
    Dict = {}
    for key, value in zip(new_np_array, sim_np_array):
        key_list = key.tolist()
        Dict[str(key_list)] = value
    sorted_d = dict( sorted(Dict.items(), key=operator.itemgetter(1),reverse=True))
    sorted_d
    top_k_id_list = []
    print("Sorted Dictionary Size : {}".format(len(sorted_d)))
    sorted_d_n_items = dict(itertools.islice(sorted_d.items(), K))
    print("Top K Dictionary Size : {}".format(len(sorted_d_n_items)))
    for key in sorted_d_n_items.keys():
        key = key[1:-1]
        key = key.replace(" ", "")
        id_id = spark.sql('SELECT _c1 FROM parquetFile WHERE _c3 ="' + str(key) + '"')

        for item in [x["_c1"] for x in id_id.rdd.collect()]:
            top_k_id_list.append(item)

    print("Top {} Image ID : {}".format(K, top_k_id_list))
    top_k_id_list
    
    
    
    
    id_id_id = spark.sql('SELECT _c2 FROM parquetFile WHERE _c3 ="' + key + '"')

    for item in [x["_c2"] for x in id_id_id.rdd.collect()]:
        id_id_id_str = item
    id_id_id_hash = spark.sql('SELECT * FROM parquetFile WHERE _c2 ="' + id_id_id_str + '"')


    new_list = []
    for item in [x["_c3"] for x in id_id_id_hash.rdd.collect()]:
        item = item.split(',')
        c=0
        for i in item:
            i = float(i)
            item[c] = i
            c=c+1
        new_list.append(item)
    new_np_array = np.array(new_list)
    print("np array of embeddings for hash value {} in parquet file : \n{}\n".format(id_id_id_str, new_np_array))

    needle_time_bucket = []
    for cs in range(50):
        start_time_bucket = time.time()
        similarities = cosine_similarity(one_key_list, new_np_array)
        needle_time_bucket.append((time.time() - start_time_bucket))
    hash_bucket_time.append((sum(needle_time_bucket) / len(needle_time_bucket)))
    sim_list = []

    for j in similarities:
        for k in j:
            p = []
            p.append(k)
            sim_list.append(p)
    sim_np_array = np.array(sim_list)

    print("Cosine Similarity size : {}".format(len(sim_np_array)))

    K = 25
    output_array = sorted(list(chain.from_iterable(sim_np_array)), reverse=True)[0:K]
    print("Top {} items : {}".format(K, output_array))


    top_k_id_hash_list = []
    print("Sorted Dictionary Size : {}".format(len(sorted_d)))
    sorted_d_n_items = dict(itertools.islice(sorted_d.items(), K))
    print("Top K Dictionary Size : {}".format(len(sorted_d_n_items)))
    for key in sorted_d_n_items.keys():

        key = key[1:-1]
        key = key.replace(" ", "")

        id_id = spark.sql('SELECT _c1 FROM parquetFile WHERE _c3 ="' + str(key) + '"')

        for item in [x["_c1"] for x in id_id.rdd.collect()]:
            top_k_id_hash_list.append(item)

    print("Top {} Image ID : {}".format(K, top_k_id_hash_list))
    top_k_id_hash_list
    
    score_recall = recall_score(top_k_id_hash_list, top_k_id_list, average='micro')
    recall_value_list.append(score_recall)
    print("RECALL VALUE : {}".format(score_recall))
print("RECALL LIST : {}".format(recall_value_list))
mean_recall = sum(recall_value_list) / len(recall_value_list)
print("MEAN RECALL : {}".format(mean_recall))
print("Time : COMPLETE DATA : {}".format(complete_bucket_time))
print("Time : HASH BUCKET DATA : {}".format(hash_bucket_time))

mean_time_list = []
for list_complete, list_hash in zip(complete_bucket_time, hash_bucket_time):
    mean_time_list.append((list_complete + list_hash)/2)
print("LIST : {}".format(mean_time_list))
