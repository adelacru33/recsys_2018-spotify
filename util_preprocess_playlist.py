'''
Copyright 2018 Alex Dela Cruz & Kaylan Tirdad

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from functools import reduce

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.sql.functions import explode, col, collect_set, struct, udf
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer

import os
import sys
import time
import operator
import csv

os.environ["PYSPARK_PYTHON"] = "/anaconda3/envs/spotify_env/bin/python"

EXE_MODE = 2                            # 0 = Tracks, 1 = Artist, 2 = Albums
DEV_MODE = False                         # Set to false when processing full data set
DEV_SIZE = 2                           # number of slices to process when developing/testing

# based on total number of unique tracks/artists/albums from data_stats
VOCAB_SIZE = 2262292 if EXE_MODE == 0 else 295860 if EXE_MODE == 1 else 734684
EXE_FIELD = 'track' if EXE_MODE == 0 else 'artist' if EXE_MODE == 1 else 'album'

DATA_PATH = "/Volumes/G-DRIVE mobile/ACM Challenge/mpd.v1/data"
# DATA_PATH = "../data"                   # directory where acm slice data is stores
RESULTS_PATH = "./results"              # location where co-occurrence will be saved, Note: directory must first exist

PREPROCESS_PATH = DATA_PATH + '/preprocessed/{}/'.format(EXE_FIELD)
FILE_FORMAT = 'pp.slice.'
FILE_TYPE = '.csv'

# track fields to extract and consolidate from acm dataset
TRACK_FIELDS = [EXE_FIELD + '_uri', 'pos']

# Setup and start spark environment
conf = SparkConf().set('spark.rdd.compress', 'True')\
                .set('spark.driver.memory', '6g')\
                .set('spark.driver.cores', '4')

sc = SparkContext(master='local', appName='spotify-rms', conf=conf)
# sc.setCheckpointDir("./spark_checkpoints")
sq = SQLContext(sc)


def log_runtime(process, time_start):
    print('{}! Time elapsed: {} seconds'.format(process, time.time() - time_start))


def union_all(*dfs):
    return reduce(DataFrame.union, dfs)


# pyspark user defined function for sorting struct
def sorter(l):
    res = sorted(l, key=operator.itemgetter(0))
    return [item[1] for item in res]


sort_udf = udf(sorter, ArrayType(IntegerType()))

# parse acm data slice into dataframe Row(pid,{..trackfields..})
def parse_slice(full_path):
    playlists = sq.read.json(full_path, multiLine=True)

    # explode playlist array into  rows
    playlists = playlists.select(explode("playlists").alias("playlist"))
    playlists = playlists.select(col("playlist.pid").alias("pid"), col("playlist.tracks").alias("tracks"))

    # explode track array into rows and select required track info
    tracks = playlists.select(col("pid"), explode("tracks").alias("track"))
    select_fields = [col('track.' + k).alias(k) for k in TRACK_FIELDS]
    select_fields.append(col('pid'))
    tracks = tracks.select(select_fields)

    return tracks

def parse_json(path):
    print("parsing data..")
    slices = []

    filenames = os.listdir(path)
    count = 0

    for filename in sorted(filenames):
        if DEV_MODE and count >= DEV_SIZE:
            break

        filename = filename.lower()
        if filename.startswith("mpd.slice") and filename.endswith(".json"):
            fullpath = os.path.join(path, filename)
            df_slice = parse_slice(fullpath)

            # df_slice.show(1)
            slices.append(df_slice)
            count += 1

    return union_all(*slices)

# deprecated function
def query_stats(df, fields, groupings, condition="", distinct=True):
    # print("quering stats: Count({}) From df GroupBy {}\n".format(fields, groupings))

    if condition != "":
        df = df.filter(condition)

    select_fields = fields + groupings
    df = df.select(select_fields)

    # df.show(1)
    if distinct:
        df = df.distinct()
    df = df.groupBy(groupings)

    df = df.count().orderBy(groupings)

    return df

def main():
    path = DATA_PATH if len(sys.argv) == 1 else sys.argv[1]
    print(path)
    time_start = time.time()
    df_playlist = parse_json(path)
    log_runtime('parse_json(path)', time_start)

    df_playlist.registerTempTable("Playlist")

    # convert track_uri to number indices
    print('indexing all track_uris')
    track_uris = sq.sql('SELECT pid, {}_uri, pos FROM Playlist'.format(EXE_FIELD))
    string_indexer = StringIndexer(inputCol=EXE_FIELD+"_uri", outputCol="index")
    model = string_indexer.fit(track_uris)

    track_uris = model.transform(track_uris).select(col("pid"), col('pos'),
                                                    col("index").cast(IntegerType()))

    print('collecting tracks within playlist')
    playlist_tracks = track_uris.groupby('pid')
    playlist_tracks = playlist_tracks.agg(collect_set(struct('pos', 'index')).alias('tracks'))

    playlist_tracks = playlist_tracks.select(col('pid'), sort_udf('tracks').alias('sorted_tracks')).orderBy('pid')

    rdd_tracks = playlist_tracks.rdd

    time_start = time.time()
    print("Starting Iterator")
    slice = 0
    count = 0
    for row in rdd_tracks.toLocalIterator():
        pid = row[0]
        track_list = row[1]
        entry = [pid] + track_list

        file_name = '{}{}{}-{}{}'.format(PREPROCESS_PATH, FILE_FORMAT, slice, slice+999, FILE_TYPE)
        with open(file_name, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(entry)

        count += 1
        if count % 1000 == 0:
            slice += 1000
            print("Processing:{}-{}".format(slice, slice+999))

    log_runtime('pre-processing tracks', time_start)


if __name__ == '__main__':
    main_start = time.time()
    main()
    log_runtime('main()', main_start)
