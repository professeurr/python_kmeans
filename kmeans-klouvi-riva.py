from math import sqrt
import random
import sys
import time

from pyspark.sql import SparkSession
from pyspark import SparkConf


def compute_distance(x, y):
    return sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


def sum_list(x, y):
    return [x[i] + y[i] for i in range(len(x))]


def mean_list(x, n):
    return [x[i] / n for i in range(len(x))]


def split_line(line):
    y = line.split(',')
    return [float(x) for x in y[:-1]], y[-1]


def log(x):
    LogBuffer.append(x)


def log_rdd(s, r):
    if debug:
        log('{}: {}'.format(s, r.collect()))


def log_partition(s, p):
    log('partitions of {}: {}'.format(s, p.getNumPartitions()))


def log_title(x):
    log("============== {} ==============".format(x))


def get_log():
    return LogBuffer


def advanced_kmeans(sc, inputs, max_steps, max_partitions, seed):
    clustering_done = False
    number_of_steps = 1
    prev_assignment, assignment, clusters = None, None, None
    error = float("inf")

    data_points = inputs.map(lambda x: (x[1], x[0][0])).persist()  # (0,Array(5.1, 3.5, 1.4, 0.2))
    log_rdd('data points', data_points)

    # A broadcast value is sent to and saved  by each executor for further use
    # instead of being sent to each executor when needed.
    nb_elem = sc.broadcast(data.count())

    #############################
    # Select initial centroids #
    #############################

    data_labels = inputs.map(lambda x: (x[1], x[0][1]))  # (0,Iris-setosa)
    nb_clusters = data_labels.map(lambda x: (x[1], x[0])).reduceByKey(lambda x, y : 1).count()
    log("Number of clusters: {} ".format(clusters))

    centroid = sc.parallelize(data_points.takeSample('withoutReplacment', nb_clusters, seed)) \
        .zipWithIndex() \
        .map(lambda x: (x[1], x[0][1]))

    log_rdd('centroids', centroid)

    while not clustering_done:

        log_title("Step {}".format(number_of_steps))
        #############################
        # Assign points to clusters #
        #############################

        joined = data_points.cartesian(centroid)
        log_partition("joined ", joined)

        joined = joined.coalesce(numPartitions=max_partitions)
        log_partition("joined after coalesce()", joined)

        # We compute the distance between the points and each cluster
        dist = joined.map(lambda x: (x[0][0], ((x[1][0], compute_distance(x[0][1], x[1][1])), x[0][1])))
        # (0, (0, 0.866025403784438))
        log_rdd("dist", dist)
        log_partition("dist", dist)

        # assignment will be our return value : It contains the datapoint,
        # the id of the closest cluster and the distance of the point to the centroid
        assignment = dist.reduceByKey(lambda x, y: x if (x[0][1] < y[0][1]) else y)
        log_partition("assignment", assignment)
        # (0, ((2, 0.5385164807134504), [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']))

        ############################################
        # Compute the new centroid of each cluster #
        ############################################

        clusters = assignment.map(lambda z: (z[1][0][0], (1, z[1][1], z[1][0][1])))
        log_rdd("clusters", clusters)

        count = clusters.reduceByKey(lambda x, y: (x[0] + y[0], sum_list(x[1], y[1]), x[2] + y[2]))

        centroid = count.map(lambda x: (x[0], mean_list(x[1][1], x[1][0])))
        log_rdd("current centroids", centroid)
        log_partition("current centroids", centroid)

        ############################
        # Is the clustering over ? #
        ############################

        # Let's see how many points have switched clusters.
        if prev_assignment:
            switch = assignment.join(prev_assignment).filter(lambda x: x[1][0][0] != x[1][1][0]).count()
        else:
            switch = nb_elem.value
        log("switch: {}".format(switch))

        if switch == 0 or number_of_steps == max_steps:
            clustering_done = True
            error = sqrt(count.map(lambda x: x[1][2]).reduce(lambda x, y: x + y)) / nb_elem.value
        else:
            prev_assignment = assignment
            number_of_steps += 1

    log("Setting up the cluster labels")
    cluster = assignment.join(data_labels).map(lambda x: (x[0], (x[1][0][0],
                                                                 (x[1][0][1][0], x[1][0][1][1], x[1][0][1][2],
                                                                  x[1][0][1][3], x[1][1]))))

    return cluster, error, number_of_steps


if __name__ == "__main__":

    LogBuffer = []
    debug = False

    spark = SparkSession.builder\
        .appName('KMeans_Python_Klouvi_Riva_{}'.format(random.randint(0, 1000000))) \
        .config("spark.logConf", "true") \
        .config("spark.logLevel", "OFF") \
        .getOrCreate()

    input_file = sys.argv[1]
    partitions = 1
    steps = 100
    rand_seed = 42
    start_time = time.time()

    lines = spark.sparkContext.textFile(input_file)
    data = lines.map(lambda x: split_line(x)).zipWithIndex()

    clustering = advanced_kmeans(spark.sparkContext, data, steps, partitions, rand_seed)

    duration = (time.time() - start_time)
    outputPath = "{}/../kmeans_python_cluster".format(input_file)
    metricsPath = "{}/../kmeans_python_metrics".format(input_file)

    log("clusters path: {}".format(outputPath))
    log("metrics path: {}".format(metricsPath))
    log("error: {}".format(clustering[1]))
    log("number of steps: {}".format(clustering[2]))
    log("duration: {} s".format(duration))

    clustering[0].sortBy(lambda x: x[1][0][0]).coalesce(1).saveAsTextFile(outputPath)

    metrics = spark.sparkContext.parallelize(get_log())
    metrics.coalesce(1).saveAsTextFile(metricsPath)
