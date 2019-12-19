from math import sqrt
import random
import sys
import time

from pyspark.sql import SparkSession


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
    nb_clusters = data_labels.map(lambda x: (x[1], x[0])).reduceByKey(lambda x, y: 1).count()
    log("Number of clusters: {} ".format(nb_clusters))

    # Select initial centroids and compute their indices
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
        log_rdd("joined ", joined)
        log_partition("joined ", joined)

        # Reduce number of partitions
        joined = joined.coalesce(numPartitions=max_partitions)
        log_partition("joined after coalesce()", joined)

        # We compute the distance between the points and each cluster
        # Append also the data point to the distance list to avoid the later join()
        # that way we reduce significantly the number of shuffles (data transfer across nodes)
        dist = joined.map(lambda x: (x[0][0], ((x[1][0], compute_distance(x[0][1], x[1][1])), x[0][1])))
        # (0, ((0, 2.882707061079915), [5.1, 3.5, 1.4, 0.2]))
        log_rdd("dist", dist)
        log_partition("dist", dist)

        # assignment will be our return value : It contains the datapoint,
        # the id of the closest cluster and the distance of the point to the centroid
        assignment = dist.reduceByKey(lambda x, y: x if (x[0][1] < y[0][1]) else y)
        #  (19, ( (2, 0.6855654600401041), Array(5.1, 3.8, 1.5, 0.3) ) )
        log_rdd("assignment", assignment)
        log_partition("assignment", assignment)

        ############################################
        # Compute the new centroid of each cluster #
        ############################################

        # Prepare the data points for the counting and summation operations
        clusters = assignment.map(lambda z: (z[1][0][0], (1, z[1][1], z[1][0][1])))
        # (0, (1, [5.1, 3.5, 1.4, 0.2], 2.882707061079915))
        log_rdd("clusters", clusters)
        log_partition("clusters", clusters)

        count = clusters.reduceByKey(lambda x, y: (x[0] + y[0], sum_list(x[1], y[1]), x[2] + y[2]))
        # (0, (64, [325.3, 205.10000000000005, 125.2, 28.500000000000004], 150.41642156981126))
        log_rdd("count", count)

        # Compute the new centroids
        centroid = count.map(lambda x: (x[0], mean_list(x[1][1], x[1][0])))
        # (0, Array(6.301030927835052, 2.8865979381443303, 4.958762886597938, 1.6958762886597938))
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
    debug = True

    spark = SparkSession.builder \
        .appName('KMeans_Python_Klouvi_Riva_{}'.format(random.randint(0, 1000000))) \
        .config("spark.logConf", "true") \
        .config("spark.logLevel", "OFF") \
        .getOrCreate()

    # read the data points file
    input_file = sys.argv[1]
    partitions = 1
    steps = 1
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
