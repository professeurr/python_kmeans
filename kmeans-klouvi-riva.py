from math import sqrt
import random
import sys
import time

from pyspark.sql import SparkSession


LogBuffer = []


def computeDistance(x, y):
    return sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


def closestCluster(dist_list):
    cluster = dist_list[0][0]
    min_dist = dist_list[0][1]
    for elem in dist_list:
        if elem[1] < min_dist:
            cluster = elem[0]
            min_dist = elem[1]
    return (cluster, min_dist)


def sumList(x, y):
    return [x[i] + y[i] for i in range(len(x))]


def meanList(x, n):
    return [x[i] / n for i in range(len(x))]


def split_line(line):
    y = line.split(',')
    return [float(x) for x in y[:-1]], y[-1]


def log(x):
    LogBuffer.append(x)


debug = False
def log_rdd(s, r):
    if debug:
        log('{}: {}'.format(s, r.collect()))


def log_title(x):
    log("============== {} ==============".format(x))

def getLog():
    return LogBuffer


def advancedKmeans(sc, inputs, nb_clusters, max_steps, max_partitions, seed):
    clusteringDone = False
    number_of_steps = 1
    prev_assignment, assignment = None, None
    error = float("inf")

    data = inputs.map(lambda x: (x[1], x[0][0])).persist()  # (0,Array(5.1, 3.5, 1.4, 0.2))
    labels = inputs.map(lambda x: (x[1], x[0][1]))  # (0,Iris-setosa)

    log_rdd('data', data)
    log_rdd('labels', labels)

    # A broadcast value is sent to and saved  by each executor for further use
    # instead of being sent to each executor when needed.
    nb_elem = sc.broadcast(data.count())

    #############################
    # Select initial centroids #
    #############################

    centroids = sc.parallelize(data.takeSample('withoutReplacment', nb_clusters, seed)) \
        .zipWithIndex() \
        .map(lambda x: (x[1], x[0][1]))

    log_rdd('centroids', centroids)
    # (0, [4.4, 3.0, 1.3, 0.2])
    # In the same manner, zipWithIndex gives an id to each cluster

    while not clusteringDone:

        log_title("Step {}".format(number_of_steps))
        #############################
        # Assign points to clusters #
        #############################

        joined = data.cartesian(centroids)
        # ((0, [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']), (0, [4.4, 3.0, 1.3, 0.2]))

        joined = joined.coalesce(numPartitions=max_partitions)

        # We compute the distance between the points and each cluster
        dist = joined.map(lambda x: (x[0][0], ((x[1][0], computeDistance(x[0][1], x[1][1])), x[0][1])))
        # (0, (0, 0.866025403784438))
        log_rdd("dist", dist)

        # assignment will be our return value : It contains the datapoint,
        # the id of the closest cluster and the distance of the point to the centroid
        assignment = dist.reduceByKey(lambda x, y: x if (x[0][1] < y[0][1]) else y)

        # (0, ((2, 0.5385164807134504), [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']))

        ############################################
        # Compute the new centroid of each cluster #
        ############################################

        clusters = assignment.map(lambda z: (z[1][0][0], (1, z[1][1], z[1][0][1])))
        # (2, [5.1, 3.5, 1.4, 0.2])

        count = clusters.reduceByKey( lambda x, y: (x[0] + y[0], sumList(x[1], y[1]), x[2] + y[2]))

        centroidesCluster = count.map(lambda x: (x[0], meanList(x[1][1], x[1][0])))

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
            clusteringDone = True
            error = sqrt(count.map(lambda x: x[1][2]).reduce(lambda x, y: x + y)) / nb_elem.value
        else:
            centroids = centroidesCluster
            prev_assignment = assignment
            number_of_steps += 1

    cluster = assignment.join(labels).map(lambda x: (x[0], (x[1][0][0], (x[1][0][1][0], x[1][0][1][1], x[1][0][1][2], x[1][0][1][3], x[1][1]))))

    return cluster, error, number_of_steps


if __name__ == "__main__":

    spark = SparkSession.builder.appName('KMeans_Python_Klouvi_Riva_{}'.format(random.randint(0, 1000000))).getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    input_file = sys.argv[1]
    max_partitions = 1
    nb_clusters = 3
    max_steps = 100
    seed = 42
    start_time = time.time()

    lines = sc.textFile(input_file)
    data = lines.map(lambda x: split_line(x)).zipWithIndex()

    clustering = advancedKmeans(sc, data, nb_clusters, max_steps, max_partitions, seed)

    duration = (time.time() - start_time)
    outputPath = "{}/../kmeans_python_cluster".format(input_file)
    metricsPath = "{}/../kmeans_python_metrics".format(input_file)

    log("clusters path: {}".format(outputPath))
    log("metrics path: {}".format(metricsPath))
    log("error: {}".format(clustering[1]))
    log("number of steps: {}".format(clustering[2]))
    log("duration: {} s".format(duration))

    clustering[0].sortBy(lambda x: x[1][0][0]).coalesce(1).saveAsTextFile(outputPath)

    metrics = sc.parallelize(getLog())
    metrics.coalesce(1).saveAsTextFile(metricsPath)

    print(clustering)
