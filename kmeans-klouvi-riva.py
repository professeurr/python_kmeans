from math import sqrt
import random
import sys

from pyspark.sql import SparkSession


LogBuffer = []

def computeDistance(x,y):
    return sqrt(sum([(a - b)**2 for a,b in zip(x,y)]))


def closestCluster(dist_list):
    cluster = dist_list[0][0]
    min_dist = dist_list[0][1]
    for elem in dist_list:
        if elem[1] < min_dist:
            cluster = elem[0]
            min_dist = elem[1]
    return (cluster,min_dist)

def sumList(x,y):
    return [x[i]+y[i] for i in range(len(x))]

def moyenneList(x,n):
    return [x[i]/n for i in range(len(x))]


def log(x):
    LogBuffer.append(x)


def logTitle(x):
    log("============== {} ==============".format(x))


def getLog():
    return LogBuffer


def advancedKmeans(sc, data, nb_clusters, maxSteps):
    clusteringDone = False
    number_of_steps = 0
    current_error = float("inf")
    # A broadcast value is sent to and saved  by each executor for further use
    # instead of being sent to each executor when needed.
    nb_elem = sc.broadcast(data.count())

    #############################
    # Select initial centroides #
    #############################

    centroides = sc.parallelize(data.takeSample('withoutReplacment',nb_clusters))\
              .zipWithIndex()\
              .map(lambda x: (x[1],x[0][1][:-1]))
    # (0, [4.4, 3.0, 1.3, 0.2])
    # In the same manner, zipWithIndex gives an id to each cluster

    while not clusteringDone:

        #############################
        # Assign points to clusters #
        #############################

        joined = data.cartesian(centroides)
        # ((0, [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']), (0, [4.4, 3.0, 1.3, 0.2]))

        # We compute the distance between the points and each cluster
        dist = joined.map(lambda x: (x[0][0],(x[1][0], computeDistance(x[0][1][:-1], x[1][1]))))
        # (0, (0, 0.866025403784438))

        dist_list = dist.groupByKey().mapValues(list)
        # (0, [(0, 0.866025403784438), (1, 3.7), (2, 0.5385164807134504)])

        # We keep only the closest cluster to each point.
        min_dist = dist_list.mapValues(closestCluster)
        # (0, (2, 0.5385164807134504))

        # assignment will be our return value : It contains the datapoint,
        # the id of the closest cluster and the distance of the point to the centroid
        assignment = min_dist.join(data)

        # (0, ((2, 0.5385164807134504), [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']))

        ############################################
        # Compute the new centroid of each cluster #
        ############################################

        clusters = assignment.map(lambda x: (x[1][0][0], x[1][1][:-1]))
        # (2, [5.1, 3.5, 1.4, 0.2])

        count = clusters.map(lambda x: (x[0],1)).reduceByKey(lambda x,y: x+y)
        somme = clusters.reduceByKey(sumList)
        centroidesCluster = somme.join(count).map(lambda x : (x[0],moyenneList(x[1][0],x[1][1])))

        ############################
        # Is the clustering over ? #
        ############################

        # Let's see how many points have switched clusters.
        if number_of_steps > 0:
            switch = prev_assignment.join(min_dist)\
                                    .filter(lambda x: x[1][0][0] != x[1][1][0])\
                                    .count()
        else:
            switch = 150
        if switch == 0 or number_of_steps == maxSteps:
            clusteringDone = True
            error = sqrt(min_dist.map(lambda x: x[1][1]).reduce(lambda x,y: x + y))/nb_elem.value
        else:
            centroides = centroidesCluster
            prev_assignment = min_dist
            number_of_steps += 1

    return (assignment, error, number_of_steps)


if __name__ == "__main__":

    spark = SparkSession.builder.appName('KMeans_Python_Klouvi_Riva_{}'.format(random.randint(0, 1000000))).getOrCreate()
    sc = spark.sparkContext
    input_file = sys.argv[1]

    lines = sc.textFile(input_file)
    data = lines.map(lambda x: x.split(','))\
            .map(lambda x: [float(i) for i in x[:4]]+[x[4]])\
            .zipWithIndex()\
            .map(lambda x: (x[1],x[0]))
    # zipWithIndex allows us to give a specific index to each point
    # (0, [5.1, 3.5, 1.4, 0.2, 'Iris-setosa'])

    clustering = advancedKmeans(sc, data, 3, 1)

    outputPath = "{}/../kmeans_python_cluster".format(input_file)
    metricsPath = "{}/../kmeans_python_metrics".format(input_file)

    log("clusters path: {}".format(outputPath))
    log("metrics path: {}".format(metricsPath))
    log("error: {}".format(clustering[1]))
    log("number of steps: {}".format(clustering[2]))

    clustering[0].coalesce(1).saveAsTextFile(outputPath)

    metrics = sc.parallelize(getLog())
    metrics.coalesce(1).saveAsTextFile(metricsPath)

    print(clustering)
