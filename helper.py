from math import sqrt

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