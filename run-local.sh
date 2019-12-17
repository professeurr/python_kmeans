#!/bin/bash

clustering_output=$2/../kmeans_python_cluster
clustering_metrics=$2/../kmeans_python_metrics

hdfs dfs -rm -f -r $clustering_output
hdfs dfs -rm -f -r $clustering_metrics

spark-submit\
  --master yarn --deploy-mode cluster \
  --executor-cores 2 \
  --num-executors 8 \
  --executor-memory 1g \
  --conf spark.executor.memoryOverhead=1g \
  --conf spark.driver.memory=1g \
  --conf spark.driver.cores=1 \
  --conf spark.logConf=true \
  --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=/home/masterai/dev/master_iasd/bigdata/project/python_kmean/log4j.properties"\
  --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=/home/masterai/dev/master_iasd/bigdata/project/python_kmean/log4j.properties"\
  $1 $2


echo "=============== Clustering data ============"
hdfs dfs -cat $clustering_output/part-00000

echo "=============== Metrics ===================="
hdfs dfs -cat $clustering_metrics/part-00000


#./run-local.sh ./target/scala-2.11/kmeans_scala_klouvi_riva_2.11-1.0.jar  hdfs://localhost:9090/bigdata/project/kmeans/iris.data.txt

