#!/bin/bash

clustering_output=$2/../kmeans_python_cluster
clustering_metrics=$2/../kmeans_python_metrics

hdfs dfs -rm -f -r $clustering_output
hdfs dfs -rm -f -r $clustering_metrics

spark-submit\
  --master yarn --deploy-mode cluster \
  --executor-cores 4 \
  --num-executors 1 \
  --executor-memory 1g \
  --conf spark.yarn.executor.memoryOverhead=1g \
  --conf spark.driver.memory=1g \
  --conf spark.driver.cores=1 \
  --conf spark.yarn.jars="file:///home/cluster/shared/spark/jars/*.jar" \
  $1 $2

echo "=============== Clustering data ============"
hdfs dfs -cat $clustering_output/part-00000

echo "=============== Metrics ===================="
hdfs dfs -cat $clustering_metrics/part-00000

#--conf spark.yarn.jars="file:///home/cluster/shared/spark/jars/*.jar" \
#./run.sh kmeans_scala_klouvi_riva_2.11-1.0.jar hdfs:///user/user159/iris.data.txt

#hdfs dfs -cat $3/log/part-00000
