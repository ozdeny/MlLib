import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import sun.security.pkcs11.wrapper.CK_SSL3_MASTER_KEY_DERIVE_PARAMS
/**
  * Created by Developer on 26.05.2017.
  */
object Main {
  def main(args: Array[String]) = {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("MlLibSvm")
    val sparkContext = new SparkContext(sparkConf)

    val data = sparkContext.textFile("C:\\Datasets\\breast-cancer-wisconsin.data.txt")

    val rdd = data.map(_.split(",")).filter(_ (6) != "?").map(_.drop(1))
      .map(_.map(_.toDouble))

    val labeledPoints = rdd.map(x => LabeledPoint(if (x.last == 4) 1 else 0, Vectors.dense(x.init)))

    val splits = labeledPoints.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val sqlContext = new SQLContext(sparkContext)
    import sqlContext.implicits._



    classifyWithSvm(training,test)

    classifyWithDecisionTree(training,test)

    classifyWithRandomForest(training,test)

  }


  def classifyWithRandomForest(trainingSamples: RDD[LabeledPoint],testSamples:RDD[LabeledPoint]) ={

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model = RandomForest.trainClassifier(trainingSamples, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testSamples.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testSamples.count()
    println("Test Error = " + testErr)
    println("Learned classification forest model:\n" + model.toDebugString)

  }


  def classifyWithDecisionTree(trainingSamples: RDD[LabeledPoint],testSamples:RDD[LabeledPoint]) =
  {
    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingSamples, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testSamples.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testSamples.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model:\n" + model.toDebugString)
  }


  def classifyWithSvm(trainingSamples: RDD[LabeledPoint],testSamples:RDD[LabeledPoint]) ={
    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(trainingSamples, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = testSamples.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

  }
}
