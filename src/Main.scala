import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
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

      val rdd = data.map(_.split(",")).filter(_(6) != "?").map(_.drop(1))
      .map(_.map(_.toDouble))

      val labeledPoints = rdd.map(x => LabeledPoint(if(x.last == 4) 1 else 0,Vectors.dense(x.init)))

      val splits = labeledPoints.randomSplit(Array(0.6, 0.4), seed = 11L)
      val training = splits(0).cache()
      val test = splits(1)

    // Run training algorithm to build the model
      val numIterations = 100
      val model = SVMWithSGD.train(training,numIterations)

    // Clear the default threshold.
     model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)




  }
}