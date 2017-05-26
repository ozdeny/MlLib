import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}

/**
  * Created by Developer on 26.05.2017.
  */
object DataConverterUtil {

  def convertRdd2Df(sc: SparkContext, rdd: RDD[LabeledPoint]): DataFrame = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    rdd.toDF()
  }

  def convertRdd2Ds(sc: SparkContext, rdd: RDD[LabeledPoint]): Dataset[LabeledPoint] = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    rdd.toDS()
  }


}
