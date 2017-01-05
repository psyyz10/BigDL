/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.models.embedding

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.text.Tokenizer
import com.intel.analytics.bigdl.models.embedding.Utils.Word2VecConfig
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.apache.spark.rdd.RDD

import scala.collection.{mutable, _}


/**
 *  Entry in vocabulary
 */
private case class WordCount(
  var word: String,
  var count: Int
)

object Word2Vec {
  def apply(params: Word2VecConfig): Word2Vec = new Word2Vec(params)
}

class Word2Vec(params: Word2VecConfig) {
  private var trainWordsCount = 0L
  private var vocabSize = 0
  private var powerSum = 0
  @transient private var contexts: Tensor[Float] = _
  @transient private var vocab: Array[WordCount] = _
  @transient private var word2Id = mutable.HashMap.empty[String, Int]
  @transient private var powerUnigram: Array[Int] = _
  var total = 0

  def getModel: Module[Float] = {
    new Sequential()
      .add(
        new ParallelTable()
          .add(new ReLU())
          .add(new ReLU()))
      .add(new MM(false, true))
      .add(new Sigmoid())
  }

  def buildVocab(words: RDD[String]): Unit = {
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= params.minCount)
      .map(x => WordCount(x._1, x._2))
      .collect()
      .sortWith((a, b) => a.count > b.count)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var id = 0
    while (id < vocabSize) {
      word2Id += vocab(id).word -> id
      trainWordsCount += vocab(id).count
      id += 1
    }

    println(s"vocabSize = $vocabSize, trainWordsCount = $trainWordsCount")
  }

  /**
   * Create power-unigram distribution to grenerate negative samples
   *
   */
  def buildNegSampleDistribution(): Unit = {
    powerSum = math.round(
      vocab.foldLeft(0.0)((sum, wordCount) => sum + math.pow(wordCount.count, params.alpha))).toInt

    powerUnigram = new Array(powerSum)

    def wordProb(id: Int) = math.pow(vocab(id).count, params.alpha) / powerSum

    var wordId = 1
    var idx = 1
    var powerProb = wordProb(wordId)
    while (idx < powerSum) {
      powerUnigram(idx) = wordId
      if (idx / powerSum > powerProb) {
        wordId += 1
        powerProb = wordProb(wordId)
      }

      if (wordId > powerSum) {
        wordId -= 1
      }

      idx += 1
    }
  }

  /**
   * Sample negative contexts from power unigram distribution
   *
   * @param context given context
   */
  def sampleNegativeContext(context: Int): Unit = {
    contexts.setValue(1, context)
    var i = 2
    while (i <= params.numNegSamples + 1) {
      val negContext = powerUnigram(math.ceil(RNG.uniform(1, powerSum)).toInt)
      if (context != negContext) {
        contexts.setValue(i, negContext)
        i += 1
      }
    }
  }

  /**
   * Computes the vector representation of each word in vocabulary.
   * @param dataset an RDD of words
   * @return a Word2VecModel
   */
  def fit[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val words = dataset.flatMap(x => x)

    buildVocab(words)

    buildNegSampleDistribution()

    val sc = dataset.context

    Tokenizer -> SentenceToWordIds


  }


//  def train(): Unit = {
//    val scOption = Engine.init(params.nodeNum, params.coreNum,
//      params.env == "spark").map(conf => {
//      conf.setAppName("Text classification")
//        .set("spark.akka.frameSize", 64.toString)
//        .set("spark.task.maxFailures", "1")
//      new SparkContext(conf)
//    })
//  }

  //  def getSimilarWords(word: Int, num: Int): Array[String] = {
  //    if ()
  //  }

}

private case class SentenceToWordIds(
  vocab: mutable.HashMap[String, Int],
  maxSentenceLength: Int)
  extends Transformer[Seq[String], Seq[Int]] {
  val sentence = mutable.ArrayBuilder.make[Int]

  override def apply(prev: Iterator[Seq[String]]): Iterator[Seq[Int]] =
    prev.map(words => {
      var sentenceLength = 0
      while (sentenceLength < maxSentenceLength) {
        val word = vocab.get(words(sentenceLength))
        word match {
          case Some(w) =>
            sentence += w
            sentenceLength += 1
          case None =>
        }
        sentenceLength += 1
      }
      sentence.result()
    })
}

object SentenceToWordIds {
  def apply(
    vocab: mutable.HashMap[String, Int],
    maxSentenceLength: Int = 1000): SentenceToWordIds =
    new SentenceToWordIds(vocab, maxSentenceLength)
}



// object Word2Vec {
//  def main(args: Array[String]): Unit = {
//    val config = Utils.parse(args)
//  }
// }
