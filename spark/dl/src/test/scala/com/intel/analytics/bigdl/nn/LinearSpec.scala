/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl._

import scala.math._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim.{L1Regularizer, L2Regularizer, SGD}
import com.intel.analytics.bigdl.utils.T

@com.intel.analytics.bigdl.tags.Parallel
class LinearSpec extends FlatSpec with Matchers {
  "Linear L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val inputN = 5
    val outputN = 2
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](batchSize, inputN).rand()
    val labels = Tensor[Double](batchSize, outputN).rand()

    val model1 = Sequential()
      .add(Linear(inputN, outputN))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential()
      .add(Linear(inputN, outputN,
        wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1)))
      .add(Sigmoid())
    val (weights2, grad2) = model2.getParameters()
    weights2.copy(weights1.clone())
    grad2.copy(grad1.clone())

    val sgd = new SGD[Double]

    def feval1(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model1.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model1.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model1.backward(input, gradInput)
      (_loss, grad1)
    }

    def feval2(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model2.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model2.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model2.backward(input, gradInput)
      (_loss, grad2)
    }

    var loss1: Array[Double] = null
    for (i <- 1 to 100) {
      loss1 = sgd.optimize(feval1, weights1, state1)._2
      println(s"${i}-th loss = ${loss1(0)}")
    }

    var loss2: Array[Double] = null
    for (i <- 1 to 100) {
      loss2 = sgd.optimize(feval2, weights2, state2)._2
      println(s"${i}-th loss = ${loss2(0)}")
    }


    weights1 should be(weights2)
    loss1 should be(loss2)
  }

  "Linear without bias L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val inputN = 5
    val outputN = 2
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](batchSize, inputN).rand()
    val labels = Tensor[Double](batchSize, outputN).rand()

    val model1 = Sequential()
      .add(Linear(inputN, outputN, withBias = false))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential()
      .add(Linear(inputN, outputN, withBias = false,
        wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1)))
      .add(Sigmoid())
    val (weights2, grad2) = model2.getParameters()
    weights2.copy(weights1.clone())
    grad2.copy(grad1.clone())


    val sgd = new SGD[Double]

    def feval1(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model1.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model1.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model1.backward(input, gradInput)
      (_loss, grad1)
    }

    def feval2(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model2.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model2.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model2.backward(input, gradInput)
      (_loss, grad2)
    }

    var loss1: Array[Double] = null
    for (i <- 1 to 100) {
      loss1 = sgd.optimize(feval1, weights1, state1)._2
      println(s"${i}-th loss = ${loss1(0)}")
    }

    var loss2: Array[Double] = null
    for (i <- 1 to 100) {
      loss2 = sgd.optimize(feval2, weights2, state2)._2
      println(s"${i}-th loss = ${loss2(0)}")
    }


    weights1 should be(weights2)
    loss1 should be(loss2)
  }

  "Linear module" should "converge to correct weight and bias" in {
    val inputN = 5
    val outputN = 2

    val linear = new Linear[Double](inputN, outputN)
    val mse = new MSECriterion[Double]

    val input = Tensor[Double](inputN)
    val res = Tensor[Double](outputN)
    var err = 0.0
    for (i <- 1 to 10000) {
      input.rand()
      for (y <- 1 to outputN) {
        res(Array(y)) = 1.0 * y
        for (x <- 1 to inputN) {
          res(Array(y)) += 0.1 * y * x * input(Array(x))
        }
      }
      val output = linear.forward(input)
      err = mse.forward(output, res)
      val grad = mse.backward(output, res)
      linear.zeroGradParameters()
      linear.backward(input, grad)
      linear.updateParameters(0.5 / log(i + 3))
    }
    val params = linear.parameters()
    val weight = params._1(0)
    val bias = params._1(1)

    val expectedWeight = Tensor[Double](outputN, inputN)
    val expectedBias = Tensor[Double](outputN)
    for (y <- 1 to outputN) {
      expectedBias(Array(y)) = 1.0 * y
      for (x <- 1 to inputN) {
        expectedWeight(Array(y, x)) = 0.1 * y * x
      }
    }

    expectedBias.map(bias, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedWeight.map(weight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(err < 1e-6)
  }

  "Linear module in batch mode" should "converge to correct weight and bias" in {
    val inputN = 5
    val outputN = 2
    val batchN = 3

    val linear = new Linear[Double](inputN, outputN)
    val mse = new MSECriterion[Double]

    val input = Tensor[Double](batchN, inputN)
    val res = Tensor[Double](batchN, outputN)
    var err = 0.0
    for (i <- 1 to 10000) {
      input.rand()
      for (k <- 1 to batchN) {
        for (y <- 1 to outputN) {
          res(Array(k, y)) = 1.0 * y
          for (x <- 1 to inputN) {
            res(Array(k, y)) += 0.1 * y * x * input(Array(k, x))
          }
        }
      }
      val output = linear.forward(input)
      err = mse.forward(output, res)
      val grad = mse.backward(output, res)
      linear.zeroGradParameters()
      linear.backward(input, grad)
      linear.updateParameters(0.5 / log(i + 3))
    }
    val params = linear.parameters()
    val weight = params._1(0)
    val bias = params._1(1)

    val expectedWeight = Tensor[Double](outputN, inputN)
    val expectedBias = Tensor[Double](outputN)
    for (y <- 1 to outputN) {
      expectedBias(Array(y)) = 1.0 * y
      for (x <- 1 to inputN) {
        expectedWeight(Array(y, x)) = 0.1 * y * x
      }
    }

    expectedBias.map(bias, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedWeight.map(weight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(err < 1e-6)
  }

  "Linear module in batch mode without bias" should "converate to correct weight and bias" in {
    val inputN = 5
    val outputN = 2
    val batchN = 3

    val linear = new Linear[Double](inputN, outputN, withBias = false)
    val mse = new MSECriterion[Double]

    val input = Tensor[Double](batchN, inputN)
    val res = Tensor[Double](batchN, outputN)
    var err = 0.0
    for (i <- 1 to 10000) {
      input.rand()
      for (k <- 1 to batchN) {
        for (y <- 1 to outputN) {
          res(Array(k, y)) = 0
          for (x <- 1 to inputN) {
            res(Array(k, y)) += 0.1 * y * x * input(Array(k, x))
          }
        }
      }
      val output = linear.forward(input)
      err = mse.forward(output, res)
      val grad = mse.backward(output, res)
      linear.zeroGradParameters()
      linear.backward(input, grad)
      linear.updateParameters(0.5 / log(i + 3))
    }
    val params = linear.parameters()
    val weight = params._1(0)

    val expectedWeight = Tensor[Double](outputN, inputN)
    for (y <- 1 to outputN) {
      for (x <- 1 to inputN) {
        expectedWeight(Array(y, x)) = 0.1 * y * x
      }
    }

    expectedWeight.map(weight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(err < 1e-6)
  }

  "Linear module in batch mode" should "be good in gradient check" in {
    val linear = new Linear[Double](5, 2)
    linear.reset()
    val input = Tensor[Double](3, 5).rand()

    val checker = new GradientChecker(1e-4, 1e-2)
    checker.checkLayer[Double](linear, input) should be(true)
  }
}
