///*
// * Copyright 2016 The BigDL Authors.
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package com.intel.analytics.bigdl.nn
//
//import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
//import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
//import com.intel.analytics.bigdl.tensor.Tensor
//
//import scala.concurrent.duration.Duration
//import scala.concurrent.{Await, Future}
//import scala.reflect.ClassTag
//import com.intel.analytics.bigdl.utils.Engine
//
//import scala.collection.mutable.ArrayBuffer
//
///**
// * The negative log likelihood criterion. It is useful to train a classification problem with n
// * classes. If provided, the optional argument weights should be a 1D Tensor assigning weight to
// * each of the classes. This is particularly useful when you have an unbalanced training set.
// *
// * The input given through a forward() is expected to contain log-probabilities of each class:
// * input has to be a 1D Tensor of size n. Obtaining log-probabilities in a neural network is easily
// * achieved by adding a LogSoftMax layer in the last layer of your neural network. You may use
// * CrossEntropyCriterion instead, if you prefer not to add an extra layer to your network. This
// * criterion expects a class index (1 to the number of class) as target when calling
// * forward(input, target) and backward(input, target).
// *
// * The loss can be described as:
// *     loss(x, class) = -x[class]
// * or in the case of the weights argument it is specified as follows:
// *     loss(x, class) = -weights[class] * x[class]
// * Due to the behaviour of the backend code, it is necessary to set sizeAverage to false when
// * calculating losses in non-batch mode.
// *
// * Note that if the target is `-1`, the training process will skip this sample.
// * In other words, the forward process will return zero output and the backward process
// * will also return zero `gradInput`.
// *
// * By default, the losses are averaged over observations for each minibatch. However, if the field
// * sizeAverage is set to false, the losses are instead summed for each minibatch.
// *
// * @param weights weights of each element of the input
// * @param sizeAverage size average of batch
// * @param ev numeric operator
// * @tparam T numeric type
// */
//@SerialVersionUID(- 8696382776046599502L)
//class TimeDistributedClassNLLWithMask[@specialized(Float, Double) T: ClassTag]
//(weights: Tensor[T] = null, sizeAverage: Boolean = true)
//  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
//  private var fInput: Tensor[T] = Tensor[T]()
//  private var fTarget: Tensor[T] = Tensor[T]()
//  private var _gradInput = Tensor[T]()  // list of cell criterions cloned from added criterion
//  private val cells: ArrayBuffer[TensorCriterion[T]]
//  = ArrayBuffer[TensorCriterion[T]]()
//
//  @transient
//  protected var results: Array[Future[Unit]] = _
//  private var total_weight = ev.fromType[Int](0)
//  if (weights != null) require(weights.dim() == 1,
//    "weights input should be 1-D Tensor" +
//      s"weights dim(${weights.dim()})")
//
//  @transient
//  private var results: Array[Future[(T, T)]] = null
//  @transient
//  private var resultsBackward: Array[Future[_]] = null
//  private val critrn = ClassNLLCriterion()
//
//  /**
//   * Clone N criterions; N depends on the time dimension of the input
//   * @param times
//   */
//  private def extend(times: Int): Unit = {
//    var t = cells.length
//    while (t < times) {
//      cells += critrn.cloneCriterion()
//        .asInstanceOf[TensorCriterion[T]]
//      t += 1
//    }
//  }
//
//
//  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
//    /**
//     * Take each time slice of input and target, and add up all outputs of slices
//     * Example with dimension=2:
//     * input.size = [B, T, D] => fInput.size = [B, D]
//     * target.size = [B, T] => fTarget.size = [B]
//     * If sizeAverage is true, the output is averaged through time dim
//     */
//    require(input.size(dimension) == target.size(dimension),
//      "target should have as many elements as input, " +
//        s"input ${input.size(dimension)}, target ${target.size(dimension)}")
//
//
//    output = ev.fromType[Int](0)
//    val nstep = input.size(dimension)
//    extend(nstep)
//
//    if (results == null || results.length != nstep) {
//      results = new Array[Future[Unit]](nstep)
//    }
//
//    var i = 0
//    while (i < nstep) {
//      val _i = i + 1
//      results(i) = Engine.model.invoke(() => {
//        fInput = input.select(dimension, _i)
//        fTarget = target.select(dimension, _i)
//        cells(_i - 1).updateOutput(fInput, fTarget)
//      })
//      i += 1
//    }
//    Engine.model.sync(results)
//
//
//    val mask = target.apply1(x => if (x != ev.fromType[Int](-1)) ev.one else ev.zero)
//    val outputBuff = Tensor()
//    outputBuff.resizeAs(mask)
//    (0 until nstep).foreach(b => {
//      outputBuff.setValue(b, ev.plus(output, cells(b).output))
//    })
//
//    if (sizeAverage) {
//      if (isMask) {
//        outputBuff.cdiv(mask.sum(2))
//        output = outputBuff.sum()
//      } else {
//        output = ev.divide(outputBuff.sum(), ev.fromType[Int](nstep))
//      }
//    } else {
//      output = outputBuff.sum()
//    }
//
//    output
//  }
//
//  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
//    /**
//     * Take each time slice of input and target, and calculate gradInput of each slice
//     * If sizeAverage is true, the gradInput is also averaged through dimension
//     */
//    require(input.size(dimension) == target.size(dimension),
//      s"target should have as many elements as input, " +
//        s"input ${input.size(dimension)}, target ${target.size(dimension)}")
//    gradInput.resizeAs(input).zero()
//
//    val nstep = input.size(dimension)
//
//    var i = 0
//    while (i < nstep) {
//      val _i = i + 1
//      results(i) = Engine.model.invoke(() => {
//        fInput = input.select(dimension, _i)
//        fTarget = target.select(dimension, _i)
//        _gradInput = gradInput.select(dimension, _i)
//        _gradInput.copy(cells(_i - 1).updateGradInput(fInput, fTarget).toTensor[T])
//        if (sizeAverage) {
//          _gradInput = _gradInput.div(ev.fromType[Int](nstep))
//        }
//      })
//      i += 1
//    }
//    Engine.model.sync(results)
//    gradInput
//  }
//
//  override def classNLLUpdateOutput(input: Tensor[T], target: Tensor[T]): T = {
//    require(input.dim() == 1 || input.dim() == 2,
//      "ClassNLLCriterion: " +
//        ErrorInfo.constrainInputAsVectorOrBatch +
//        s"input dim(${input.dim()})")
//    val nClasses = input.size(input.dim())
//    if (input.dim() == 1) {
//      require(input.dim() == target.dim(),
//        "ClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget +
//          s" Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }")
//      val curTarget = ev.toType[Int](target.valueAt(1))
//      assert(curTarget >= 1 && curTarget <= nClasses || curTarget == -1,
//        s"curTarget ${curTarget} is out of range, should be 1 to ${nClasses}")
//      total_weight = if (weights != null) weights(Array(curTarget)) else ev.fromType[Int](1)
//      output = if (curTarget == -1) ev.zero
//      else ev.times(ev.negative(input.valueAt(curTarget)), total_weight)
//    } else if (input.dim() == 2) {
//      val batchSize = input.size(1)
//      val targetSize = target.size()
//      target.squeeze()
//      require(target.dim() == 1,
//        "ClassNLLCriterion: illegal target! Target should be 1D tensor after squeeze," +
//          s"but target's size is: ${ target.size() }, please check your data.")
//
//      total_weight = ev.fromType[Int](0)
//      output = ev.fromType[Int](0)
//
//      if (results == null || results.length != batchSize) {
//        results = new Array[Future[(T, T)]](batchSize)
//      }
//
//      var i = 1
//      while (i <= batchSize) {
//        val _i = i
//        results(_i - 1) = Engine.model.invoke( () => {
//          val curTarget = ev.toType[Int](target.valueAt(_i))
//          assert(curTarget >= 1 && curTarget <= nClasses || curTarget == -1,
//            s"curTarget ${curTarget} is out of range 1 to ${nClasses}")
//          if (curTarget == -1) (ev.zero, ev.one)
//          else {
//            val curWeight = if (weights != null) weights.valueAt(curTarget) else ev.fromType[Int](1)
//            (ev.times(input.valueAt(_i, curTarget), curWeight), curWeight)
//          }
//        })
//        i += 1
//      }
//
//      i = 0
//      while (i < batchSize) {
//        val (o, w) = Await.result(results(i), Duration.Inf)
//        output = ev.minus(output, o)
//        total_weight = ev.plus(total_weight, w)
//        i += 1
//      }
//      target.resize(targetSize)
//    }
//    if (sizeAverage && total_weight != 0) {
//      output = ev.divide(output, total_weight)
//    }
//    output
//  }
//
//  override def classNLLupdateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
//    require(input.dim() == 1 || input.dim() == 2,
//      "ClassNLLCriterion: " +
//        ErrorInfo.constrainInputAsVectorOrBatch +
//        s"input dim ${input.dim()}")
//    assert(ev.toType[Double](total_weight) > 0.0, "total weight must larger than 0")
//    gradInput.resizeAs(input)
//    gradInput.zero()
//
//    if (input.dim() == 1) {
//      require(input.dim() == target.dim(),
//        "ClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget +
//          s" Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }")
//      val curTarget = ev.toType[Int](target.valueAt(1))
//      if (curTarget == -1) return gradInput
//      gradInput.setValue(curTarget, if (weights != null) ev.times(ev.fromType[Int](-1),
//        weights.valueAt(curTarget))
//      else ev.fromType[Int](-1))
//      if (sizeAverage) gradInput.setValue(curTarget, ev.divide(gradInput.valueAt(curTarget),
//        total_weight))
//    }
//    else if (input.dim() == 2) {
//      val batchSize = input.size(1)
//      val targetSize = target.size()
//      target.squeeze()
//      if (resultsBackward == null || resultsBackward.length != batchSize) {
//        resultsBackward = new Array[Future[_]](batchSize)
//      }
//
//      var i = 1
//      while (i <= batchSize) {
//        val _i = i
//        resultsBackward(_i - 1) = Engine.model.invoke(() => {
//          val curTarget = ev.toType[Int](target.valueAt(_i))
//          if (curTarget != -1) {
//            gradInput.setValue(_i, curTarget, if (weights != null) ev.times(ev.fromType[Int](-1),
//              weights.valueAt(curTarget))
//            else ev.fromType[Int](-1))
//            if (sizeAverage) gradInput.setValue(_i, curTarget, ev.divide(gradInput.valueAt(_i,
//              curTarget), total_weight))
//          }
//        })
//        i += 1
//      }
//
//      i = 0
//      while (i < batchSize) {
//        Await.result(resultsBackward(i), Duration.Inf)
//        i += 1
//      }
//      target.resize(targetSize)
//    }
//    gradInput
//  }
//}
//
//object TimeDistributedClassNLLWithMask {
//  def apply[@specialized(Float, Double) T: ClassTag](
//    weights: Tensor[T] = null,
//    sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : ClassNLLCriterion[T] = {
//    new ClassNLLCriterion[T](weights, sizeAverage)
//  }
//}
//
//private class ClassNLL[@specialized(Float, Double) T: ClassTag]
//(weights: Tensor[T] = null, sizeAverage: Boolean = true)
//  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
//  private var total_weight = ev.fromType[Int](0)
//  if (weights != null) require(weights.dim() == 1,
//    "weights input should be 1-D Tensor" +
//      s"weights dim(${weights.dim()})")
//
//  @transient
//  private var results: Array[Future[(T, T)]] = null
//  @transient
//  private var resultsBackward: Array[Future[_]] = null
//
//  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
//    require(input.dim() == 1 || input.dim() == 2,
//      "ClassNLLCriterion: " +
//        ErrorInfo.constrainInputAsVectorOrBatch +
//        s"input dim(${input.dim()})")
//    val nClasses = input.size(input.dim())
//    if (input.dim() == 1) {
//      require(input.dim() == target.dim(),
//        "ClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget +
//          s" Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }")
//      val curTarget = ev.toType[Int](target.valueAt(1))
//      assert(curTarget >= 1 && curTarget <= nClasses || curTarget == -1,
//        s"curTarget ${curTarget} is out of range, should be 1 to ${nClasses}")
//      total_weight = if (weights != null) weights(Array(curTarget)) else ev.fromType[Int](1)
//      output = if (curTarget == -1) ev.zero
//      else ev.times(ev.negative(input.valueAt(curTarget)), total_weight)
//    } else if (input.dim() == 2) {
//      val batchSize = input.size(1)
//      val targetSize = target.size()
//      target.squeeze()
//      require(target.dim() == 1,
//        "ClassNLLCriterion: illegal target! Target should be 1D tensor after squeeze," +
//          s"but target's size is: ${ target.size() }, please check your data.")
//
//      total_weight = ev.fromType[Int](0)
//      output = ev.fromType[Int](0)
//
//      if (results == null || results.length != batchSize) {
//        results = new Array[Future[(T, T)]](batchSize)
//      }
//
//      var i = 1
//      while (i <= batchSize) {
//        val _i = i
//        results(_i - 1) = Engine.model.invoke( () => {
//          val curTarget = ev.toType[Int](target.valueAt(_i))
//          assert(curTarget >= 1 && curTarget <= nClasses || curTarget == -1,
//            s"curTarget ${curTarget} is out of range 1 to ${nClasses}")
//          if (curTarget == -1) (ev.zero, ev.one)
//          else {
//            val curWeight = if (weights != null) weights.valueAt(curTarget) else ev.fromType[Int](1)
//            (ev.times(input.valueAt(_i, curTarget), curWeight), curWeight)
//          }
//        })
//        i += 1
//      }
//
//      i = 0
//      while (i < batchSize) {
//        val (o, w) = Await.result(results(i), Duration.Inf)
//        output = ev.negative(o)
//        total_weight = ev.plus(total_weight, w)
//        i += 1
//      }
//      target.resize(targetSize)
//    }
//    if (sizeAverage && total_weight != 0) {
//      output = ev.divide(output, total_weight)
//    }
//    output
//  }
//
//  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
//    require(input.dim() == 1 || input.dim() == 2,
//      "ClassNLLCriterion: " +
//        ErrorInfo.constrainInputAsVectorOrBatch +
//        s"input dim ${input.dim()}")
//    assert(ev.toType[Double](total_weight) > 0.0, "total weight must larger than 0")
//    gradInput.resizeAs(input)
//    gradInput.zero()
//
//    if (input.dim() == 1) {
//      require(input.dim() == target.dim(),
//        "ClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget +
//          s" Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }")
//      val curTarget = ev.toType[Int](target.valueAt(1))
//      if (curTarget == -1) return gradInput
//      gradInput.setValue(curTarget, if (weights != null) ev.times(ev.fromType[Int](-1),
//        weights.valueAt(curTarget))
//      else ev.fromType[Int](-1))
//      if (sizeAverage) gradInput.setValue(curTarget, ev.divide(gradInput.valueAt(curTarget),
//        total_weight))
//    }
//    else if (input.dim() == 2) {
//      val batchSize = input.size(1)
//      val targetSize = target.size()
//      target.squeeze()
//      if (resultsBackward == null || resultsBackward.length != batchSize) {
//        resultsBackward = new Array[Future[_]](batchSize)
//      }
//
//      var i = 1
//      while (i <= batchSize) {
//        val _i = i
//        resultsBackward(_i - 1) = Engine.model.invoke(() => {
//          val curTarget = ev.toType[Int](target.valueAt(_i))
//          if (curTarget != -1) {
//            gradInput.setValue(_i, curTarget, if (weights != null) ev.times(ev.fromType[Int](-1),
//              weights.valueAt(curTarget))
//            else ev.fromType[Int](-1))
//            if (sizeAverage) gradInput.setValue(_i, curTarget, ev.divide(gradInput.valueAt(_i,
//              curTarget), total_weight))
//          }
//        })
//        i += 1
//      }
//
//      i = 0
//      while (i < batchSize) {
//        Await.result(resultsBackward(i), Duration.Inf)
//        i += 1
//      }
//      target.resize(targetSize)
//    }
//    gradInput
//  }
//}
//