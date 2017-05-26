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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import util.control.Breaks._

class BinaryTreeLSTM[T](
  inputSize: Int,
  hiddenSize: Int,
  gateOutput: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends TreeLSTM[T](inputSize, hiddenSize) {
  val modules: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]]()
  val composers: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]]()
  val leafModules: ArrayBuffer[Module[T]] = ArrayBuffer[Module[T]]()
  val composer: Module[T] = createComposer()
  val leafModule: Module[T] = createLeafModule()
//  val outputModule: Module[T] = createOutputModule()
  val cells = ArrayBuffer[Module[T]]()

  def createLeafModule(): Module[T] = {
    val input = Identity().apply()
    val c = Linear(inputSize, hiddenSize).apply()
    var h: ModuleNode[T] = null
    if (gateOutput) {
      val o = Sigmoid().apply(Linear(inputSize, hiddenSize).apply(input))
      CMulTable().apply(o, Tanh().apply(c))
    } else {
      h = Tanh().apply(c)
    }

    val leafModule = Graph(Array(input), Array(c, h))

    if (this.leafModule != null) {
      shareParams(leafModule, this.leafModule)
    }

    leafModule
  }

  def createComposer(): Module[T] = {
    val (lc, lh) = (Identity().apply(), Identity().apply())
    val (rc, rh) = (Identity().apply(), Identity().apply())

    def newGate(): ModuleNode[T] = CAddTable().apply(
      Linear(hiddenSize, hiddenSize).apply(lh),
      Linear(hiddenSize, hiddenSize).apply(rh)
    )

    val i = Sigmoid().apply(newGate())
    val lf = Sigmoid().apply(newGate())
    val rf = Sigmoid().apply(newGate())
    val update = Tanh().apply(newGate())
    val c = CAddTable().apply(
      CMulTable().apply(i, update),
      CMulTable().apply(lf, lc),
      CMulTable().apply(rf, rc)
    )

    val h = if (this.gateOutput) {
      val o = Sigmoid().apply(newGate())
      CMulTable().apply(o, Tanh().apply(c))
    } else {
      Tanh().apply(c)
    }

    val composer = Graph(Array(lc, lh, rc, rh), Array(c, h))

    if (this.composers != null) {
      shareParams(composer, this.composer)
    }

    composer
  }

//  def createOutputModule(): Module[T] = {
//    if (outputModuleFun == null) return null
//
//    val outputModule = outputModuleFun()
//    if (this.outputModule != null) {
//      shareParams(outputModule, this.outputModule)
//    }
//
//    outputModule
//  }

  override def updateOutput(input: Table): Tensor[T] = {
    val tensorTree = new TensorTree[T](input(2))
    for (i <- tensorTree.size(1)) {
      if (tensorTree.noChild(i)) {
        val numLeafModules = leafModules.size
        if (numLeafModules == 0) {
          cells.append(createLeafModule())
        } else {
          cells.append(leafModules.remove(numLeafModules - 1))
        }
      } else {
        val numComposers = composers.size
        if (numComposers == 0) {
          cells.append(createComposer())
        } else {
          cells.append(composers.remove(numComposers - 1))
        }
      }
    }
    forward(input(1), tensorTree, tensorTree.getRoot())
    output.resize(cells.size, hiddenSize)
    for (i: Int <- cells.size) {
      output.select(1, i).copy(unpackState(cells(i - 1).output.toTable)._2)
    }
    output
  }


  def forward(input: Tensor[T], tree: TensorTree[T], nodeIndex: Int): Table = {
//    var (lLoss, rLoss) = (ev.zero, ev.zero)
    if (tree.noChild(nodeIndex)) {
      cells(nodeIndex - 1)
        .forward(input.select(1, tree.leafIndex(nodeIndex))).toTable
    } else {
      val leftOut = forward(input, tree, ev.toType[Int](tree.children(nodeIndex)(0)))
      val rigthOut = forward(input, tree, ev.toType[Int](tree.children(nodeIndex)(1)))
      val (lc, lh) = unpackState(leftOut)
      val (rc, rh) = unpackState(rigthOut)
//      lLoss = leftOut(2)
//      rLoss = rigthOut(2)

      cells(nodeIndex - 1).forward(T(lc, lh, rc, rh)).toTable
    }

//    var loss: T = ev.zero
//    if (outputModuleFun != null) {
//      val numOutputModules = outputModules.size
//      if (numOutputModules == 0) {
//        treeNode.outputModule = createOutputModule()
//      } else {
//        treeNode.outputModule = outputModules.remove(numOutputModules - 1)
//      }
//      treeNode.output = treeNode.outputModule.forward(treeNode.state(2))
//      if (train) {
//        loss = criterion.forward(treeNode.output, treeNode.label)
//        loss = ev.plus(loss, ev.plus(lLoss, rLoss))
//      }
//    }
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val tensorTree = new TensorTree[T](input(2))
    backward(input(1), tensorTree, gradOutput, tensorTree.getRoot())
    gradInput
  }

  def backward(
    inputs: Tensor[T],
    tree: TensorTree[T],
    gradOutput: Tensor[T],
    nodeIndex: Int
  ): Unit = {
    var outputGrad = memZero
    gradInput[Tensor[T]](1).resizeAs(inputs)

    if (tree.noChild(nodeIndex)) {
      gradInput[Tensor[T]](1)
        .select(1, tree.leafIndex(nodeIndex))
        .copy(
          cells(nodeIndex - 1)
            .backward(
              inputs.select(1, tree.leafIndex(nodeIndex)),
              T(gradOutput(1), gradOutput[Tensor[T]](2) + outputGrad)).toTensor)

      leafModules.append(cells(nodeIndex - 1))
      cells(nodeIndex - 1) = null
    } else {
      val (lc, lh, rc, rh) = getChildStates(treeNode)
      val composerGrad =
        treeNode
          .module
          .backward(T(lc, lh, rc, rh),
            T(gradOutput(1), gradOutput[Tensor[T]](2) + outputGrad)).toTable

      composers.append(treeNode.module)
      treeNode.module = null
      backward(treeNode.children(1), inputs,
        T(composerGrad(1), composerGrad(2)))
      backward(treeNode.children(2), inputs,
        T(composerGrad(3), composerGrad(4)))
    }

    treeNode.state = null
    treeNode.output = null
  }

  case class TreeNode(
    children: Array[TreeNode],
    leafIndex: Int,
    label: Activity,
    var state: Table,
    var output: Activity,
    var module: Module[T],
    var outputModule: Module[T]
  )

  def getChildStates(tree: TreeNode): (Tensor[T], Tensor[T], Tensor[T], Tensor[T]) = {
    val children = tree.children
    if (children.length > 1) {
      val (lc, lh) = unpackState(children(0).state)
      val (rc, rh) = unpackState(children(1).state)
      (lc, lh, rc, rh)
    } else {
      null
    }
  }

  def unpackState(state: Table): (Tensor[T], Tensor[T]) = {
    if (state.length() == 0) {
      (memZero, memZero)
    } else {
      (state(1), state(2))
    }
  }
}

object BinaryTreeLSTM {
  def apply[T](
    inputSize: Int,
    hiddenSize: Int,
    gateOutput: Boolean = true
  )(implicit ev: TensorNumeric[T]): BinaryTreeLSTM[T] =
    new BinaryTreeLSTM(inputSize, hiddenSize, gateOutput)
}

class TensorTree[T](val content: Tensor[T])
  (implicit ev: TensorNumeric[T]) {
  def size: Array[Int] = content.size()

  def children(index: Int): Array[T] = content.select(1, index).toBreezeVector().toArray

  def addChild(parent: Int, child: T): Unit = {
    for (i <- 1 to size(2)) {
      if (content(Array(parent, i)) == 0) {
        content.setValue(parent, i, child)
        break()
      }
    }
  }

  def markAsRoot(index: Int): Unit = {
    content.setValue(index, size(2), ev.negative(ev.one))
  }

  def getRoot(): Int = {
    for (i <- 1 to size(1)) {
      if (content(Array(i, size(2))) == -1) {
        return i
      }
    }

    -1
  }

  def markAsLeaf(index: Int, leafIndex: Int): Unit = {
    content.setValue(index, size(2), ev.fromType(leafIndex))
  }

  def leafIndex(index: Int): Int = {
    ev.toType[Int](content(Array(index, size(2))))
  }

  def hasChild(index: Int): Boolean = {
    content(Array(index, 1)) != 0
  }

  def noChild(index: Int): Boolean = {
    content(Array(index, 1)) == 0
  }
}

