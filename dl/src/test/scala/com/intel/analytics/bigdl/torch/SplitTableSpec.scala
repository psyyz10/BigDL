/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn.SplitTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class SplitTableSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A SplitTable selects a tensor as an output" should "generate correct output and grad" in {
    val seed = 100
    Random.setSeed(seed)

    val module = new SplitTable[Double](1, 2)
    val input = Tensor[Double](3, 5).randn()
    val grd1 = Tensor[Double](5).randn()
    val grd2 = Tensor[Double](5).randn()
    val grd3 = Tensor[Double](5).randn()
    val gradOutput = T(1.0 -> grd1, 2.0 -> grd2, 3.0 -> grd3)
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)

    val code =
      s"""
      torch.manualSeed($seed)
      module = nn.SplitTable(1, 2)
      output1 = module:forward(input)[1]
      output2 = module:forward(input)[2]
      output3 = module:forward(input)[3]
      gradInput = module:backward(input, gradOutput)
               """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output1", "output2", "output3", "gradInput"))
    val torchOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val torchOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val torchOutput3 = torchResult("output3").asInstanceOf[Tensor[Double]]
    val torchOutput = T(torchOutput1, torchOutput2, torchOutput3)
    val torchgradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    torchOutput should be(output)
    torchgradInput should be(gradInput)
  }
}