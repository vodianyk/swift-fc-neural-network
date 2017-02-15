//
//  FullyConnectedNeuralNetwork.swift
//  BPNeuralNetwork
//
//  Created by Dmitry Vodianyk on 1/18/17.
//  Copyright © 2017 Dmitry Vodianyk. All rights reserved.
//

import Foundation

class FullyConnectedNeuralNetwork {
    
    var weights = [Matrix]()
    var costFunction: CostFunction
    
    init(shape: [Int]) {
        var previous = -1
        for (index, value) in shape.enumerated() {
            if (index > 0) {
                let w = Matrix(previous, value)
                self.weights.append(w)
            }
            previous = value
        }
        self.costFunction = CrossEntropyCost()
    }
    
    func run(_ input: Matrix) -> Matrix {
        var result = input
        for w in self.weights {
            result = result.dot(w).sigmoid()
        }
        return result
    }

    func train(_ input: Matrix, _ desiredOutput: Matrix, minibatch: Int = input.count.rows, nEpochs: Int = 1000, learningRate: Double = 0.1,
               statusCallback: @escaping (_ epochIndex: Int, _ error: Double, _ finished: Bool) -> ()) {

        DispatchQueue.global().async(execute: {
            for epochIndex in 1 ... nEpochs {
                var error = 0.0
                let batchSize = input.count.rows % minibatch == 0 ? minibatch : input.count.rows
                let nMinibatches = (input.count.rows / batchSize)

                for j in 0 ..< nMinibatches {
                    let inputRowPosition = j * batchSize * input.count.columns
                    let batchInput = Matrix(batchSize, input.count.columns,
                                        Array(input.grid[inputRowPosition ..< inputRowPosition + batchSize * input.count.columns]))

                    let outputRowPosition = j * batchSize * desiredOutput.count.columns
                    let batchOutput = Matrix(batchSize, desiredOutput.count.columns,
                                        Array(desiredOutput.grid[outputRowPosition ..< outputRowPosition + batchSize * desiredOutput.count.columns]))

                    error += self.trainEpoch(batchInput, batchOutput, learningRate)
                }
                DispatchQueue.main.sync(execute: { () -> Void in
                    statusCallback(epochIndex, error, epochIndex == nEpochs)
                })
            }
        })
    }
    
    // MARK: Private
    
    private func trainEpoch(_ input: Matrix, _ desiredOutput: Matrix, _ learningRate: Double) -> Double {
        // feedforward
        var activations = [ input ]
        var zVectors = [Matrix]()
        var activation = input
        for w in self.weights {
            let z = activation.dot(w)
            zVectors.append(z)
            activation = z.sigmoid()
            activations.append(activation)
        }
        
        // backpropagation
        var delta = self.costFunction.delta(output: activations.last!, desiredOutput: desiredOutput, zValue: zVectors.last!)
        var gradients = [ activations[activations.count - 2].T().dot(delta) ]
        
        for i in 2 ..< self.weights.count + 1 {
            let z = zVectors[zVectors.count - i]
            let sP = z.sigmoidPrime()
            delta = delta.dot(self.weights[self.weights.count - i + 1].T()) * sP
            let gradient = activations[activations.count - i - 1].T().dot(delta)
            gradients.append(gradient)
        }
        
        for (index, gradient) in gradients.reversed().enumerated() {
            self.weights[index] = self.weights[index] - (gradient * learningRate)
        }
        
        return self.costFunction.calculate(output: activations.last!, desiredOutput: desiredOutput)
    }
}
