//
//  CostFunction.swift
//  BPNeuralNetwork
//
//  Created by Dmitry Vodianyk on 2/15/17.
//  Copyright © 2017 Dmitry Vodianyk. All rights reserved.
//

import Foundation

protocol CostFunction {
    func calculate(output: Matrix, desiredOutput: Matrix) -> Double
    func delta(output: Matrix, desiredOutput: Matrix, zValue: Matrix) -> Matrix
}

class QuadraticCost: CostFunction {
    func calculate(output: Matrix, desiredOutput: Matrix) -> Double {
        return 0.5 * ((desiredOutput - output).pow2().sumElements())
    }
    func delta(output: Matrix, desiredOutput: Matrix, zValue: Matrix) -> Matrix {
        return (output - desiredOutput) * zValue.sigmoidPrime()
    }
}

class CrossEntropyCost: CostFunction {
    func calculate(output: Matrix, desiredOutput: Matrix) -> Double {
        let cost = -1 * desiredOutput * output.ln()
        return (cost - (1 - desiredOutput) * (1 - output).ln()).sumElements()
    }
    func delta(output: Matrix, desiredOutput: Matrix, zValue: Matrix) -> Matrix {
        return (output - desiredOutput)
    }
}
