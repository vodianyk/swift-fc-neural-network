//
//  Matrix.swift
//  BPNeuralNetwork
//
//  Created by Dmitry Vodianyk on 1/18/17.
//  Copyright © 2017 Dmitry Vodianyk. All rights reserved.
//

import Foundation
import Accelerate

struct Matrix {
    var grid = [Double]()
    
    private var nRows = 0
    private var nColumns = 0
    
    var count: (rows: Int, columns: Int) {
        return (self.nRows, self.nColumns)
    }
    
    var countTotal: Int {
        return self.nRows * self.nColumns
    }
    
    init(_ nRows: Int, _ nColumns: Int) {
        for _ in 0 ..< nRows {
            for _ in 0 ..< nColumns {
                let value = 2.0 * Double(arc4random()) / Double(UINT32_MAX) - 1.0
                self.grid.append(value)
            }
        }
        self.nRows = nRows
        self.nColumns = nColumns
    }
    
    init() {
        self.init(0, 0)
    }
    
    init(_ nRows: Int, _ nColumns: Int, _ grid: [Double]) {
        self.grid = grid
        self.nRows = nRows
        self.nColumns = nColumns
    }
}

// MARK: Matrix operations

extension Matrix {
    func dot(_ other: Matrix) -> Matrix {
        var resultGrid = [Double](repeating : 0.0, count : self.count.rows * other.count.columns)
        vDSP_mmulD(self.grid, 1, other.grid, 1, &resultGrid, 1,
                   vDSP_Length(self.count.rows), vDSP_Length(other.count.columns), vDSP_Length(self.count.columns))
        return Matrix(self.count.rows, other.count.columns, resultGrid)
    }
    
    func sumElements() -> Double {
        var sum = 0.0
        vDSP_sveD(self.grid, 1, &sum, vDSP_Length(self.countTotal))
        return sum
    }
    
    func pow2() -> Matrix {
        var resultGrid = [Double](repeating: 0.0, count: self.countTotal)
        vDSP_vsqD(self.grid, 1, &resultGrid, 1, vDSP_Length(self.countTotal))
        return Matrix(self.count.rows, self.count.columns, resultGrid)
    }
    
    func T() -> Matrix {
        var resultGrid = [Double](repeating : 0.0, count : self.countTotal)
        vDSP_mtransD(self.grid, 1, &resultGrid, 1, vDSP_Length(self.count.columns), vDSP_Length(self.count.rows))
        return Matrix(self.count.columns, self.count.rows, resultGrid)
    }
    
    func ln() -> Matrix {
        var resultGrid = [Double](repeating: 0.0, count: self.countTotal)
        vvlog(&resultGrid, self.grid, [Int32(self.countTotal)])
        return Matrix(self.count.columns, self.count.rows, resultGrid)
    }
}

// MARK: operators

extension Matrix {
    static func - (left: Matrix, right: Matrix) -> Matrix {
        var resultGrid = [Double](repeating : 0.0, count : left.countTotal)
        vDSP_vsubD(right.grid, 1, left.grid, 1, &resultGrid, 1, vDSP_Length(left.countTotal))
        return Matrix(left.count.rows, left.count.columns, resultGrid)
    }
    static func - (left: Double, right: Matrix) -> Matrix {
        let leftGrid = [Double](repeating : left, count : right.countTotal)
        var resultGrid = [Double](repeating : 0.0, count : right.countTotal)
        vDSP_vsubD(right.grid, 1, leftGrid, 1, &resultGrid, 1, vDSP_Length(right.countTotal))
        return Matrix(right.count.rows, right.count.columns, resultGrid)
    }
    static func * (left: Matrix, right: Double) -> Matrix {
        var resultGrid = [Double](repeating : 0.0, count : left.countTotal)
        var s = right
        vDSP_vsmulD(left.grid, 1, &s, &resultGrid, 1, vDSP_Length(left.countTotal))
        return Matrix(left.count.rows, left.count.columns, resultGrid)
    }
    static func * (left: Double, right: Matrix) -> Matrix {
        return right * left
    }
    static func * (left: Matrix, right: Matrix) -> Matrix {
        var resultGrid = [Double](repeating : 0.0, count : left.countTotal)
        vDSP_vmulD(left.grid, 1, right.grid, 1, &resultGrid, 1, vDSP_Length(left.countTotal))
        return Matrix(left.count.rows, left.count.columns, resultGrid)
    }
}

// MARK: Node Transfer Functions

extension Matrix {
    func sigmoid() -> Matrix {
        let newValues = self.grid.map { 1.0 / (1.0 + exp(-$0)) }
        return Matrix(self.count.rows, self.count.columns, newValues)
    }
    func sigmoidPrime() -> Matrix {
        let newValues = self.grid.map { exp(-$0) / pow((1 + exp(-$0)), 2) }
        return Matrix(self.count.rows, self.count.columns, newValues)
    }
}

// MARK: description

extension Matrix: CustomStringConvertible {
    var description: String {
        var result = ""
        for i in 0 ..< self.count.rows {
            for j in 0 ..< self.count.columns {
                result += " \(self.grid[i * self.count.columns + j])"
            }
            result += "\n"
        }
        return result
    }
}

// MARK: subscript

extension Matrix {
    subscript(row: Int, column: Int) -> Double {
        get {
            assert(row > 0 && column > 0 && row < self.count.rows && column < self.count.columns, "Index out of range")
            return self.grid[row * self.count.columns + column]
        }
        set(newValue) {
            assert(row > 0 && column > 0 && row < self.count.rows && column < self.count.columns, "Index out of range")
            self.grid[row * self.count.columns + column] = newValue
        }
    }
    
    subscript(index: Int) -> [Double] {
        get {
            assert(index > 0 && index < self.count.rows, "Index out of range")
            return Array(self.grid[index * self.count.columns ..< index * self.count.columns + self.count.columns])
        }
        set(newValue) {
            assert(index > 0 && index < self.count.rows, "Index out of range")
            self.grid.replaceSubrange(index * self.count.columns ..< index * self.count.columns + self.count.columns, with: newValue)
        }
    }
}
