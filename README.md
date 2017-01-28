# swift-fc-neural-network
A Fully Connected Neural Network in Swift based on [vDSP](https://developer.apple.com/reference/accelerate/vdsp). Using a backpropagation algorithm to train.

# Code Example

```
let input = Matrix(4, 2, [
    0, 0,
    0, 1,
    1, 0,
    1, 1
])

let desiredOutput = Matrix(4, 1, [
    0,
    1,
    1,
    0
])

let nn = FullyConnectedNeuralNetwork(shape: [2, 3, 1])
print(nn.run(input))

nn.train(input, desiredOutput, minibatch: input.count.rows, nEpochs: 100000, learningRate: 0.3) { 
	(iteration, error, isFinished) in
	
    print("iteration: \(iteration), error: \(error)")
    if (isFinished) {
        print(nn.run(input))
    }
}

```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details