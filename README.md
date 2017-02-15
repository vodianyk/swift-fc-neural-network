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

The [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) cost function is used by default. The [quadratic](https://en.wikipedia.org/wiki/Loss_function#Quadratic_loss_function) cost function can be set manually once a neural network is initialized:


```
let nn = FullyConnectedNeuralNetwork(shape: [2, 3, 1])
nn.costFunction = QuadraticCost()
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.txt) file for details