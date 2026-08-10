import numpy as np

class SigmoidNeuron:
    def __init__(self, num_inputs, learning_rate = 0.5):
        self.weight = np.random.randn(num_inputs) * 0.1
        self.bias = 0.0
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        z = np.dot(self.weight, x) + self.bias
        return self.sigmoid(z)

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def train(self,training_inputs, labels, epochs = 2000):
        for epoch in range(epochs):
            total_loss = 0
            for x,y in zip(training_inputs, labels):
                x = np.array(x)
                z = np.dot(self.weight, x) + self.bias
                a = self.sigmoid(z)
                error = a- y
                gradient = error * self.sigmoid_derivative(z)

                self.weight -= self.learning_rate * gradient * x
                self.bias -= self.learning_rate * gradient
                total_loss += 0.5 * (error ** 2)

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
labels = [1, 1, 1, 0]  

n = SigmoidNeuron(num_inputs = 2)
n.train(inputs, labels, epochs = 2000)

for x, y in zip(inputs, labels):
    raw = n.predict(np.array(x))
    thresholded = 1 if raw > 0.5 else 0
    print(f"{x[0]:>3} {x[1]:>3} | {raw:>10.4f} | {thresholded:>11} | {y}")