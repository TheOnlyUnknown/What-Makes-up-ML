import numpy as np
class Perceptron:
    def __init__(self, num_inputs, learning_rate = 0.1):
        self.weights = np.random.randn(num_inputs) * 0.1 #Weight input is between -1 to 1 like 68% of the time
        self.bias=0.0 # The threshold remains 0 until unless ran through a couple of times here epcohs is 20. 
        self.learning_rate = learning_rate # The change in weight and bias everytime the output is wrong. Low enough to not over change and big enough to change it appropriately


    def predict(self,x):
        z = np.dot(self.weights, x) + self.bias
        return 1 if z>0 else 0 

    def train(self, training_inputs, labels, epochs = 20):
        for epoch in range(epochs):
            total_errors = 0
            for x,y in zip(training_inputs, labels):
                x = np.array(x)
                y_hat = self.predict(x)
                error = y - y_hat

                if error != 0:
                    self.weights += self.learning_rate * error * x
                    self.bias += self.learning_rate * error
                    total_errors += 1

            if total_errors == 0:
                print(f"Converged after {epochs + 1} epochs.")
                break

inputs = [(0,0), (0,1), (1,0), (1,1)]
labels = [1,1,1,0]
p =Perceptron(num_inputs = 2)
p.train(inputs, labels, epochs = 50)

for x,y in zip(inputs, labels):
    prediction = p.predict(np.array(x))
    print(x, "-> predicted:", prediction, "expected: ", y)
