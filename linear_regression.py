import numpy as np
import matplotlib.pyplot as plt
    
class LinearRegression:
    """_summary_

    X [numpy array]: data points
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.cost_history = []


    def predict(self, X): #hypothesis
        #y = ax + b
        #y = bo + b1.x1 + b2.x2 + ... + bnxn
        return self.bias + np.dot(X, self.weights)
    
    def cost_function(self, X, y):
        m = len(y)
        predictions = self.predict(X)
        
        mse = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

        return mse
    
    def gradient_descent(self, X, y):
        m = len(y)
        predictions = self.predict(X)

        #calculate gradients
        #dj/dw = (1 / m) * Σ((h(x) - y) * x)
        # Gradient for feature weights
        dj = (1 / m) * np.dot(X.T, (predictions - y))
        
        #db/dw = (1 / m) * Σ((h(x) - y)))
        # Gradient for intercept (bias)
        db = (1 / m) * np.sum(predictions - y) #derivative of mse
        
        #Update params
        self.weights -= self.learning_rate * dj
        self.bias -= self.learning_rate * db
    
    def fit(self, X, y):
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0
        batch = 100

        for i in range(self.n_iterations):
            self.gradient_descent(X, y)
            cost = self.cost_function(X, y)
            self.cost_history.append(cost)
            
            if i % batch == 0:
                print(f"Iteration {i}: Cost = {cost}")



if __name__ == "__main__":
    np.random.seed(0) #randomness is the same for every call

    X = 2 * np.random.rand(100, 1)
    y = 5 + 3 * X + np.random.rand(100, 1)

    #add ones column to X
    X_b = np.c_[X]
    # print(X)
    # print(y)

    
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_b, y) 

    print("Learned intercept:", model.bias)
    print("Learned weights:", model.weights)

    plt.plot(model.cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost History")
    plt.show()

    test_data = np.array([[1.5], [3.0], [5.0]])  # Example test inputs
    predictions = model.predict(test_data)
    print("Predictions on test data:", predictions)


