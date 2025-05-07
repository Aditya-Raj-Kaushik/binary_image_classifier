import numpy as np

class MultiLayerNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size=1, learning_rate=0.01):
        self.layers = []
        self.biases = []
        self.lr = learning_rate
        
        # Layer sizes
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Xavier initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append(w)
            self.biases.append(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.Z = []
        self.A = [X]
        
        for i in range(len(self.layers) - 1):
            z = self.A[-1] @ self.layers[i] + self.biases[i]
            self.Z.append(z)
            self.A.append(self.relu(z))
        
        # Output layer (sigmoid)
        z_out = self.A[-1] @ self.layers[-1] + self.biases[-1]
        self.Z.append(z_out)
        self.A.append(self.sigmoid(z_out))
        
        return self.A[-1]

    def compute_loss(self, y_true, y_pred):
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

    def backward(self, X, y_true):
        m = X.shape[0]
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        dZ = self.A[-1] - y_true
        
        for i in reversed(range(len(self.layers))):
            dW = self.A[i].T @ dZ / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            self.layers[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

            if i > 0:
                dA = dZ @ self.layers[i].T
                dZ = dA * self.relu_derivative(self.Z[i-1])

    def train(self, X, y, epochs=300):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y)
            if (epoch + 1) % 10 == 0:
                acc = self.accuracy(X, y)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def accuracy(self, X, y_true):
        preds = self.predict(X)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        return np.mean(preds == y_true) * 100
