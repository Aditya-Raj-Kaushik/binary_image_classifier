import numpy as np

class MultiLayerNeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size=1, learning_rate=0.5):
        # Initializing the weights for each layer using Xavier initialization function

        # Layer 1: Input -> Hidden layer-1 
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden_size1))
        
        # Layer 2: Hidden layer-1 -> Hidden layer-2 
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(1 / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        
        # Layer 3: Hidden layer-2 -> Hidden layer-3 
        self.W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(1 / hidden_size2)
        self.b3 = np.zeros((1, hidden_size3))
        
        # Output layer: Hidden layer-3 -> Output
        self.W4 = np.random.randn(hidden_size3, output_size) * np.sqrt(1 / hidden_size3)
        self.b4 = np.zeros((1, output_size))  

        self.lr = learning_rate

    # Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # ReLU activation function
    def relu(self, z):
        return np.maximum(0, z)
    
    # Softmax activation function (not used in final version)
    def softmax(self, z):
        exp_vals = np.exp(z - np.max(z, axis=1, keepdims=True)) 
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    # Forward pass
    def forward(self, X):
        # Layer 1
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)  # ReLU activation 
        
        # Layer 2
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)  # ReLU activation 
        
        # Layer 3
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.sigmoid(self.Z3)  # Sigmoid activation 
        
        # Output layer
        self.Z4 = self.A3 @ self.W4 + self.b4
        self.A4 = self.sigmoid(self.Z4) 
        return self.A4
    
    # Calculating binary cross-entropy loss
    def compute_loss(self, y_true, y_pred):
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss
    
    # Derivative of sigmoid function
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    # Derivative of ReLU 
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    # Backward pass
    def backward(self, X, y_true):
        m = X.shape[0]  
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        # Gradient for output layer
        dZ4 = self.A4 - y_true
        dW4 = self.A3.T @ dZ4 / m
        db4 = np.sum(dZ4, axis=0, keepdims=True) / m

        # Gradient for hidden layer-3 
        dA3 = dZ4 @ self.W4.T
        dZ3 = dA3 * self.sigmoid_derivative(self.A3)
        dW3 = self.A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Gradient for hidden layer-2
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Gradient for hidden layer-1 
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Updating weights and biases using gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)               
            loss = self.compute_loss(y, y_pred)     
            self.backward(X, y)                     
            if (epoch + 1) % 10 == 0:
                acc = self.accuracy(X, y)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

    # Predicting class labels 
    def predict(self, X):
        probs = self.forward(X)
        return (probs > 0.5).astype(int)

    # Calculating the accuracy
    def accuracy(self, X, y_true):
        preds = self.predict(X)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        return np.mean(preds == y_true) * 100
