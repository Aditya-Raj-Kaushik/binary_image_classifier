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
        self.b4 = np.zeros((1, output_size))  # Fixed shape to match output size

        self.lr = learning_rate

    # Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # ReLU activation function
    def relu(self, z):
        return np.maximum(0, z)
    
    # Softmax activation function
    def softmax(self, z):
        exp_vals = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    # Forward pass
    def forward(self, X):
        # Layer 1
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)  # ReLU activation 
        
        # Layer 2
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.softmax(self.Z2)  # Softmax activation
        
        # Layer 3
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.sigmoid(self.Z3)  # Sigmoid activation 
        
        # Output layer
        self.Z4 = self.A3 @ self.W4 + self.b4
        self.A4 = self.sigmoid(self.Z4)  # Sigmoid activation for output
        
        return self.A4
