from utils import load_images
import kagglehub
import os
from model import MultiLayerNeuralNetwork
from sklearn.model_selection import train_test_split

# Download and locate dataset
dataset_path = kagglehub.dataset_download("kipshidze/apple-vs-orange-binary-classification")

# The actual images are inside 'fruit-dataset' folder
data_path = os.path.join(dataset_path, "fruit-dataset")

# Load images
X, y = load_images(data_path, image_size=(64, 64))
print("Loaded images:", X.shape)
print("Loaded labels:", y.shape)
print("Sample labels:", y[:5])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize the model with 8 hidden layers
model = MultiLayerNeuralNetwork(
    input_size=X.shape[1],
    hidden_layers=[64, 32, 64, 32, 64, 32, 16, 8],
    output_size=1,
    learning_rate=0.01
)

# Train the model
model.train(X_train, y_train, epochs=300)

# Evaluate the model
train_acc = model.accuracy(X_train, y_train)
test_acc = model.accuracy(X_test, y_test)

print(f"\nTraining Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
