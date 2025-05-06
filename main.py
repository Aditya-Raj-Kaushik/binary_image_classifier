from utils import load_images
import kagglehub
import os
from model import MultiLayerNeuralNetwork

# Download and locate dataset
dataset_path = kagglehub.dataset_download("kipshidze/apple-vs-orange-binary-classification")

# The actual images are inside 'fruit-dataset' folder
data_path = os.path.join(dataset_path, "fruit-dataset")

# Load images
X, y = load_images(data_path, image_size=(64, 64))

print("Loaded images:", X.shape)
print("Loaded labels:", y.shape)
print("Sample labels:", y[:5])


model = MultiLayerNeuralNetwork(
    input_size=X.shape[1],
    hidden_size1=256,
    hidden_size2=128,
    hidden_size3=128,
    output_size=1,
    learning_rate=0.01
    
)


model.train(X, y, epochs=500)

final_acc = model.accuracy(X, y)
print(f"\nFinal training accuracy: {final_acc:.2f}%")