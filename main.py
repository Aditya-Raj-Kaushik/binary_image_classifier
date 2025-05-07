from utils import load_images
import kagglehub
import os
from model import MultiLayerNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle  
# Downloading and locating dataset
dataset_path = kagglehub.dataset_download("kipshidze/apple-vs-orange-binary-classification")
data_path = os.path.join(dataset_path, "fruit-dataset")

# Loading images
X, y = load_images(data_path, image_size=(64, 64))
print("Loaded images:", X.shape)
print("Loaded labels:", y.shape)
print("Sample labels:", y[:5])

# Normalizing features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initializing model
model = MultiLayerNeuralNetwork(
    input_size=X.shape[1],
    hidden_layers=[128, 64, 64, 32, 32],
    output_size=1,
    learning_rate=0.005
)

# Training
model.train(X_train, y_train, epochs=500, early_stopping=True, patience=30)

# Evaluation
train_acc = model.accuracy(X_train, y_train)
test_acc = model.accuracy(X_test, y_test)
y_test_pred = model.predict(X_test)

print(f"\nTraining Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#  Saving model and scaler as .pkl 
with open('saved_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)
print("\nModel and scaler saved as 'saved_model.pkl'")
