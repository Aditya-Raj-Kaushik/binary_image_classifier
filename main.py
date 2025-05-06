import kagglehub

# Download latest version
path = kagglehub.dataset_download("kipshidze/apple-vs-orange-binary-classification")

print("Path to dataset files:", path)