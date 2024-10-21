import kagglehub

# Specify the download path
custom_path = "data"  # Replace with your desired path

# Download latest version to the specified path
path = kagglehub.dataset_download("jessicali9530/caltech256")

print("Path to dataset files:", path)
