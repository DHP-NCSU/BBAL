import os
import pickle
import torchvision

# Create directories
os.makedirs('downloads/cifar100', exist_ok=True)
os.makedirs('data/cifar100/train', exist_ok=True)
os.makedirs('data/cifar100/test', exist_ok=True)

# Download CIFAR-100 to downloads directory
trainset = torchvision.datasets.CIFAR100(root='./downloads/cifar100', train=True, download=True)
testset = torchvision.datasets.CIFAR100(root='./downloads/cifar100', train=False, download=True)

# Save as pickle files in data directory
train_data = {
    'data': trainset.data.transpose(0, 3, 1, 2),
    'fine_labels': trainset.targets
}

test_data = {
    'data': testset.data.transpose(0, 3, 1, 2),
    'fine_labels': testset.targets
}

with open('data/cifar100/train/train', 'wb') as f:
    pickle.dump(train_data, f)

with open('data/cifar100/test/test', 'wb') as f:
    pickle.dump(test_data, f)

# Clean up downloads directory
import shutil
shutil.rmtree('downloads/cifar100')

print("CIFAR-100 dataset downloaded and processed successfully!")