sudo apt-get update
sudo apt-get install libgl1-mesa-glx unzip
conda update pillow
# conda update torchvision
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install chardet

sudo apt install nvidia-cuda-toolkit

git remote set-url origin https://TheFatBlue:<your-token>@github.com/DHP-NCSU/CEAL.git

for file in gs*; do mv "$file" "${file}.log"; done