sudo apt-get --purge remove cuda nvidia* libnvidia-*
dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
sudo apt-get remove cuda-*
sudo apt autoremove
sudo apt-get update

#Download CUDA 10.0
wget  --no-clobber https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
#install CUDA kit dpkg
dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-10-0
#Slove libcurand.so.10 error
wget --no-clobber http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
#-nc, --no-clobber: skip downloads that would download to existing files.
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
####################################################################

git_repo_url = 'https://github.com/TadasBaltrusaitis/OpenFace.git'
project_name = splitext(basename(git_repo_url))[0]
# clone openface
git clone -q --depth 1 $git_repo_url

# install new CMake becaue of CUDA10
wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz
tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local

# Get newest GCC
sudo apt-get update
sudo apt-get install build-essential 
sudo apt-get install g++-8

#added 5/15/2022. Thanks to @weskhoo
sudo apt-key del 7fa2af80
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

# install python dependencies
pip install -q youtube-dl

# Finally, actually install OpenFace
cd OpenFace && bash ./download_models.sh && sudo bash ./install.sh
