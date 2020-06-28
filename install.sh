# Setup v4l2loopback

sudo apt-get install python-opencv # 10x performance improvement if installed (see below)
sudo apt-get install ffmpeg # useful for debugging

sudo modprobe -r v4l2loopback
sudo apt remove v4l2loopback-dkms

sudo apt-get install linux-generic
sudo apt install dkms

git clone https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback
make

#instal mod
sudo cp -R . /usr/src/v4l2loopback-1.1
sudo dkms add -m v4l2loopback -v 1.1
sudo dkms build -m v4l2loopback -v 1.1
sudo dkms install -m v4l2loopback -v 1.1
sudo reboot
