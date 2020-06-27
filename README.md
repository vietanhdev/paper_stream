

## Installation

```bash
# python
pip install numpy

# linux
apt-get install v4l2loopback-utils

# linux (optional)
apt-get install python-opencv # 10x performance improvement if installed (see below)
apt-get install ffmpeg # useful for debugging

# Insert the v4l2loopback kernel module.
modprobe v4l2loopback devices=2 # will create two fake webcam devices
```

- Fix v4l

```
Ubuntu 18.04
Linux 5.3.0-46-generic

#remove apt package
sudo modprobe -r v4l2loopback
sudo apt remove v4l2loopback-dkms

#install aux
sudo apt-get install linux-generic
sudo apt install dkms

#install v4l2loopback from the repository
https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback
make

#instal mod
sudo cp -R . /usr/src/v4l2loopback-1.1
sudo dkms add -m v4l2loopback -v 1.1
sudo dkms build -m v4l2loopback -v 1.1
sudo dkms install -m v4l2loopback -v 1.1
sudo reboot
```
