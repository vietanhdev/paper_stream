

## Installation

```bash
# Insert the v4l2loopback kernel module.
sh install.sh

sudo modprobe v4l2loopback devices=2 exclusive_caps=1 video_nr=3 card_label="PaperStreamCam" # will create two fake webcam devices
```