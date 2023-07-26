# PaperStream - Online Paper Streaming for remote education

Blog post: [https://aicurious.io/blog/2020-09-02-xay-dung-giai-phap-stream-giay-viet-ibm-hackathon-2020](https://aicurious.io/blog/2020-09-02-xay-dung-giai-phap-stream-giay-viet-ibm-hackathon-2020).

https://github.com/vietanhdev/paper_stream/assets/18329471/b1b62fb3-609d-4107-b6da-eeb99dbff5c2

This project is the source code for [IBM Hackathon at SoICT](https://www.facebook.com/events/733476740727552/). My team got **Second Place**. Note that our solution has 2 parts: (1) paper transformation and (2) video conference website. This repository only contains the first part. In the second part, we build a streaming website using Flask + ReactJS with video streaming solution from <https://vidyo.io/>.

Currently we don't have time for this project. However, we will add more info later.

## Setup virtual camera

```bash
# Insert the v4l2loopback kernel module.
sh install.sh

sudo modprobe v4l2loopback devices=2 exclusive_caps=1 video_nr=3 card_label="PaperStreamCam" # will create two fake webcam devices
```
