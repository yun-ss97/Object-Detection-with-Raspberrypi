# Object Detection with Raspberrypi

### Purpose
Object Detection with YOLOv3 and PiCamera in Raspberrypi

### Implementation in detail
There are several object detection models such as Faster R-CNN, FPN, and etc. 

Among these, I've selected YOLOv3 as baseline model because what I desired in this project was detection with real-time speed in the inference stage. 

In case of detection model, I used Pytorch Implemention of YOLOv3 from [[here]](https://github.com/eriklindernoren/PyTorch-YOLOv3).

The main work in this repository is detecting the object through PiCamera connected to Raspberrypi.


### How to detect object with PiCamera

* Pre-requisite

- Raspberry Pi 4B
- PiCamera

For those who are not familiar with using Raspberry Pi and PiCamera, check the [additional material](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera).


* Code Example

```python
python PiCamera.py
```


#### Credit

##### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
