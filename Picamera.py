#-*- coding: UTF-8 -*-#

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import tensorflow as tf
import time
import torch
from torch.autograd import Variable
import torch.nn as nn

from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import IPython

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SZ = 28
frame_width = 416
frame_height = 416
frame_resolution = [frame_width, frame_height]
frame_rate = 30
margin = 30
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))
# allow the camera to warmup
time.sleep(0.1)


# capture frames from the camera
for (num,frame) in enumerate(camera.capture_continuous(rawCapture, format="rgb", use_video_port=True)):
    
    model = Darknet('config/yolov3.cfg', img_size=416).to(device)

    
    model.load_darknet_weights("weights/yolov3.weights")
    model.eval()  # Set in evaluation mode

    classes = load_classes('data/coco.names')  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    print("\nPerforming object detection:")
    prev_time = time.time()

    image = frame.array
    # save frame
    img = Image.fromarray(image)
    new_path = os.path.join('data/samples','frame'+str(num)+'.jpg')
    img.save(new_path)

    dataloader = DataLoader(
        ImageFolder('data/samples', transform= \
            transforms.Compose([DEFAULT_TRANSFORMS, Resize(416)])),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    

    for _, (_, input_imgs) in enumerate(dataloader):
    # image: array to tensor
    # image = torch.tensor(image)
    # image = torch.unsqueeze(image, dim = 0).resize(1,3,416,416)
        
        input_imgs = Variable(input_imgs.type(Tensor))
    

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            # conf_thres: 0.95 , nms_thres: 0.01
            detections = non_max_suppression(detections, 0.7, 0.3)

            # top 5 confidence prediction values
            # detections = detections[:5,:]
            # IPython.embed(); exit(1);

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Inference Time: %s" % (inference_time))

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")

    # Create plot
    img = np.array(Image.open(new_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        
        detections = rescale_boxes(detections, 416, img.shape[:2])
        detections = detections.reshape(-1,7)
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)


        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            detections = np.maximum(0, detections)
            # IPython.embed(); exit(1);

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(new_path).split(".")[0]
    output_path = os.path.join("output", f"{filename}.jpg")
    plt.savefig(output_path,pad_inches=0)
    plt.close()

    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break