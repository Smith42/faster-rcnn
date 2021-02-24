from torchvision.models.detection import fasterrcnn_resnet50_fpn as rcnn
import torch
import numpy as np

if __name__ == "__main__":
    model = rcnn(pretrained=True)

    xs = np.load("data/ims.npy")

    ps = model(xs) # this returns a list of dicts containing bboxes: ([x1, y1, x2, y2]) labels: (int64) and scores: (float between 0 and 1?)

    #np.save("logs/outs.npy", ps)
