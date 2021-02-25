import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn as rcnn
from torchvision.models.detection import retinanet_resnet50_fpn as rnet
import torch
import imageio
import numpy as np
import glob
import cv2
from os.path import basename

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def imloader(fname):
    """ Load jpg image from fname """
    return np.asarray(imageio.imread(fname))

def mmn(ar):
    return (ar - ar.min())/(ar.max() - ar.min())

def dump(x, p, threshold=0.6, name="test.png"):
    """ dump out prediction (bbox) im to the logdir """
    # TODO get bbox labels
    for score, bbox, label_vector in zip(p["scores"], p["boxes"], p["labels"]):

        label = COCO_INSTANCE_CATEGORY_NAMES[label_vector.numpy()]

        if score > threshold:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(x, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(x, label, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    print(name)
    cv2.imwrite(name, x)

if __name__ == "__main__":
    # rcnn = faster-rcnn
    # rnet = retinanet
    #model = rcnn(pretrained=True)
    model = rnet(pretrained=True)
    model.eval() # processing mode

    fis = sorted(glob.glob("data/room0*.jpg"))

    raw_xs = np.array(list(map(imloader, fis)))
    xs = np.rollaxis(raw_xs, 3, 1)
    xs = np.array([[mmn(ch) for ch in x] for x in xs])
    xs = torch.from_numpy(xs).float()

    # this returns a list of dicts containing bboxes: ([x1, y1, x2, y2]) labels: (int64 corresponding to label in COCO_INSTANCE_CATEGORY_NAMES) and scores: (float between 0 and 1)
    ps = model(xs) 

    threshold = 0.5
    names = map(lambda s: "logs/" + basename(s).rsplit(".", 1)[0] + "_out.png", fis)
    for x, p, name in zip(raw_xs, ps, names):
        dump(x, p, threshold, name)
