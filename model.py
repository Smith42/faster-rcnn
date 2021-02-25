import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn as rcnn
import torch
import imageio
import numpy as np
import glob
import cv2
from os.path import basename

def imloader(fname):
    """ Load jpg image from fname """
    return np.asarray(imageio.imread(fname))

def mmn(ar):
    return (ar - ar.min())/(ar.max() - ar.min())

def dump(x, p, threshold=0.6, name="test.png"):
    """ dump out prediction (bbox) im to the logdir """
    # TODO get bbox labels
    for score, bbox in zip(p["scores"], p["boxes"]):
        if score > threshold:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(x, (x1, y1), (x2, y2), (255, 0, 0), 1)

    print(name)
    cv2.imwrite(name, x)

if __name__ == "__main__":
    model = rcnn(pretrained=True)
    model.eval() # processing mode

    fis = sorted(glob.glob("data/room*.jpg"))

    raw_xs = np.array(list(map(imloader, fis)))
    xs = np.rollaxis(raw_xs, 3, 1)
    xs = np.array([[mmn(ch) for ch in x] for x in xs])
    xs = torch.from_numpy(xs).float()

    ps = model(xs) # this returns a list of dicts containing bboxes: ([x1, y1, x2, y2]) labels: (int64) and scores: (float between 0 and 1?)

    threshold = 0.6
    names = map(lambda s: "logs/" + basename(s).rsplit(".", 1)[0] + "_out.png", fis)
    for x, p, name in zip(raw_xs, ps, names):
        dump(x, p, threshold, name)
