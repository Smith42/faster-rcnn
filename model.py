import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn as rcnn
import torch
import imageio
import numpy as np
import glob
import cv2

def imloader(fname):
    """ Load jpg image from fname """
    return np.asarray(imageio.imread(fname))

def mmn(ar):
    return (ar - ar.min())/(ar.max() - ar.min())

def dump(x, p):
    """ dump out prediction (bbox) im to the logdir """
    # TODO get bbox overlay working
    # TODO get bbox labels via mpl
    # TODO get im dumping working
    for i in range(len(p["boxes"])):
        x1, y1, x2, y2 = map(int, p
    return 

if __name__ == "__main__":
    model = rcnn(pretrained=True)
    model.eval() # processing mode

    fis = sorted(glob.glob("data/livingroom/room*.jpg"))[:5]

    xs = np.array(list(map(imloader, fis)))
    # This is messy TODO fix:
    xs = np.rollaxis(xs, 3, 1)
    xs = np.array([[mmn(ch) for ch in x] for x in xs])
    xs = torch.from_numpy(xs).float()

    ps = model(xs) # this returns a list of dicts containing bboxes: ([x1, y1, x2, y2]) labels: (int64) and scores: (float between 0 and 1?)

    plt.imshow(np.rollaxis(xs.numpy()[0], 0, 3))
    plt.savefig("test.png")

    #np.save("logs/outs.npy", ps)
