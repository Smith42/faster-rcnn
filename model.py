from torchvision.models.detection import fasterrcnn_resnet50_fpn as rcnn
import torch
import imageio
import numpy as np
import glob

def imloader(fname):
    """ Load jpg image from fname """
    return np.asarray(imageio.imread(fname))

def mmn(ar):
    return (ar - ar.min())/(ar.max() - ar.min())

if __name__ == "__main__":
    model = rcnn(pretrained=True)
    model.eval() # processing mode

    fis = np.random.permutation(glob.glob("data/livingroom/*.jpg"))[:5]

    xs = np.array(list(map(imloader, fis)))
    # This is messy TODO fix:
    xs = np.array([[mmn(ch) for ch in np.swapaxes(x, 0, -1)] for x in xs])
    xs = torch.from_numpy(xs).float()

    ps = model(xs) # this returns a list of dicts containing bboxes: ([x1, y1, x2, y2]) labels: (int64) and scores: (float between 0 and 1?)

    #np.save("logs/outs.npy", ps)
