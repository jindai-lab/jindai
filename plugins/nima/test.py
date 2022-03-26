import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from .model import *

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])

base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_state(model_path=None):
    if not model_path:
        model_path = glob.glob(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.pkl')
        )[0]
    global model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()


def load_image(img):
    im = Image.open(img) if isinstance(img, str) else img
    if len(im.getbands()) != 3:
        im2 = Image.new('RGB', im.size)
        im2.paste(im)
        return im2
    return im


def predict(images):
    for img in images:
        try:
            im = load_image(img)
            imt = test_transform(im)
            imt = imt.unsqueeze(dim=0)
            imt = imt.to(device)
            with torch.no_grad():
                out = model(imt)
            out = out.view(10, 1)
            mean = 0
            for j, e in enumerate(out, 1):
                mean += j * e.item()
            yield (img, mean)
        except:
            yield img, -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to pretrained model')
    parser.add_argument('--test_csv', type=str, help='test csv file')
    parser.add_argument('--test_images', type=str, help='path to folder containing images')
    parser.add_argument('--out', type=str, help='dest for images with predicted score')
    args = parser.parse_args()

    try:
        load_state(args.model)
        print('successfully loaded model')
    except:
        raise

    test_imgs = sorted([os.path.join(args.test_images, f) for f in os.listdir(args.test_images)])

    if args.out and not os.path.exists(args.out):
        os.makedirs(args.out)

    for i, (img, score) in enumerate(predict(test_imgs)):
        if args.test_csv:
            gt = test_df[test_df[0] == int(img.split('.')[0])].to_numpy()[:, 1:].reshape(10, 1)
            gt_mean = 0.0
            for l, e in enumerate(gt, 1):
                gt_mean += l * e
            print(img.split('.')[0] + ' mean: %.3f | GT: %.3f' % (mean, gt_mean))
        else:
            print(img, score)
        if args.out:
            im = Image.open(img) if isinstance(img, str) else img
            plt.imshow(im)
            plt.axis('off')
            plt.title('%.3f' % (score))
            plt.savefig(os.path.join(args.out, os.path.basename(img) + '_predicted.png'))
