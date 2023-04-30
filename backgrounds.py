import cv2
import numpy as np
from skimage import io
from PIL import Image
from PyQt5 import QtCore, QtGui
from pathlib import Path
import json
import colorsys
import os
import shutil
import yellowbrick.cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import conversion
from unittest.mock import patch
from yellowbrick.utils import KneeLocator
from functools import partial
from multiprocessing import Manager, Pool

MAXK = 6
output = 'out'
input = '/home/archrichard/Pictures/5120x1440'
gamma = 1
lut = [pow(x/255.0, gamma)*255 for x in range(256)]
colors_dict = Manager().dict()
NOLIME = True

# Fast transform method
def get_scaled_color(pic):
    image = QtGui.QImage(str(pic))
    scaled_image = image.scaled(
        1, 1, aspectRatioMode=QtCore.Qt.IgnoreAspectRatio, transformMode=QtCore.Qt.FastTransformation)
    color = QtGui.QColor(scaled_image.pixel(0, 0)).getRgb()[:-1]
    return color

def correct_lime(color):
    if sum(color) >= 160 and color[1]-color[2]-color[0] >= 100:
        return [val*0.7 for val in color]


def correct_lime(color):
    h, s, v = colorsys.rgb_to_hsv(*map(lambda x: x/255, color))
    h *= 360
    print(h,s,v)
    if h >= 30 and h <= 100 and s > 0.45 and v > 0.4:
        newcolor = tuple(
            map(lambda x: round(x*255), colorsys.hsv_to_rgb((h+10)/360, s*.65, v*.9)))
        print(f"yellow changed from: {color} to {newcolor}")
        return newcolor
    if h >= 80 and h <= 200 and s > 0.4 and (v-max(h-80, 0)/500) > 0.5:
        newcolor = tuple(
            map(lambda x: round(x*255), colorsys.hsv_to_rgb(h/360, s*.85, v*.6)))
        print(f"lime changed from: {color} to {newcolor}")
        return newcolor
    return color

def darken(color):
    return tuple(lut[val] for val in color)

# Average color method
def get_average_color(pic):
    img = io.imread(pic)
    average = img.mean(axis=0).mean(axis=0)
    print(average)
    return tuple(round(x) for x in average)


# Initial naive k-means
def get_dominant_color(pic, all=False):
    img = io.imread(pic)
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    # dominant = palette[np.argmax(counts)]
    if all:
        sorted_palette = [x[1] for x in sorted(enumerate(map(lambda x: tuple(
            round(i) for i in x), palette)), key=lambda x: counts[x[0]])]
        return sorted_palette
    else:
        return palette[np.argmax(counts)]

# K means method with sillouhette hyperparameter. Returns a palette of colors. The elbow function below can be adapted to do the same but better by providing arguments to KElbowVisualizer.
def sillouhette_palette(pic, max_k=10):
    img = io.imread(pic)
    data = np.float32(img.reshape(-1, 3))
    data = conversion.rgb_to_lab(data)

    best_score = -1
    best_k = 0
    best_labels = []

    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        print(score)

        if score > best_score:
            best_score = score
            # best_k = k
            best_labels, best_centers = labels, kmeans.cluster_centers_

    _, counts = np.unique(best_labels, return_counts=True)
    sorted_palette = [x[1] for x in sorted(enumerate(map(lambda x: tuple(
        round(i) for i in x), best_centers)), key=lambda x: counts[x[0]])]
    return sorted_palette

# Elbow palette with CIELAB colorspace
# The CIELAB colorspace is designed to be perceptually uniform, which means that the Euclidean distance between two colors in CIELAB space approximates the perceptual difference between those colors
def elbow_palette(pic, max_k):
    img = io.imread(pic)
    data = np.float32(img.reshape(-1, 3))
    data = conversion.rgb_to_lab(data)

    model = KMeans(init='k-means++', n_init=2)
    S = 1
    # Decrease sensitivity if elbow not found
    while S:
        with patch("yellowbrick.cluster.elbow.KneeLocator", partial(KneeLocator, S=S)):
            visualizer = yellowbrick.cluster.KElbowVisualizer(
                model, k=(2, max_k+1), metric='distortion',distance_metric='euclidean')
            plt.clf()
            res = visualizer.fit(data)        # Fit the data to the visualizer
            best_k = res.elbow_value_
        if best_k:
            plt.plot([],[], label = f"S = {S:.1f}")
            S=0
        else: S-=.2
    plt.legend(loc=7, fancybox=True, facecolor="white")
    res.show(outpath=f"{output}/graph/{pic.stem}.png")

    kmeans = KMeans(n_clusters=best_k,init='k-means++', n_init=3)
    labels = kmeans.fit_predict(data)
    _, counts = np.unique(labels, return_counts=True)
    print(counts)
    sorted_palette = [x[1] for x in sorted(enumerate(map(lambda x: tuple(
        round(i) for i in x), conversion.lab_to_rgb(kmeans.cluster_centers_))), key=lambda x: counts[x[0]])]
    if not res.elbow_value_:
        return sorted_palette[max_k//2:]
    return sorted_palette


# def process(pic):
#     print(pic)
#     q = get_dominant_color(pic)
#     q = correct_lime(q)
#     colors_dict[pic] = q
#     print(q)
#     old_im = Image.open(pic)
#     old_size = old_im.size
#     new_size = (5120, 2144)
#     new_im = Image.new("RGB", new_size, color=q)
#     box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
#     new_im.paste(old_im, box)
#     new_im.save(f'{output}/{pic.stem}.png')


def process(pic, nolime=NOLIME):
    print(pic)
    palette = elbow_palette(pic, max_k=MAXK)
    colors_dict[str(pic)] = tuple(palette)
    print(palette)
    for tag in range(len(palette)):
        q = palette.pop()
        old_im = Image.open(pic)
        old_size = old_im.size
        new_size = (5120, 2144)
        new_im = Image.new("RGB", new_size, color=q)
        box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
        new_im.paste(old_im, box)
        if nolime:
            qq = correct_lime(q)
            print(q)
            if qq == q:
                new_im.save(f'{output}/{pic.stem}_c{tag}.png')
            else:
                new_im.save(f'{output}/{pic.stem}-lime_c{tag}.png')
                print("lime:", q)
                old_im = Image.open(pic)
                old_size = old_im.size
                new_size = (5120, 2144)
                new_im = Image.new("RGB", new_size, color=qq)
                box = tuple((n - o + 1) // 2 for n, o in zip(new_size, old_size))
                new_im.paste(old_im, box)
                new_im.save(f'{output}/{pic.stem}-modifedlime_c{tag}.png')
        else:
            new_im.save(f'{output}/{pic.stem}_c{tag}.png')

def process_item(item):
    if item.suffix == '.jpg':
        if os.path.exists(f'{output}/0/{item.stem}.png'):
            print(item, 'exists')
            return
        process(item)

def scan(target):
    """Scans a directory
    and its sub-directories.
    """
    if os.path.exists(f"{output}/graph"):
        shutil.rmtree(f"{output}/graph")
    os.mkdir(f"{output}/graph")
    with Pool(processes=15, maxtasksperchild=1) as pool:
        pool.map(process_item, Path(target).glob("*"))


if __name__ == '__main__':
    scan(input)
    Path(f"{output}/data.json").write_text(json.dumps(colors_dict.copy()))
