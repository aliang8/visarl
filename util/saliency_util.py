import cv2
import json
import numpy as np
from tqdm import tqdm


def gaussian_mask(sizex, sizey, sigma=33, center=None, fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)

    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0]) == False and np.isnan(center[1]) == False:
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey, sizex))

    return fix * np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)


def fixation_to_dense_map(
    fix_arr, width, height, imgfile, alpha=0.5, threshold=10, sigma=10, color=True
):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
    threshold : heatmap threshold(0~255)
    return heatmap
    """

    heatmap = np.zeros((height, width), np.float32)
    for n_subject in range(fix_arr.shape[0]):
        heatmap += gaussian_mask(
            width,
            height,
            sigma,
            (fix_arr[n_subject, 0], fix_arr[n_subject, 1]),
            fix_arr[n_subject, 2],
        )

    # Normalization
    heatmap = heatmap / np.amax(heatmap)
    heatmap = heatmap * 255
    heatmap = heatmap.astype("uint8")

    if not color:
        return heatmap

    if imgfile.any():
        # Resize heatmap to imgfile shape
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Create mask
        mask = np.where(heatmap <= threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images
        marge = imgfile * mask + heatmap_color * (1 - mask)
        marge = marge.astype("uint8")
        marge = cv2.addWeighted(imgfile, 1 - alpha, marge, alpha, 0)
        return marge, heatmap
    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap


def get_heatmap_from_ann_file(annotations_file, size=(224, 224), color=False):
    with open(annotations_file) as f:
        ann = f.read()
        ann = json.loads(ann)
        data = ann["data"]
        fix_arr = np.array([[int(d["x"]), int(d["y"]), int(d["value"])] for d in data])

        img = np.zeros((400, 400, 3)).astype("uint8")  # placeholder
        H, W, _ = img.shape

        # Create heatmap
        heatmap = fixation_to_dense_map(
            fix_arr, W, H, img, alpha=0.7, threshold=10, sigma=25, color=color
        )

        # resize heatmap to original image size
        resized_heatmap = cv2.resize(heatmap, size, interpolation=cv2.INTER_AREA)
    return resized_heatmap
