import cv2
import os

from sklearn.metrics import jaccard_score, f1_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import random
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

import pywt

import numpy as np

def load_data(path):
    img_lst, masks_lst = [], []
    for filename in os.listdir(path + "images/"):
        img_lst += [cv2.imread(path + "images/" + filename) ]
        masks_lst += [cv2.imread(path + "masks/" + filename)]
    return img_lst, masks_lst



class PreprocessingUnit:
    def __init__(self, data, masks):
        self.masks = masks
        self.rgb_data = data
        self.raw_data = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), data))


    def _find_valley_point(self, hist):
        n = 2
        for key, value in enumerate(hist):
            if key > n-1 and key < hist.size - n and min(hist[key-n : key+n+1]) < min([hist[i] for i in range(key-n, key+n+1) if i!=key]):
                return key
        raise Exception("no valley point")


    def blurring(self):
        kernel_size = 15

        return list(map(lambda img: cv2.medianBlur(img, kernel_size), self.raw_data))

    def binarization(self, data):
        # cv2.namedWindow('sd', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)

        ret_lst = list()
        closing_kernel = np.ones((16 , 16), np.uint8)

        for img in data:
            opening_kernel = np.ones((int(img.shape[0] / 100) + 1 , int(img.shape[1] / 100) + 1), np.uint8)

            h = np.histogram(img.ravel(), 256, [0, 256])[0]
            ret, thresh1 = cv2.threshold(img, self._find_valley_point(h), 255, cv2.THRESH_BINARY)

            closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, closing_kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, opening_kernel)
            ret_lst.append(opening)

            # cv2.imshow("sd", opening)
            # cv2.imshow("img", img)
            #
            # cv2.waitKey(0)

        return ret_lst

    def segmentation(self, data):

        ret_lst = list()
        for img, init_img, img_true in zip(data, self.rgb_data, self.masks):
            trans = cv2.distanceTransform(img, distanceType=cv2.DIST_L2,maskSize=5)
            normalized = cv2.normalize(trans, None, 0, 255, cv2.NORM_MINMAX)


            coeffs2 = pywt.dwt2(normalized, 'bior1.3')
            LL, (LH, HL, HH) = coeffs2

            # LL = normalized
            opening_kernel = np.ones((int(img.shape[0] / 100) + 1, int(img.shape[1] / 100) + 1), np.uint8)
            opening = cv2.morphologyEx(LL, cv2.MORPH_OPEN, opening_kernel)
            opening = cv2.resize(opening, img.shape)
            norm_image = cv2.normalize(opening, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

            ret, thresh2 = cv2.threshold(norm_image, 1, 255, cv2.THRESH_BINARY)



            _, contours, _ = cv2.findContours(normalized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            markers = np.full(img.shape, 0, dtype=np.int32)

            for i in range(len(contours)):
                cv2.drawContours(image=markers,contours=contours, contourIdx=i, color=i+1, thickness=-1)

            cv2.circle(markers, (1, 1), 3, (255, 255, 255), -1)

            markers = cv2.normalize(markers, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F).astype(np.int32)
            cv2.watershed( init_img, markers)


            # mark = markers.astype('uint8')

            img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)
            img_true[np.where((img_true != 0).any(axis=1))] = 255
            img_true = np.array(img_true).ravel()
            img_pred = np.array(thresh2).ravel()
            iou = f1_score(img_true, img_pred, pos_label=0)
            ret_lst.append(iou)
        print(ret_lst)
        return sum(ret_lst)/len(ret_lst)
