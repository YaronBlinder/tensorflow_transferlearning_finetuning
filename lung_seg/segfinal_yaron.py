import glob
import os
import time

import cv2
import numpy as np
import scipy.misc
from numba import guvectorize
from skimage import color
from sklearn.cluster import KMeans

RblackHatKernel=177
RmaxCorners=2300
RCradius=6
Rthreshold=35
RmedianKernel=5
RextraCut=9
clipLimit=2.0
tileGridSize=(8, 8)
RframeWidth=25
LblackHatKernel=133
LmaxCorners=1840
LCradius=6
Lthreshold=35
LmedianKernel=1
LextraCut=14
LframeWidth=20
# Nmask = (cv2.imread(Nmask_path, 0) / 255).astype('uint8')


@guvectorize(['uint8[:,:](uint8[:,:], uint8, float64, uint8, uint8)'], '(n,n),(),(),(),()->(n,n)', target='cuda', nopython=True)
def get_cmask(img, maxCorners=1900, qualityLevel=0.001, minDistance=1, Cradius=6):
    corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
    corners = np.int0(corners)
    cmask = np.zeros(img.shape)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(cmask, (x, y), Cradius, 1, -1)
    return cmask

@guvectorize(['uint8[:,:](uint8[:,:])'], '(n,n)->(n,n)', target='cuda')
def contourMask(image):
    im2, contours, hierc = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = np.zeros(len(contours))
    for j in range(len(contours)):
        cnt = contours[j]
        area[j] = cv2.contourArea(cnt)
    mask = np.zeros(image.shape)
    cv2.drawContours(mask, contours, np.argmax(area), (255), -1)  # draw largest contour
    contours.clear()
    return mask

@guvectorize(['uint8[:,:](uint8[:,:], uint8, uint8)'], '(n,n),(),()->(n,n)', target='cuda')
def eraseMax(img, eraseLineCenter=0, eraseLineWidth=30):
    sumpix0 = np.sum(img, 0)
    max_r2 = np.int_(len(sumpix0) / 3) + np.argmax(sumpix0[np.int_(len(sumpix0) / 3):np.int_(len(sumpix0) * 2 / 3)])
    cv2.line(img, (max_r2 + eraseLineCenter, 0), (max_r2 + eraseLineCenter, 512), 0, eraseLineWidth)
    return img


@guvectorize(['uint8[:,:](uint8[:,:], uint8)'], '(n,n),()->(n,n)', target='cuda')
def eraseLeft(img, extraCut=0):
    sumpix0 = np.sum(img, 0)
    max_r2 = np.int_(len(sumpix0) / 3) + np.argmax(sumpix0[np.int_(len(sumpix0) / 3):np.int_(len(sumpix0) * 2 / 3)])
    img[:, max_r2 - extraCut:511] = 0
    return img


@guvectorize(['uint8[:,:](uint8[:,:], uint8)'], '(n,n),()->(n,n)', target='cuda')
def eraseRight(img, extraCut=0):
    sumpix0 = np.sum(img, 0)
    max_r2 = np.int_(len(sumpix0) / 3) + np.argmax(sumpix0[np.int_(len(sumpix0) / 3):np.int_(len(sumpix0) * 2 / 3)])
    img[:, 0:max_r2 + extraCut] = 0
    return img


@guvectorize(['uint8[:,:](uint8[:,:], uint8)'], '(n,n),()->(n,n)', target='cuda')
def eraseFrame(img, width=1):
    img[0:width, :] = 0
    img[511 - width:511, :] = 0
    img[:, 0:width] = 0
    img[:, 511 - width:511] = 0
    return img


@guvectorize(['uint8[:,:](uint8[:,:], uint8[:,:])'], '(n,n),(n,n)->(n,n)', target='cuda')
def segLeft(img_cluster, Nmask): #, blackHatKernel=169, threshold=45, medianKernel=23, maxCorners=1900, Cradius=6,clipLimit=2.0, tileGridSize=(8, 8), extraCut=0, frameWidth=1):
    img = np.copy(img_cluster)
    # rows, cols = img.shape
    img = eraseRight(img, LextraCut)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img = clahe.apply(img)
    ker = LblackHatKernel
    kernel = np.ones((ker, ker), np.uint8)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh = cv2.threshold(blackhat, Lthreshold, 255, 0)
    cmask = get_cmask(img, LmaxCorners, Cradius=LCradius)
    mask = np.multiply(cmask, thresh).astype('uint8')
    median = cv2.medianBlur(mask, LmedianKernel)
    contour_mask = contourMask(median)
    frame = eraseFrame(contour_mask, LframeWidth) * Nmask
    return frame.astype('uint8')


@guvectorize(['uint8[:,:](uint8[:,:], uint8[:,:])'], '(n,n),(n,n)->(n,n)', target='cuda')
def segRight(img_cluster, Nmask): #, blackHatKernel=169, threshold=45, medianKernel=23, maxCorners=3800, Cradius=6, clipLimit=2.0, tileGridSize=(8, 8), extraCut=0, frameWidth=28):
    img = np.copy(img_cluster)
    # rows, cols = img.shape
    img = eraseLeft(img, RextraCut)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img = clahe.apply(img)
    ker = RblackHatKernel
    kernel = np.ones((ker, ker), np.uint8)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh = cv2.threshold(blackhat, Rthreshold, 255, 0)
    cmask = get_cmask(img, RmaxCorners, Cradius=RCradius)
    mask = np.multiply(cmask, thresh).astype('uint8')
    median = cv2.medianBlur(mask, RmedianKernel)
    contour_mask = contourMask(median)
    frame = eraseFrame(contour_mask, RframeWidth) * Nmask
    return frame.astype('uint8')


@guvectorize(['uint8[:,:](uint8[:,:])'], '(n,n)->(n,n)',target='cuda')
def clusterSeg(im1):
    # im1 = cv2.imread(filename, 0)
    im = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)  # gray to rgb
    im_lab = color.rgb2lab(im)  # rgb to lab
    data = np.array([im_lab[..., 1].ravel(), im_lab[..., 2].ravel()])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data.T)  # kmeans to our image
    segmentation = kmeans.labels_.reshape(im.shape[:-1])  # fit kmeans to our data
    color_mean = color.label2rgb(segmentation, im, kind='gray')
    lower_gray = np.array([110, 110, 110])  # range of gray in the lungs
    upper_gray = np.array([190, 190, 190])

    dst = cv2.inRange(color_mean, lower_gray, upper_gray)
    for i in range(len(im)):  # for every col draw pixels in lung
        for j in range(len(im[1])):  # for every row pixels in lung
            if dst[i, j] > 0:
                color_mean[i, j] = 0
                if dst[i, j] <= 0:
                    color_mean[i, j] = 255

    return (0.975 * im1 + 0.025 * color_mean[:, :, 0]).astype(
        'uint8')  # optimal combination betweem original and mask


def segfinal(input_dir, output_dir):
    time_start = time.clock()
    for filename in glob.glob(os.path.join(input_dir, '*.png')):
        basename = os.path.basename(filename)

        # Nmask = (cv2.imread(Nmask_path, 0) / 255).astype('uint8')
        #
        # cluster_img = clusterSeg(filename)
        #
        # Rmask = segRight(cluster_img, Nmask, RblackHatKernel, Rthreshold, RmedianKernel, RmaxCorners, RCradius,
        #                  clipLimit,
        #                  tileGridSize, RextraCut, RframeWidth)
        # Lmask = segLeft(cluster_img, Nmask, LblackHatKernel, Lthreshold, LmedianKernel, LmaxCorners, LCradius,
        #                 clipLimit,
        #                 tileGridSize, LextraCut, LframeWidth)
        #
        # mask = ((Rmask + Lmask) / 255).astype('uint8')
        # img = cv2.imread(filename, 0)
        # img_lungs = img * mask

        im = cv2.imread(filename, 0)
        img_lungs = seg_image(im, Nmask)

        # saving the images
        outpath = output_dir + basename
        scipy.misc.imsave(outpath, img_lungs)
    print('Program runtime:', '%.2f' % (time.clock() - time_start), 'seconds')


@guvectorize(['uint8[:,:](uint8[:,:], uint8[:,:])'], '(n,n),(n,n)->(n,n)', target='cuda')
def seg_image(image, Nmask):
    # Nmask = (cv2.imread(Nmask_path, 0) / 255).astype('uint8')
    cluster_img = clusterSeg(image)
    Rmask = segRight(cluster_img, Nmask)#, RblackHatKernel, Rthreshold, RmedianKernel, RmaxCorners, RCradius, clipLimit,tileGridSize, RextraCut, RframeWidth)
    Lmask = segLeft(cluster_img, Nmask) #, LblackHatKernel, Lthreshold, LmedianKernel, LmaxCorners, LCradius, clipLimit, tileGridSize, LextraCut, LframeWidth)
    mask = ((Rmask + Lmask) / 255).astype('uint8')
    img = cv2.imread(filename, 0)
    img_lungs = img * mask
    return img_lungs
