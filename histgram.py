import os
import numpy as np
import cv2 as cv



def arrayToHist(grayArray, nums):
    if (len(grayArray.shape) != 2):
        print("length error")
        return None
    n=len(np.nonzero(grayArray)[0])

    w, h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if (hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    #n = w * h
    for key in hist.keys():
        hist[key] = float(hist[key]) / n
    hist[0]=0
    return hist



def histMatch(grayArray, h_d):
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray, 256)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des




def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    imdir_match = "D:\code\pytorch-demo\\removal\\nsd.jpg"
    imdir = "D:\code\pytorch-demo\\removal\\sd.jpg"
    out_file = "D:\code\pytorch-demo\\removal\\"
    mkdir(out_file)

    data1 = cv.imread(imdir)
    data2 = cv.imread(imdir_match)
    # data1 = cv.resize(data1, dsize=(1000, 1000))
    com = np.zeros_like(data1)
    for i in range(3):
        f1 = data1[:, :, i]
        f2 = data2[:, :, i]
        hist_m = arrayToHist(f2, 256)
        im_f1 = histMatch(f1, hist_m)

        com[:, :, i] = im_f1

    out_full_path = os.path.join(out_file, '201612_match.tif')
    cv.imwrite(out_full_path, com)
