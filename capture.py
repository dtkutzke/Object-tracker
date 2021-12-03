#!/usr/bin/python
''' Constructs a videocapture device on either webcam or a disk movie file.
Press q to exit

Junaed Sattar
October 2021
'''
from __future__ import division
import numpy as np
import cv2
print( "** CV2 version **  ", cv2.__version__)
import sys
from matplotlib import pyplot as plt

'''global data common to all vision algorithms'''
isTracking = False
r = g = b = 0.0
image = np.zeros((640, 480, 3), np.uint8)
trackedImage = np.zeros((640, 480, 3), np.uint8)
imageHeight, imageWidth, planes = image.shape

'''(Demetri) Global variables for mean shift'''
regionWidth = 30
rW = int(regionWidth / 2)
regionHeight = 20
rH = int(regionHeight / 2)
histBinWidth = 256
xLast = yLast = 0
eps = 0.10
MAX_ITER = 5
NEIGHBORHOOD_SIZE = 20

# One histogram for every RGB value
hisFeature = np.zeros([histBinWidth, histBinWidth, histBinWidth])


'''Create a region of interest ROI around an x,y point'''
def getRoi(x, y):
    global imageHeight, imageWidth, rH, rW, image
    if (rW <= x < imageWidth - rW) and (rH <= y < imageHeight - rH):
        return image[y - rH:y + rH, x - rW:x + rW]
    else:
        return None



'''Hellinger has the advantage of a mapping from reals to [0,1]'''
def calcHellinger(p1, p2):
    BC = cv2.compareHist(p1, p2, cv2.HISTCMP_BHATTACHARYYA)
    if BC is not None:
        print("Hellinger", np.sqrt(1 - BC))
        return np.sqrt(1 - BC)
    else:
        return None


def convolveWithKernel(roi_in):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    #fig, (ax1, ax2) = plt.subplots(2)
    #ax1.imshow(cv2.cvtColor(roi_in, cv2.COLOR_BGR2RGB))
    #ax1.set_title("Non-convolved")
    #ax2.imshow(cv2.cvtColor(cv2.filter2D(roi_in, -1, kernel), cv2.COLOR_BGR2RGB))
    #ax2.set_title("Convolved")
    return cv2.filter2D(roi_in, -1, kernel)

'''Defines a color model for the target of interest.
   Now, just reading pixel color at location
'''
def TuneTracker(x, y):

    global r, g, b, image, trackedImage, hisFeature, xLast, yLast

    xLast = x
    yLast = y

    # Bounding box defined by preset size
    roi = getRoi(x, y)

    # Convolve with a kernel
    roi = convolveWithKernel(roi)

    # Compute and normalize the histogram
    hisFeature = cv2.calcHist([roi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth], [0, 256, 0, 256, 0, 256])
    #hisFeature /= hisFeature.sum()


def mapHistToRoi(roi_in, hist):
    roiheight, roiwidth, implanes = roi_in.shape

    # Create a pdf in the new region of interest
    pdf = np.zeros([roiheight, roiwidth])
    # Now take the last histogram and compute the pdf over of the new region of interest
    # roi_in[1, 1, 1] = some value for green
    for j in range(roiwidth):
        for i in range(roiheight):
            b, g, r = roi_in[i, j]
            pdf[i, j] = hist[b, g, r]

    # return pdf
    if pdf.sum() != 0:
        return pdf / pdf.sum()
    else:
        return None


'''Essentially, compute the center of mass of a given pdf and return (x,y)'''
def computeCenterOfMass(pdf_in):
    thresh = 0.0000001
    height, width = pdf_in.shape

    # Compute the mean of the flattened array
    m = np.max(pdf_in)

    pdf_flat = pdf_in.flatten('C')
    pdf_flat = abs(pdf_flat - m)
    pdf_flat = pdf_flat - thresh
    idx = np.argmin(pdf_flat)
    (y, x) = np.unravel_index(idx, (height, width), 'C')
    return x, y

'''Generate a new target candidate location'''
def generateNewTestPoint(x_last, y_last, max_dist):
    global imageHeight, imageWidth, rH, rW, image
    searchSize = np.max([2*rW,2*rH])
    x_new = np.random.randint(x_last-searchSize, x_last+searchSize)
    y_new = np.random.randint(y_last-searchSize, y_last+searchSize)
    #while np.linalg.norm(np.array((x_new, y_new)) - np.array((x_last, y_last))) > max_dist:
    #    x_new = np.random.randint(x_last - searchSize, x_last + searchSize)
    #    y_new = np.random.randint(y_last - searchSize, y_last + searchSize)

    return x_new, y_new

''' Have to update this to perform Sequential Monte Carlo
    tracking, i.e. the particle filter steps.

    Currently this is doing naive color thresholding.
'''
def doTracking():
    global isTracking, image, r, g, b, trackedImage, hisFeature, xLast, yLast, pdfFeature
    if isTracking:
        print(image.shape)

        # Compute the roi
        newRoi = getRoi(xLast, yLast)


        validRoiUpdate = False
        if newRoi is not None:
            validRoiUpdate = True

        if validRoiUpdate:
            newRoi = convolveWithKernel(newRoi)
            # Compute the pdf from the histogram and region of interest
            hisNew = cv2.calcHist([newRoi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth], [0, 256, 0, 256, 0, 256])
            #hisNew /= hisNew.sum()

            dist = calcHellinger(hisNew, hisFeature)
            mostProbableX, mostProbableY = generateNewTestPoint(xLast, yLast, NEIGHBORHOOD_SIZE)

            it = 0
            while dist > eps and it < MAX_ITER:
                xTest, yTest = mostProbableX, mostProbableY

                # Compute the new region of interest over the global coordinates
                newRoi = getRoi(xTest, yTest)


                validRoiUpdate = False
                if newRoi is not None:
                    validRoiUpdate = True

                    # Update xLast and yLast to reflect the new global mean coordinates
                    xLast, yLast = xTest, yTest

                if validRoiUpdate:
                    newRoi = convolveWithKernel(newRoi)
                    # Compute the pdf from the histogram and region of interest
                    hisNew = cv2.calcHist([newRoi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth],
                                          [0, 256, 0, 256, 0, 256])
                    #if hisNew.sum() != 0:
                    #    hisNew /= hisNew.sum()
                    #else:
                    #    print(" * ERROR * Problem in normalization")

                    dist = calcHellinger(hisNew, hisFeature)
                    print("Iteration count = ", it)
                    it += 1

        print("New location", xLast, yLast)

        cv2.rectangle(image, (xLast - rW, yLast - rH), (xLast + rW, yLast + rH), (255, 0, 0), 2)


def clickHandler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print('left button released')
        TuneTracker(x, y)


def mapClicksRoiToGlobal(x, y, bottom_x, bottom_y):
    return x + bottom_x, y + bottom_y


def mapClicks(x, y, curWidth, curHeight):
    global imageHeight, imageWidth
    imageX = x * imageWidth / curWidth
    imageY = y * imageHeight / curHeight
    return int(imageX), int(imageY)


def captureVideo(src):
    global image, isTracking, trackedImage
    cap = cv2.VideoCapture(src)
    if cap.isOpened() and src == '0':
        ret = cap.set(3, 640) and cap.set(4, 480)
        if ret == False:
            print('Cannot set frame properties, returning')
            return
    else:
        frate = cap.get(cv2.CAP_PROP_FPS)
        print(frate, ' is the framerate')
        waitTime = int(1000 / frate)

    #    waitTime = time/frame. Adjust accordingly.
    if src == 0:
        waitTime = 1
    if cap:
        print('Succesfully set up capture device')
    else:
        print('Failed to setup capture device')

    windowName = 'Input View, press q to quit'
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, clickHandler)
    while (True):
        # Capture frame-by-frame
        ret, image = cap.read()
        if ret == False:
            break

        # Display the resulting frame
        if isTracking:
            doTracking()
        cv2.imshow(windowName, image)
        inputKey = cv2.waitKey(waitTime) & 0xFF
        if inputKey == ord('q'):
            break
        elif inputKey == ord('t'):
            isTracking = not isTracking

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


print('Starting program')
if __name__ == '__main__':
    arglist = sys.argv
    src = 0
    print('Argument count is ', len(arglist))
    if len(arglist) == 2:
        src = arglist[1]
    else:
        src = 0
    captureVideo(src)
else:
    print('Not in main')
