#!/usr/bin/python
''' Constructs a videocapture device on either webcam or a disk movie file.
Press q to exit

Junaed Sattar
October 2021
'''
from __future__ import division
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt


'''global data common to all vision algorithms'''
isTracking = False
r = g = b = 0.0
image = np.zeros((640,480,3), np.uint8)
trackedImage = np.zeros((640,480,3), np.uint8)
imageHeight, imageWidth, planes = image.shape

'''(Demetri) Global variables for mean shift'''
regionWidth = 30
rW = int(regionWidth/2)
regionHeight = 20
rH = int(regionHeight/2)
histBinWidth = 256
xLast = yLast = 0
eps = 0.5

# One histogram for every RGB value
hisFeature = np.zeros([histBinWidth, 3])
#his = np.zeros([histBinWidth])
pdfFeature = np.zeros([regionHeight, regionWidth])

# Borrowed from
# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
# K(x,y) = exp(-(1/2)*(x^2 + y^2)/sigma^2)
def gaussianKernel(sig=1.0):
    global image
    imheight, imwidth, implanes = image.shape
    center_y = int(imheight/2)
    center_x = int(imwidth/2)
    kernel = np.zeros((imheight, imwidth))
    for i in range(imheight):
        for j in range(imwidth):
            diff = (i - center_y) ** 2 + (j - center_x) ** 2
            kernel[i, j] = np.exp(-diff / (2 * sig ** 2))

    return kernel / np.sum(kernel)

'''Create a region of interest ROI around an x,y point'''
def getRoi(x,y):
    global imageHeight, imageWidth, rH, rW
    if (imageWidth > x >= 0) and (imageHeight > y >= 0):
        return image[y - rH:y + rH, x - rW:x + rW]


'''Compute the Bhattacharyya coefficient for two histograms'''
def calcBhattacharyyaCoeff(p1, p2):
    if p1.shape == p2.shape:
        height, width = p1.shape
        BC = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                BC[i, j] = np.sqrt(p1[i, j]*p2[i, j])

        return BC.sum()
    else:
        return None


'''Compute the Bhattacharyya distance between two histograms'''
def calcBhattacharyya(p1, p2):
    BC = calcBhattacharyyaCoeff(p1, p2)
    if BC is not None:
        return -np.log(BC)
    else:
        return None

'''Hellinger has the advantage of a mapping from reals to [0,1]'''
def calcHellinger(p1, p2):
    BC = calcBhattacharyyaCoeff(p1, p2)
    if BC is not None:
        print("Hellinger", np.sqrt(1 - BC))
        return np.sqrt(1-BC)
    else:
        return None

def convolveWithKernel():
    kernel = gaussianKernel(1)
    return np.matmul(kernel, image)

'''Defines a color model for the target of interest.
   Now, just reading pixel color at location
'''
def TuneTracker(x,y):
    global r,g,b, image, trackedImage, hisFeature, pdfFeature, xLast, yLast

    xLast = x
    yLast = y

    # Bounding box defined by preset size
    roi = getRoi(x, y)

    # Normalize the image between 0 and 1
    #normImage = cv2.normalize(roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Calculate the histogram
    for i in range(3):
        hisFeature[:, i] = cv2.calcHist([roi], [i], None, [histBinWidth], [0, 256]).reshape((256,))
        hisFeature[:, i] /= hisFeature[:, i].sum()
        #print("Sum of feature histogram ", hisFeature[:, i].sum() )

    pdfFeature = mapHistToRoi(roi, hisFeature)

    #plt.plot(his[0])
    #plt.show()
    #b,g,r = image[y,x]
    #sumpixels = float(b)+float(g)+float(r)
    #if sumpixels != 0:
    #    b = b/sumpixels,
    #    r = r/sumpixels
    #    g = g/sumpixels
    #print( r,g,b, 'at location ', x,y )


def mapHistToRoi(roi_in, hist):
    roiheight, roiwidth, implanes = roi_in.shape

    # Create a pdf in the new region of interest
    pdf = np.zeros([roiheight, roiwidth])
    # Now take the last histogram and compute the pdf over of the new region of interest
    for j in range(roiwidth):
        for i in range(roiheight):
            var = 1
            for k in range(implanes):
                var *= hist[roi_in[i, j, k], k]

            pdf[i, j] = var

    #plt.imshow(pdf)
    #plt.show()
    print("Map hist to ROI new PDF sum",  pdf.sum())
    #return pdf
    return pdf / pdf.sum()
    #return pdf / np.amax(pdf)


def plotHistogram():
    pass


'''Essentially, compute the center of mass of a given pdf and return (x,y)'''
def computeCenterOfMass(pdf_in):
    thresh = 0.001
    height, width = pdf_in.shape

    # Compute the mean of the flattened array
    m = np.max(pdf_in)
    #m = np.mean(pdf_in)
    # Total mass
    #M = pdf_in.sum()

    # Now compute the x and y locations that are closest
    for i in range(height):
        for j in range(width):
            if abs(pdf_in[i, j] - m) < thresh:
                return j, i

    #xcm = 0
    #for i in range(height):
    #    xcm += i*pdf_in[i, ]


''' Have to update this to perform Sequential Monte Carlo
    tracking, i.e. the particle filter steps.

    Currently this is doing naive color thresholding.
'''
def doTracking():
    global isTracking, image, r,g,b, trackedImage, hisFeature, xLast, yLast, pdfFeature
    if isTracking:
        print( image.shape )
        imheight, imwidth, implanes = image.shape

        # Compute the roi
        newRoi = getRoi(xLast, yLast)

        # Compute the pdf from the histogram and region of interest
        hisNew = np.zeros([histBinWidth, 3])
        for i in range(3):
            hisNew[:, i] = cv2.calcHist([newRoi], [i], None, [histBinWidth], [0, 256]).reshape((256,))
            hisNew[:, i] /= hisNew[:, i].sum()

        pdfNew = mapHistToRoi(newRoi, hisNew)

#        dist = calcHistBhattacharyya(pdfNew, pdfFeature)
        dist = calcHellinger(pdfNew, pdfFeature)

        while dist > eps:
            xMean, yMean = computeCenterOfMass(pdfNew)
            xMean, yMean = mapClicksRoiToGlobal(xMean, yMean, xLast+rH, yLast-rW)
            xLast, yLast = xMean, yMean

            newRoi = getRoi(xLast, yLast)

            # Compute the pdf from the histogram and region of interest
            hisNew = np.zeros([histBinWidth, 3])
            for i in range(3):
                hisNew[:, i] = cv2.calcHist([newRoi], [i], None, [histBinWidth], [0, 256]).reshape((256,))
                hisNew[:, i] /= hisNew[:, i].sum()

            pdfNew = mapHistToRoi(newRoi, hisNew)

            dist = calcHellinger(pdfNew, pdfFeature)


        print("New location", xLast, yLast)
        #xMean = 300
        #yMean = 300

        cv2.rectangle(image, (xLast-rW, yLast-rH), (xLast + rW, yLast + rH), (255, 0, 0), 2)

#        xLast = xMean
#        yLast = yMean

        #for j in range( imwidth ):
        #    for i in range( imheight ):
        #        bb, gg, rr = image[i,j]
        #        sumpixels = float(bb)+float(gg)+float(rr)
        #        if sumpixels == 0:
        #            sumpixels = 1
        #        if rr/sumpixels >= r and gg/sumpixels >= g and bb/sumpixels >= b:
        #            image[i,j] = [255,255,255];
        #        else:
        #            image[i,j] = [0,0,0];


def clickHandler( event, x,y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print( 'left button released' )
        TuneTracker( x, y )


def mapClicksRoiToGlobal(x, y, bottom_x, bottom_y):
    return x + bottom_x, y + bottom_y

def mapClicks( x, y, curWidth, curHeight ):
    global imageHeight, imageWidth
    imageX = x*imageWidth/curWidth
    imageY = y*imageHeight/curHeight
    return int(imageX), int(imageY)

def captureVideo(src):
    global image, isTracking, trackedImage
    cap = cv2.VideoCapture(src)
    if cap.isOpened() and src=='0':
        ret = cap.set(3,640) and cap.set(4,480)
        if ret==False:
            print( 'Cannot set frame properties, returning' )
            return
    else:
        frate = cap.get(cv2.CAP_PROP_FPS)
        print( frate, ' is the framerate' )
        waitTime = int( 1000/frate )

#    waitTime = time/frame. Adjust accordingly.
    if src == 0:
        waitTime = 1
    if cap:
        print( 'Succesfully set up capture device' )
    else:
        print( 'Failed to setup capture device' )

    windowName = 'Input View, press q to quit'
    cv2.namedWindow(windowName)
    cv2.setMouseCallback( windowName, clickHandler )
    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()
        if ret==False:
            break

        # Display the resulting frame
        if isTracking:
            doTracking()
        cv2.imshow(windowName, image )
        inputKey = cv2.waitKey(waitTime) & 0xFF
        if inputKey == ord('q'):
            break
        elif inputKey == ord('t'):
            isTracking = not isTracking

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


print( 'Starting program' )
if __name__ == '__main__':
    arglist = sys.argv
    src = 0
    print( 'Argument count is ', len(arglist) )
    if len(arglist) == 2:
        src = arglist[1]
    else:
        src = 0
    captureVideo(src)
else:
    print( 'Not in main' )

