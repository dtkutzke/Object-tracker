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
image = np.zeros((640, 480, 3), np.uint8)
trackedImage = np.zeros((640, 480, 3), np.uint8)
imageHeight, imageWidth, planes = image.shape

'''(Demetri) Global variables for mean shift'''
regionWidth = 100
rW = int(regionWidth / 2)
regionHeight = 100
rH = int(regionHeight / 2)
histBinWidth = 256
xLast = yLast = 0
eps = 0.30
MAX_ITER = 2

# One histogram for every RGB value
hisFeature = np.zeros([histBinWidth, 3])
#hisFeature = np.zeros([histBinWidth, histBinWidth, histBinWidth])
# his = np.zeros([histBinWidth])
pdfFeature = np.zeros([regionHeight, regionWidth])


# Borrowed from
# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
# K(x,y) = exp(-(1/2)*(x^2 + y^2)/sigma^2)
def gaussianKernel(roi_in, sig=1.0):
    imheight, imwidth = roi_in.shape
    center_y = int(imheight / 2)
    center_x = int(imwidth / 2)
    kernel = np.zeros((imheight, imwidth))
    for i in range(imheight):
        for j in range(imwidth):
            diff = (i - center_y) ** 2 + (j - center_x) ** 2
            kernel[i, j] = np.exp(-diff / (2 * sig ** 2))

    return kernel / np.sum(kernel)


'''Create a region of interest ROI around an x,y point'''
def getRoi(x, y):
    global imageHeight, imageWidth, rH, rW, image
    if (rW <= x < imageWidth - rW) and (rH <= y < imageHeight - rH):
        return image[y - rH:y + rH, x - rW:x + rW]
    else:
        return None


'''Compute the Bhattacharyya coefficient for two histograms'''
def calcBhattacharyyaCoeff(p1, p2):
    if p1.shape == p2.shape:
        bins, channels = p1.shape
        BC = np.zeros([channels])
        for c in range(channels):
            BC[c] = np.dot(np.sqrt(p1[:,c]),np.sqrt(p2[:,c]))
            if BC[c] > 1:
                print("BC coefficient bigger than one, somehow!")


        #height, width, planes = p1.shape
        #BC = np.zeros([height, width, planes])
        #BC = 0
        #for i in range(height):
        #    for j in range(width):
        #        for k in range(planes):
                    #BC[i, j] = np.sqrt(p1[i, j] * p2[i, j])
                    #BC[i, j, k] = np.sqrt(np.dot(p1[i, j, k], p2[i, j, k]))
        #            BC += np.sqrt(p1[i, j, k] * p2[i, j, k])

        #if BC > 1:
        #    print("BC coefficient bigger than one, somehow!")

        return BC
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
        return np.sqrt(1 - BC)
    else:
        return None


#def convolveWithKernel(roi_in):
#    kernel = gaussianKernel(roi_in)
#    return convolve2D(roi_in, kernel)

# https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f3810i
# def convolve2D(image, kernel, padding=0, strides=1):
#     # Cross Correlation
#     kernel = np.flipud(np.fliplr(kernel))
#
#     # Gather Shapes of Kernel + Image + Padding
#     xKernShape = kernel.shape[0]
#     yKernShape = kernel.shape[1]
#     xImgShape = image.shape[0]
#     yImgShape = image.shape[1]
#
#     # Shape of Output Convolution
#     xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
#     yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
#     output = np.zeros((xOutput, yOutput))
#
#     # Apply Equal Padding to All Sides
#     if padding != 0:
#         imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
#         imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
#         print(imagePadded)
#     else:
#         imagePadded = image
#
#     # Iterate through image
#     for y in range(image.shape[1]):
#         # Exit Convolution
#         if y > image.shape[1] - yKernShape:
#             break
#         # Only Convolve if y has gone down by the specified Strides
#         if y % strides == 0:
#             for x in range(image.shape[0]):
#                 # Go to next row once kernel is out of bounds
#                 if x > image.shape[0] - xKernShape:
#                     break
#                 try:
#                     # Only Convolve if x has moved by the specified Strides
#                     if x % strides == 0:
#                         output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
#                 except:
#                     break
#
#     return output


'''Defines a color model for the target of interest.
   Now, just reading pixel color at location
'''
def TuneTracker(x, y):
    global r, g, b, image, trackedImage, hisFeature, pdfFeature, xLast, yLast

    xLast = x
    yLast = y

    # Bounding box defined by preset size
    roi = getRoi(x, y)

    # Normalize the image between 0 and 1
    # normImage = cv2.normalize(roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Calculate the histogram
    for i in range(3):
        hisFeature[:, i] = cv2.calcHist([roi], [i], None, [histBinWidth], [0, 256]).reshape((256,))
        hisFeature[:, i] /= hisFeature[:, i].sum()
        # print("Sum of feature histogram ", hisFeature[:, i].sum() )
    #hisFeature = cv2.calcHist([roi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth], [0, 256, 0, 256, 0, 256])
    #hisFeature /= hisFeature.sum()

    #pdfFeature = mapHistToRoi(roi, hisFeature)

    # plt.plot(his[0])
    # plt.show()
    # b,g,r = image[y,x]
    # sumpixels = float(b)+float(g)+float(r)
    # if sumpixels != 0:
    #    b = b/sumpixels,
    #    r = r/sumpixels
    #    g = g/sumpixels
    # print( r,g,b, 'at location ', x,y )


def mapHistToRoi(roi_in, hist):
    roiheight, roiwidth, implanes = roi_in.shape

    # Create a pdf in the new region of interest
    pdf = np.zeros([roiheight, roiwidth])
    # Now take the last histogram and compute the pdf over of the new region of interest
    # roi_in[1, 1, 1] = some value for green
    for j in range(roiwidth):
        for i in range(roiheight):
            var = 1
            for c in range(implanes):
            #b, g, r = roi_in[i, j]
                #var = hist[b,g,r]
                var *= hist[roi_in[i,j,c],c]

            pdf[i, j] = var

    # plt.imshow(pdf)
    # plt.show()
    print("Map hist to ROI new PDF sum", pdf.sum())
    # return pdf
    if pdf.sum() != 0:
        return pdf / pdf.sum()
    else:
        return None
    # return pdf / np.amax(pdf)


def plotHistogram():
    pass


'''Essentially, compute the center of mass of a given pdf and return (x,y)'''


def computeCenterOfMass(pdf_in):
    thresh = 0.001
    height, width = pdf_in.shape

    # Compute the mean of the flattened array
    m = np.max(pdf_in)
    # m = np.mean(pdf_in)
    # Total mass
    # M = pdf_in.sum()

    # Now compute the x and y locations that are closest
    for y in range(height):
        for x in range(width):
            if abs(pdf_in[y, x] - m) < thresh:
                return x, y

    # xcm = 0
    # for i in range(height):
    #    xcm += i*pdf_in[i, ]


def generateNewTestPoint(x_last, y_last, max_dist):
    global imageHeight, imageWidth, rH, rW, image
    x_new = np.random.randint(rW, imageWidth - rW)
    y_new = np.random.randint(rH, imageHeight - rH)
    while np.linalg.norm(np.array((x_new, y_new)) - np.array((x_last, y_last))) > max_dist:
        x_new = np.random.randint(rW, imageWidth - rW)
        y_new = np.random.randint(rH, imageHeight - rH)

    return x_new, y_new

''' Have to update this to perform Sequential Monte Carlo
    tracking, i.e. the particle filter steps.

    Currently this is doing naive color thresholding.
'''
def doTracking():
    global isTracking, image, r, g, b, trackedImage, hisFeature, xLast, yLast, pdfFeature
    if isTracking:
        print(image.shape)
        imheight, imwidth, implanes = image.shape

        # Compute the roi
        newRoi = getRoi(xLast, yLast)
        validRoiUpdate = False
        if newRoi is not None:
            validRoiUpdate = True

        if validRoiUpdate:
            # Compute the pdf from the histogram and region of interest
            hisNew = np.zeros([histBinWidth, 3])
            for i in range(3):
                hisNew[:, i] = cv2.calcHist([newRoi], [i], None, [histBinWidth], [0, 256]).reshape((256,))
                hisNew[:, i] /= hisNew[:, i].sum()
            #hisNew = cv2.calcHist([newRoi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth], [0, 256, 0, 256, 0, 256])
            #hisNew /= hisNew.sum()
            #pdfNew = mapHistToRoi(newRoi, hisNew)

            #        dist = calcHistBhattacharyya(pdfNew, pdfFeature)
            #dist = calcHellinger(pdfNew, pdfFeature)

            dist = calcHellinger(hisNew, hisFeature)



            it = 0
            while all(dist > eps) and it < MAX_ITER:

                pdf = mapHistToRoi(image, hisFeature)
                if pdf is not None:
                    mostProbableX, mostProbableY = computeCenterOfMass(pdf)
                else:
                    mostProbableX, mostProbableY = np.random.randint(rW, imageWidth - rW), np.random.randint(rH,imageHeight - rH)
                # Local to region of interest
    #            xMean, yMean = computeCenterOfMass(pdfNew)
                xTest, yTest = generateNewTestPoint(mostProbableX, mostProbableY, 20)
                #xTest, yTest = generateNewTestPoint(xLast, yLast)
                # Remap these values to the global image
                #xMean, yMean = mapClicksRoiToGlobal(xMean, yMean, xLast - rW, yLast + rH)



                # Compute the new region of interest over the global coordinates
                newRoi = getRoi(xTest, yTest)

                validRoiUpdate = False
                if newRoi is not None:
                    validRoiUpdate = True
                    # Update xLast and yLast to reflect the new global mean coordinates
                    xLast, yLast = xTest, yTest

                if validRoiUpdate:
                    # Compute the pdf from the histogram and region of interest
                    for i in range(3):
                        hisNew[:, i] = cv2.calcHist([newRoi], [i], None, [histBinWidth], [0, 256]).reshape((256,))
                        if hisNew[:, i].sum() != 0:
                            hisNew[:, i] /= hisNew[:, i].sum()
                        else:
                            print(" * ERROR * Problem in normalization")

                #hisNew = cv2.calcHist([newRoi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth],
                #                          [0, 256, 0, 256, 0, 256])
                #if hisNew.sum() != 0:
                #    hisNew /= hisNew.sum()
                #else:
                #    print(" * ERROR * Problem in normalization")

                    #pdfNew = mapHistToRoi(newRoi, hisNew)

                    dist = calcHellinger(hisNew, hisFeature)
                    print("Iteration count = ", it)
                    it += 1

        print("New location", xLast, yLast)
        # xMean = 300
        # yMean = 300

        cv2.rectangle(image, (xLast - rW, yLast - rH), (xLast + rW, yLast + rH), (255, 0, 0), 2)


#        xLast = xMean
#        yLast = yMean

# for j in range( imwidth ):
#    for i in range( imheight ):
#        bb, gg, rr = image[i,j]
#        sumpixels = float(bb)+float(gg)+float(rr)
#        if sumpixels == 0:
#            sumpixels = 1
#        if rr/sumpixels >= r and gg/sumpixels >= g and bb/sumpixels >= b:
#            image[i,j] = [255,255,255];
#        else:
#            image[i,j] = [0,0,0];


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
