#!/usr/bin/python
''' Constructs a videocapture device on either webcam or a disk movie file.
Press q to exit

Junaed Sattar
October 2021
'''
from __future__ import division
import numpy as np
import cv2
import os.path
import os
print( "** CV2 version **  ", cv2.__version__)
import sys
from matplotlib import pyplot as plt
from datetime import datetime

'''global data common to all vision algorithms'''
isTracking = False
r = g = b = 0
image = np.zeros((640, 480, 3), np.uint8)
trackedImage = np.zeros((640, 480, 3), np.uint8)
imageHeight, imageWidth, planes = image.shape

'''(Demetri) Global variables for mean shift
Note that these can all be changed'''
# Where to place any output data
dataDir = './Data'
# Prefix common to all output data
dataString = 'tracker_run'
# Should we save the output data?
saveData = True
# Need the frame rate for computing when to output data
frameRate = 0
# How many seconds we want to save data (e.g., every 5 s)
outputF = 5
# Width of the ROI
regionWidth = 60
# Height of the ROI
regionHeight = 60
# The width of a histogram bin
histBinWidth = 20
# What's the minimum value of the histogram
histMin = 0
# What's the maximum value of the histogram
histMax = 1
xLast = yLast = 0
# Threshold for convergence of the mean-shift
eps = 0.30
# Termination criteria for mean shift
maxItr = 10

'''These variables should NOT be changed'''
rW = int(regionWidth / 2)
rH = int(regionHeight / 2)
# Definition of neighborhood size
nbrSize = np.min([rW, rH])
# Just counts how many frames we've iterated through
frameCount = 0
roiFeature = np.zeros([regionHeight, regionWidth, 3])
roiConvolved = np.zeros([regionHeight, regionWidth, 3])
# One histogram for every RGB value
hisFeature = np.zeros([histBinWidth, histBinWidth, histBinWidth])
# Histogram array output arrays
hisFeatureList = []
hisFileLabelList = []
hellingerDistList = []
candidateRoi = []


'''Create a region of interest ROI around an x,y point'''
def GetRoi(x, y):
    global imageHeight, imageWidth, rH, rW, image
    if (rW <= x < imageWidth - rW) and (rH <= y < imageHeight - rH):
        roi = image[y - rH:y + rH, x - rW:x + rW]
        return roi
    else:
        return None


'''Normalize to [0, 1]'''
def Normalize(roi_in):
    return cv2.normalize(roi_in, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


'''Hellinger has the advantage of a mapping from reals to [0,1]
NOTE that opencv HISTCMP_BHATTACHARYYA does actually compute Hellinger'''
def ComputeHellinger(p1, p2):
    BC = cv2.compareHist(p1, p2, cv2.HISTCMP_BHATTACHARYYA)
    print("Hellinger: ", BC)
    return BC


'''Perform convolution with the kernel'''
def ConvolveWithKernel(roi_in):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    flipCode = 0
    anchorx = 0
    anchory = 0
    rows, cols = kernel.shape
    anchor = (cols - anchorx - 1, rows - anchory - 1)
    kernel = cv2.flip(kernel, flipCode)
    return cv2.filter2D(roi_in, -1, kernel, None, anchor)


'''Computes the mean of a 3D histogram'''
def GetMeanOfHistogram(h):
    nBBins, nGBins, nRBins = h.shape
    h[h == 0] = np.nan
    h_flat = h.flatten()
    mn = np.nanmean(h_flat)
    scaled = abs(h_flat - mn)
    scaled[np.isnan(scaled)] = 0
    idx = np.argmax(scaled)
    (bBin, gBin, rBin) = np.unravel_index(idx, (nBBins, nGBins, nRBins), 'C')
    return bBin, gBin, rBin



'''Defines a color model for the target of interest.
   Now, just reading pixel color at location
'''
def TuneTracker(x, y):
    global r, g, b, image, trackedImage, hisFeature, xLast, yLast, roiFeature, roiConvolved

    xLast = x
    yLast = y

    # Bounding box defined by preset size
    roi = GetRoi(x, y)
    roiFeature = np.copy(roi)
    roi = Normalize(roi)

    # Convolve with a kernel
    roi = ConvolveWithKernel(roi)
    roiConvolved = np.copy(roi)

    # Compute and normalize the histogram
    hisFeature = cv2.calcHist([roi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth], [histMin, histMax, histMin, histMax, histMin, histMax])

#    b, g, r = np.unravel_index(np.argmax(hisFeature.flatten()), (histBinWidth, histBinWidth, histBinWidth))
    # Create a new temporary variable
    b, g, r = GetMeanOfHistogram(np.copy(hisFeature))
    print("Mean pixel bin for ", histBinWidth, " bins (", b, g, r, ")")


'''Generate a new target candidate location'''
def GenerateNewTestPoint(x_last, y_last, max_dist):
    global imageHeight, imageWidth, rH, rW, image
    y_new = np.random.randint(rH, imageHeight - rH)
    x_new = np.random.randint(rW, imageWidth-rW)
    while np.linalg.norm(np.array((x_new, y_new)) - np.array((x_last, y_last))) > max_dist:
        #x_new = np.random.randint(x_last - searchSize, x_last + searchSize)
        y_new = np.random.randint(rH, imageHeight - rH)
        x_new = np.random.randint(rW, imageWidth - rW)
        #y_new = np.random.randint(y_last - searchSize, y_last + searchSize)

    return x_new, y_new

'''Save a small file containing the configuration of this run'''
def SaveConfigDetails():
    with open(dataDir+"/"+dataString+"_config.txt", 'a', encoding='utf-8') as f:
        f.write("*** CONFIGURATION FILE *** "+dataString+"\n")
        f.write("Run date/time: " + datetime.now().strftime("%m/%d/%y %H:%M:%S")+"\n")
        f.write("Roi width: " + str(regionWidth)+"\n")
        f.write("Roi height: " + str(regionHeight)+"\n")
        f.write("Histogram bin width: " + str(histBinWidth)+"\n")
        f.write("Histogram min/max: " + str(histMin)+","+str(histMax)+"\n")
        f.write("Tracker similarity threshold: " + str(eps)+"\n")
        f.write("Tracker maximum iterations to converge: " + str(maxItr)+"\n")
        f.write("Maximum neighborhood search size: " + str(nbrSize)+"\n")
        f.write("Mean pixel bin frequency index (bgr): (" +str(b)+","+str(g)+","+str(r)+")\n")
        f.write("Figure output boolean: " + str(saveData)+"\n")


'''Helper function to extract the iterator for this data run'''
def GetDataRunCount():
    if os.path.isdir(dataDir):
        # Load the files, extract iter from last file
        onlyfiles = [f for f in os.listdir(dataDir) if os.path.isfile(os.path.join(dataDir, f))]
        if onlyfiles:
            last_file = onlyfiles[-1]
            if last_file.find(dataString) == 0:
                spl = last_file.split(dataString, 1)[1]
                currIt = spl.split('_', 1)[0]
                if int(currIt) < 9:
                    return "0" + str(int(currIt) + 1)
                else:
                    return str(int(currIt) + 1)
        else:
            return "0" + str(1)
    else:
        print("Creating data directory to save results")
        os.mkdir(dataDir)
        return "0" + str(1)


'''Saves all the tracked histograms and plots and saves the region of interest'''
def PerformCleanup():
    # Plot and save all histograms
    for h, f, d in zip(hisFeatureList, hisFileLabelList, hellingerDistList):
        PlotAndSaveHistogram(hisFeature, h, f, d)

    # Plot the candidate RoIs
    for c, h, d in zip(candidateRoi, hisFileLabelList, hellingerDistList):
        f = plt.figure(3)
        plt.imshow(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), aspect='auto')
        plt.title("Hellinger: "+str(d))
        plt.savefig(dataDir + "/" + h + "_candidateRoI.png")
        plt.close(f)

    # Plot the original feature RoI
    tmp = cv2.cvtColor(roiFeature, cv2.COLOR_BGR2RGB)
    f = plt.figure(2)
    plt.imshow(tmp, aspect='auto')
    plt.savefig(dataDir + "/" + dataString + "_featureRoI.png")
    plt.close(f)

    # Plot the original feature RoI after convolution
    tmp = cv2.cvtColor(roiConvolved, cv2.COLOR_BGR2RGB)
    f = plt.figure(4)
    plt.imshow(tmp, aspect='auto')
    plt.savefig(dataDir + "/" + dataString + "_featureRoIConvolved.png")
    plt.close(f)

    # Save all configuration details for this run
    SaveConfigDetails()


'''Plots slices of a three dimensional histogram defined by the global b, g, r'''
def PlotAndSaveHistogram(h1, h2=None, fig_name=None, h_dist=None):
    global b, g, r
    fig = plt.figure(1, figsize=(10, 4))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    axList = [ax1, ax2, ax3]

    if fig_name is not None and h_dist is not None:
        fig.suptitle(fig_name+"\nHellinger: "+str(h_dist))

    xx = np.linspace(histMin, histMax, histBinWidth)
    xpos, ypos = np.meshgrid(xx, xx)
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    colors = ['b', 'g', 'r']
    sliceName = ["B fixed", "G fixed", "R fixed"]
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    #for i in range(3):
    ## ============= GR SLICE
    dz = h1[b,:,:].ravel()
    clrs1 = [colors[0]]*len(dz)
    ww = histMax/histBinWidth
    axList[0].bar3d(xpos, ypos, np.zeros_like(h1[b,:,:].ravel()), ww, ww, h1[b,:,:].ravel(), shade=False, color=clrs1, alpha=0.40, edgecolor='k')
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc=colors[0], alpha=0.40)
    if h2 is not None:
        axList[0].bar3d(xpos, ypos, np.zeros_like(h2[b,:,:].ravel()), ww, ww, h2[b,:,:].ravel(), shade=False, color=clrs1, edgecolor='k')
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc=colors[0])

    axList[0].set_title(sliceName[0])
    axList[0].set_xlim(histMin, histMax)
    axList[0].set_ylim(histMin, histMax)
    axList[0].set_xlabel("G bins")
    axList[0].set_ylabel("R bins")
    axList[0].set_zlabel("Frequency")
    axList[0].legend([proxy1,proxy2],['Feature','Candidate'])

    ## ============= BR SLICE
    dz = h1[:,g,:].ravel()
    clrs1 = [colors[1]]*len(dz)
    axList[1].bar3d(xpos, ypos, np.zeros_like(h1[:,g,:].ravel()), ww, ww, h1[:,g,:].ravel(), shade=False, color=clrs1, alpha=0.40, edgecolor='k')
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc=colors[1], alpha=0.40)
    if h2 is not None:
        axList[1].bar3d(xpos, ypos, np.zeros_like(h2[:,g,:].ravel()), ww, ww, h2[:,g,:].ravel(), shade=False, color=clrs1, edgecolor='k' )
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc=colors[1])

    axList[1].set_title(sliceName[1])
    axList[1].set_xlim(histMin, histMax)
    axList[1].set_ylim(histMin, histMax)
    axList[1].set_xlabel("B bins")
    axList[1].set_ylabel("R bins")
    axList[1].set_zlabel("Frequency")
    axList[1].legend([proxy1,proxy2],['Feature','Candidate'])

    ## ============= BG SLICE
    dz = h1[:,:,r].ravel()
    clrs1 = [colors[2]]*len(dz)
    axList[2].bar3d(xpos, ypos, np.zeros_like(h1[:,:,r].ravel()), ww, ww, h1[:,:,r].ravel(), shade=False, color=clrs1, alpha=0.40, edgecolor='k')
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc=colors[2], alpha=0.40)
    if h2 is not None:
        axList[2].bar3d(xpos, ypos, np.zeros_like(h2[:,:,r].ravel()), ww, ww, h2[:,:,r].ravel(), shade=False, color=clrs1, edgecolor='k')
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc=colors[2])

    axList[2].set_title(sliceName[2])
    axList[2].set_xlim(histMin, histMax)
    axList[2].set_ylim(histMin, histMax)
    axList[2].set_xlabel("B bins")
    axList[2].set_ylabel("G bins")
    axList[2].set_zlabel("Frequency")
    axList[2].legend([proxy1,proxy2],['Feature','Candidate'])

    if fig_name is not None:
        plt.savefig(dataDir + "/" + fig_name+"_histograms.png")
        plt.close(fig)
    else:
        plt.savefig("Fig.png")
        plt.close(fig)


''' Have to update this to perform Sequential Monte Carlo
    tracking, i.e. the particle filter steps.

    Currently this is doing naive color thresholding.
'''
def doTracking():
    global isTracking, image, r, g, b, trackedImage, hisFeature, xLast, yLast, frameCount, frameRate, outputF
    if isTracking:
        frameCount += 1
        print(" ** Frame count: ", frameCount)
        print(image.shape)

        # Compute the roi
        newRoi = GetRoi(xLast, yLast)
        newRoi = Normalize(newRoi)

        validRoiUpdate = False
        if newRoi is not None:
            validRoiUpdate = True

        if validRoiUpdate:
            newRoi = ConvolveWithKernel(newRoi)
            # Compute the pdf from the histogram and region of interest
            hisNew = cv2.calcHist([newRoi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth],
                                      [histMin, histMax, histMin, histMax, histMin, histMax])

            dist = ComputeHellinger(hisNew, hisFeature)

            if saveData:
                runName = "_Frame" + str(frameCount) + "_Iter0"
                dataFileName = dataString + runName
                if np.remainder(frameCount, frameRate * outputF) == 0:
                    hisFeatureList.append(hisNew)
                    hisFileLabelList.append(dataFileName)
                    hellingerDistList.append(dist)
                    candidateRoi.append(newRoi)

            mostProbableX, mostProbableY = GenerateNewTestPoint(xLast, yLast, nbrSize)
            dist_prev = dist
            it = 0
            while dist > eps and it < maxItr:
                xTest, yTest = mostProbableX, mostProbableY

                # Compute the new region of interest over the global coordinates
                newRoi = GetRoi(xTest, yTest)
                newRoi = Normalize(newRoi)

                validRoiUpdate = False
                if newRoi is not None:
                    validRoiUpdate = True

                if validRoiUpdate:
                    newRoi = ConvolveWithKernel(newRoi)
                    # Compute the pdf from the histogram and region of interest
                    hisNew = cv2.calcHist([newRoi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth],
                                          [histMin, histMax, histMin, histMax, histMin, histMax])
                    #if hisNew.sum() != 0:
                    #    hisNew /= hisNew.sum()
                    #else:
                    #    print(" * ERROR * Problem in normalization")

                    dist = ComputeHellinger(hisNew, hisFeature)
                    if dist < dist_prev:
                        # Update xLast and yLast to reflect the new global mean coordinates
                        xLast, yLast = xTest, yTest
                        dist_prev = dist

                    if saveData:
                        runName = "_Frame" + str(frameCount) + "_Iter" + str(it+1)
                        dataFileName = dataString + runName
                        if np.remainder(frameCount, frameRate * outputF) == 0:
                            hisFeatureList.append(hisNew)
                            hisFileLabelList.append(dataFileName)
                            hellingerDistList.append(dist)
                            candidateRoi.append(newRoi)

                    mostProbableX, mostProbableY = GenerateNewTestPoint(xLast, yLast, nbrSize)
                    print("Iteration count = ", it)
                    it += 1
                else:
                    mostProbableX, mostProbableY = GenerateNewTestPoint(xLast, yLast, nbrSize)

        #print("New location", xLast, yLast)

        cv2.rectangle(image, (xLast - rW, yLast - rH), (xLast + rW, yLast + rH), (255, 0, 0), 2)




def clickHandler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print('left button released')
        TuneTracker(x, y)



def mapClicks(x, y, curWidth, curHeight):
    global imageHeight, imageWidth
    imageX = x * imageWidth / curWidth
    imageY = y * imageHeight / curHeight
    return int(imageX), int(imageY)


def captureVideo(src):
    global image, isTracking, trackedImage, frameRate
    cap = cv2.VideoCapture(src)
    if cap.isOpened() and src == '0':
        ret = cap.set(3, 640) and cap.set(4, 480)
        if ret == False:
            print('Cannot set frame properties, returning')
            return
    else:
        frate = cap.get(cv2.CAP_PROP_FPS)
        frameRate = frate
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
            if saveData:
                print("**** Quitting program and saving all plots ***")
                PerformCleanup()

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

    if saveData:
        runCount = GetDataRunCount()
        dataString += runCount
        print("*** Starting tracker session: ", dataString)

    captureVideo(src)
else:
    print('Not in main')
