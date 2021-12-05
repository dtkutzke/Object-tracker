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

'''global data common to all vision algorithms'''
isTracking = False
r = g = b = 0.0
image = np.zeros((640, 480, 3), np.uint8)
trackedImage = np.zeros((640, 480, 3), np.uint8)
imageHeight, imageWidth, planes = image.shape

'''(Demetri) Global variables for mean shift'''
dataDir = './Data'
dataString = 'tracker_run'
saveData = True
frameRate = 0
# How many seconds we want to save data (e.g., every 5 s)
outputF = 5

frameCount = 0

regionWidth = 60
rW = int(regionWidth / 2)
regionHeight = 60
rH = int(regionHeight / 2)
histBinWidth = 20
histMin = 0
histMax = 1
xLast = yLast = 0
eps = 0.5
#eps = 0.5
maxItr = 4
nbrSize = np.min([rW, rH])



# One histogram for every RGB value
hisFeature = np.zeros([histBinWidth, histBinWidth, histBinWidth])

# Histogram array
hisFeatureList = []
hisFileLabelList = []
hellingerDistList = []

'''Create a region of interest ROI around an x,y point'''
def getRoi(x, y):
    global imageHeight, imageWidth, rH, rW, image
    if (rW <= x < imageWidth - rW) and (rH <= y < imageHeight - rH):
        roi = image[y - rH:y + rH, x - rW:x + rW]
        return cv2.normalize(roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #return roi
    else:
        return None



'''Hellinger has the advantage of a mapping from reals to [0,1]'''
def calcHellinger(p1, p2):
    BC = cv2.compareHist(p1, p2, cv2.HISTCMP_BHATTACHARYYA)
    print("Hellinger: ", BC)
    return BC
    #if BC is not None:
    #    print("Hellinger", np.sqrt(1 - BC))
    #    return np.sqrt(1 - BC)
    #else:
    #    return None

def convolveWithKernel(roi_in):
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
    #fig, (ax1, ax2) = plt.subplots(2)
    #ax1.imshow(cv2.cvtColor(roi_in, cv2.COLOR_BGR2RGB))
    #ax1.set_title("Non-convolved")
    #ax2.imshow(cv2.cvtColor(cv2.filter2D(roi_in, -1, kernel), cv2.COLOR_BGR2RGB))
    #ax2.set_title("Convolved")
    return cv2.filter2D(roi_in, -1, kernel, None, anchor)

'''Defines a color model for the target of interest.
   Now, just reading pixel color at location
'''
def TuneTracker(x, y):

    global r, g, b, image, trackedImage, hisFeature, xLast, yLast

    xLast = x
    yLast = y

    # Bounding box defined by preset size
    roi = getRoi(x, y)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.savefig(dataDir+"/"+dataString+"_featureROI.png")

    # Convolve with a kernel
    roi = convolveWithKernel(roi)

    # Compute and normalize the histogram
    hisFeature = cv2.calcHist([roi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth], [histMin, histMax, histMin, histMax, histMin, histMax])

    b, g, r = np.unravel_index(np.argmax(hisFeature.flatten()), (histBinWidth, histBinWidth, histBinWidth))
    print("Max pixel bin for ", histBinWidth, "(", b, g, r, ")")



def saveConfigDetails():
    with open(dataDir+"/"+dataString+"_config.txt", 'a', encoding='utf-8') as f:
        f.write("*** CONFIGURATION FILE *** "+dataString+"\n")
        f.write("Roi width: " + str(regionWidth)+"\n")
        f.write("Roi height: " + str(regionHeight)+"\n")
        f.write("Histogram bin width: " + str(histBinWidth)+"\n")
        f.write("Histogram min/max: " + str(histMin)+","+str(histMax)+"\n")
        f.write("Tracker similarity threshold: " + str(eps)+"\n")
        f.write("Tracker maximum iterations to converge: " + str(maxItr)+"\n")
        f.write("Maximum neighborhood search size: " + str(nbrSize)+"\n")
        f.write("Maximum pixel bin frequency index (bgr): (" +str(b)+","+str(g)+","+str(r)+")\n")
        f.write("Figure output boolean: " + str(saveData)+"\n")


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


'''Essentially, compute the center of mass of a given pdf and return (x, y, z)'''
def computeCenterOfMass(hist_in):
    thresh = 0.0000001
    nB, nG, nR = hist_in.shape

    hist_flat = hist_in.flatten('C')

    # Compute the mean of the flattened array
    m = np.mean(hist_flat)

    hist_flat = abs(hist_flat - m)
    hist_flat = hist_flat - thresh
    idx = np.argmin(hist_flat)
    (b, g, r) = np.unravel_index(idx, (nB, nG, nR), 'C')
    return b, g, r

def getClosestIndexToValue(roi_in, b_in, g_in, r_in):
    height, width, planes = roi_in.shape
    # Flatten the input ROI to be (height*width) x 3
    roi_in_flat = np.reshape(roi_in, (height*width, planes), order='C')
    for i in range(height*width):
        if all(roi_in_flat[i] == b_in, g_in, r_in):
            return np.unravel_index(i, (height, width), 'C')

'''Generate a new target candidate location'''
def generateNewTestPoint(x_last, y_last, max_dist):
    global imageHeight, imageWidth, rH, rW, image
    y_new = np.random.randint(rH, imageHeight - rH)
    x_new = np.random.randint(rW, imageWidth-rW)
    while np.linalg.norm(np.array((x_new, y_new)) - np.array((x_last, y_last))) > max_dist:
        #x_new = np.random.randint(x_last - searchSize, x_last + searchSize)
        y_new = np.random.randint(rH, imageHeight - rH)
        x_new = np.random.randint(rW, imageWidth - rW)
        #y_new = np.random.randint(y_last - searchSize, y_last + searchSize)

    return x_new, y_new

def plotAndSaveHistogram(h1, h2=None, fig_name=None, h_dist=None):
    global b, g, r
    fig = plt.figure(figsize=(10, 4))
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
    sliceName = ["GR", "BR", "BG"]
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    #for i in range(3):
    ## ============= GR SLICE
    dz = h1[b,:,:].ravel()
    clrs1 = [colors[0]]*len(dz)
    ww = histMax/histBinWidth
    axList[0].bar3d(xpos, ypos, np.zeros_like(h1[b,:,:].ravel()), ww, ww, h1[b,:,:].ravel(), shade=False, color=clrs1, edgecolor='k')
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc=colors[0])
    if h2 is not None:
        axList[0].bar3d(xpos, ypos, np.zeros_like(h2[b,:,:].ravel()), ww, ww, h2[b,:,:].ravel(), shade=False, alpha=0.40, color=clrs1, edgecolor='k')
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc=colors[0], alpha=0.40)

    axList[0].set_title(sliceName[0])
    axList[0].set_xlim(histMin, histMax)
    axList[0].legend([proxy1,proxy2],['Feature','Candidate'])

    ## ============= BR SLICE
    dz = h1[:,g,:].ravel()
    clrs1 = [colors[1]]*len(dz)
    axList[1].bar3d(xpos, ypos, np.zeros_like(h1[:,g,:].ravel()), ww, ww, h1[:,g,:].ravel(), shade=False, color=clrs1, edgecolor='k')
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc=colors[1])
    if h2 is not None:
        axList[1].bar3d(xpos, ypos, np.zeros_like(h2[:,g,:].ravel()), ww, ww, h2[:,g,:].ravel(), shade=False, alpha=0.40, color=clrs1, edgecolor='k' )
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc=colors[1], alpha=0.40)

    axList[1].set_title(sliceName[1])
    axList[1].set_xlim(histMin, histMax)
    axList[1].legend([proxy1,proxy2],['Feature','Candidate'])

    ## ============= GR SLICE
    dz = h1[:,:,r].ravel()
    clrs1 = [colors[2]]*len(dz)
    axList[2].bar3d(xpos, ypos, np.zeros_like(h1[:,:,r].ravel()), ww, ww, h1[:,:,r].ravel(), shade=False, color=clrs1, edgecolor='k')
    proxy1 = plt.Rectangle((0, 0), 1, 1, fc=colors[2])
    if h2 is not None:
        axList[2].bar3d(xpos, ypos, np.zeros_like(h2[:,:,r].ravel()), ww, ww, h2[:,:,r].ravel(), shade=False, alpha=0.40, color=clrs1, edgecolor='k')
        proxy2 = plt.Rectangle((0, 0), 1, 1, fc=colors[2], alpha=0.40)

    axList[2].set_title(sliceName[2])
    axList[2].set_xlim(histMin, histMax)
    axList[2].legend([proxy1,proxy2],['Feature','Candidate'])
    #plt.show()

    if fig_name is not None:
        plt.savefig(dataDir + "/" + fig_name+"_histograms.png")
    else:
        plt.savefig("Fig.png")

    #print("Look at those plots!")


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
        newRoi = getRoi(xLast, yLast)

        validRoiUpdate = False
        if newRoi is not None:
            validRoiUpdate = True

        if validRoiUpdate:
            newRoi = convolveWithKernel(newRoi)
            # Compute the pdf from the histogram and region of interest
            hisNew = cv2.calcHist([newRoi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth],
                                      [histMin, histMax, histMin, histMax, histMin, histMax])



            #hisNew /= hisNew.sum()

            dist = calcHellinger(hisNew, hisFeature)

            if saveData:
                runName = "_Frame" + str(frameCount) + "_Iter0"
                dataFileName = dataString + runName
                if np.remainder(frameCount, frameRate * outputF) == 0:
                    #plotAndSaveHistogram(hisFeature, hisNew, dataFileName)
                    hisFeatureList.append(hisNew)
                    hisFileLabelList.append(dataFileName)
                    hellingerDistList.append(dist)
                    # print("====== Saving data now ======")

            mostProbableX, mostProbableY = generateNewTestPoint(xLast, yLast, nbrSize)
            dist_prev = dist
            it = 0
            while dist > eps and it < maxItr:
                xTest, yTest = mostProbableX, mostProbableY

                # Compute the new region of interest over the global coordinates
                newRoi = getRoi(xTest, yTest)


                validRoiUpdate = False
                if newRoi is not None:
                    validRoiUpdate = True

                if validRoiUpdate:
                    newRoi = convolveWithKernel(newRoi)
                    # Compute the pdf from the histogram and region of interest
                    hisNew = cv2.calcHist([newRoi], [0, 1, 2], None, [histBinWidth, histBinWidth, histBinWidth],
                                          [histMin, histMax, histMin, histMax, histMin, histMax])
                    #if hisNew.sum() != 0:
                    #    hisNew /= hisNew.sum()
                    #else:
                    #    print(" * ERROR * Problem in normalization")

                    dist = calcHellinger(hisNew, hisFeature)
                    if dist < dist_prev:
                        # Update xLast and yLast to reflect the new global mean coordinates
                        xLast, yLast = xTest, yTest
                        dist_prev = dist

                    if saveData:
                        runName = "_Frame" + str(frameCount) + "_Iter" + str(it+1)
                        dataFileName = dataString + runName
                        if np.remainder(frameCount, frameRate * outputF) == 0:
                           # plotAndSaveHistogram(hisFeature, hisNew, dataFileName)
                            hisFeatureList.append(hisNew)
                            hisFileLabelList.append(dataFileName)
                            hellingerDistList.append(dist)

                    mostProbableX, mostProbableY = generateNewTestPoint(xLast, yLast, nbrSize)
                    print("Iteration count = ", it)
                    it += 1
                else:
                    mostProbableX, mostProbableY = generateNewTestPoint(xLast, yLast, nbrSize)

        #print("New location", xLast, yLast)

        cv2.rectangle(image, (xLast - rW, yLast - rH), (xLast + rW, yLast + rH), (255, 0, 0), 2)


def performCleanup():
    for h, f, d in zip(hisFeatureList, hisFileLabelList, hellingerDistList):
        plotAndSaveHistogram(hisFeature, h, f, d)


def clickHandler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print('left button released')
        TuneTracker(x, y)


def getDataRunCount():
    if os.path.isdir(dataDir):
        # Load the files, extract iter from last file
        onlyfiles = [f for f in os.listdir(dataDir) if os.path.isfile(os.path.join(dataDir, f))]
        if onlyfiles:
            last_file = onlyfiles[-1]
            if last_file.find(dataString) == 0:
                return int(last_file[len(dataString)]) + 1
        else:
            return 1
    else:
        print("Creating data directory to save results")
        os.mkdir(dataDir)
        return 1



def mapClicksRoiToGlobal(x, y, bottom_x, bottom_y):
    return x + bottom_x, y + bottom_y


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
            print("Quitting program and saving all plots")
            if saveData:
                performCleanup()

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
        runCount = getDataRunCount()
        dataString += str(runCount)

    captureVideo(src)
else:
    print('Not in main')
