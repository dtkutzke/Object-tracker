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
r=g=b=0.0
image = np.zeros((640,480,3), np.uint8)
trackedImage = np.zeros((640,480,3), np.uint8)
imageWidth=imageHeight=0

'''Defines a color model for the target of interest.
   Now, just reading pixel color at location
'''
def TuneTracker(x,y):
    global r,g,b, image, trackedImage
    trackedImage = image[x-10:x+10, y-14:y+14]
    #plt.imshow(trackedImage)
    b,g,r = image[y,x]
    sumpixels = float(b)+float(g)+float(r)
    if sumpixels != 0:
        b = b/sumpixels,
        r = r/sumpixels
        g = g/sumpixels
    print( r,g,b, 'at location ', x,y ) 


''' Have to update this to perform Sequential Monte Carlo
    tracking, i.e. the particle filter steps.

    Currently this is doing naive color thresholding.
'''
def doTracking():
    global isTracking, image, r,g,b, trackedImage
    if isTracking:
        print( image.shape )
        imheight, imwidth, implanes = image.shape
#        f = np.fft.fft2(image)
#        fshift = np.fft.fftshift(f)
#        magnitude_spectrum = 20 * np.log(np.abs(fshift))
#        # Mask the data so that pixels <= 0 are 0
#        magnitude_spectrum[magnitude_spectrum < 0] = 0
#        magnitude_spectrum[magnitude_spectrum > 255] = 255
#        magnitude_spectrum_int = magnitude_spectrum.astype(int)
#        crow, ccol = imheight // 2, imwidth // 2
#        # High pass filtering
#        fshift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0
        # Low pass filtering
        #fshift[0:crow - 30, 0:ccol - 30] = 0
        #fshift[crow + 31:, ccol + 31:] = 0
        # f_ishift = np.fft.ifftshift(fshift)
        # img_back = np.fft.ifft2(f_ishift)
        # img_back = np.real(img_back)
        # img_back[img_back < 0] = 0
        #for f in range(3):
        #    trackedImage[:,:,f] /= np.amax(trackedImage[:,:,f])

        his = cv2.calcHist([trackedImage], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        # Now we convolve the distribution
        #for h in range(3):
        #    plt.plot(his[h])

        #print('Hello world')
        # plt.subplot(131), plt.imshow(image, cmap='gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(132), plt.imshow(magnitude_spectrum_int, cmap='gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        # plt.subplot(133), plt.imshow(img_back)
        # plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
        plt.show()
        for j in range( imwidth ):
            for i in range( imheight ):
                bb, gg, rr = image[i,j]
                sumpixels = float(bb)+float(gg)+float(rr)
                if sumpixels == 0:
                    sumpixels = 1
                if rr/sumpixels >= r and gg/sumpixels >= g and bb/sumpixels >= b:
                    image[i,j] = [255,255,255];
                else:
                    image[i,j] = [0,0,0];                    
    

def clickHandler( event, x,y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print( 'left button released' )
        TuneTracker( x, y )


def mapClicks( x, y, curWidth, curHeight ):
    global imageHeight, imageWidth
    imageX = x*imageWidth/curWidth
    imageY = y*imageHeight/curHeight
    return imageX, imageY
        
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
    
