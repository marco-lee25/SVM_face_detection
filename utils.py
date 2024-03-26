from numpy import *
from sklearn import *
from scipy import ndimage
from scipy import signal

import skimage.color
import skimage.exposure
import skimage.io
import skimage.util
import fnmatch
import matplotlib.pyplot as plt
import zipfile
import os

def load_dataset(file_path, train_subsample=4, test_maxsample=472):
    # the dataset is too big, so subsample the training and test sets...
    # reduce training set by a factor of 4
    train_subsample = train_subsample  
    train_counter = [0, 0]
    # maximum number of samples in each class for test set
    test_maxsample = test_maxsample
    test_counter = [0, 0]

    imgdata = {'train':[], 'test':[]}
    classes = {'train':[], 'test':[]}

    zfile = zipfile.ZipFile(file_path, 'r')

    for name in zfile.namelist():
        # check file name matches
        if fnmatch.fnmatch(name, "faces/*/*/*.png"):

            # filename is : faces/train/face/fname.png
            (fdir1, fname)  = os.path.split(name)     # get file name
            (fdir2, fclass) = os.path.split(fdir1) # get class (face, nonface)
            (fdir3, fset)   = os.path.split(fdir2) # get training/test set
            # class 1 = face; class 0 = non-face
            myclass = int(fclass == "face")  

            loadme = False
            if fset == 'train':
                if (train_counter[myclass] % train_subsample) == 0:
                    loadme = True
                train_counter[myclass] += 1
            elif fset == 'test':
                if test_counter[myclass] < test_maxsample:
                    loadme = True
                test_counter[myclass] += 1

            if (loadme):
                # open file in memory, and parse as an image
                myfile = zfile.open(name)
                #img = matplotlib.image.imread(myfile)
                img = skimage.io.imread(myfile)[:,:,:3] # Dropping the alpha value
                # convert to grayscale
                img = skimage.color.rgb2gray(img)
                myfile.close()

                # append data
                imgdata[fset].append(img)
                classes[fset].append(myclass)


    zfile.close()
    imgsize = img.shape
    return imgdata, classes, imgsize


def image_montage(X, imsize=None, maxw=10):
    """X can be a list of images, or a matrix of vectorized images.
      Specify imsize when X is a matrix."""
    tmp = []
    numimgs = len(X)
    
    # create a list of images (reshape if necessary)
    for i in range(0,numimgs):
        if imsize != None:
            tmp.append(X[i].reshape(imsize))
        else:
            tmp.append(X[i])
    
    # add blanks
    if (numimgs > maxw) and (mod(numimgs, maxw) > 0):
        leftover = maxw - mod(numimgs, maxw)
        meanimg = 0.5*(X[0].max()+X[0].min())
        for i in range(0,leftover):
            tmp.append(ones(tmp[0].shape)*meanimg)
    
    # make the montage
    tmp2 = []
    for i in range(0,len(tmp),maxw):
        tmp2.append( hstack(tmp[i:i+maxw]) )
    montimg = vstack(tmp2) 
    return montimg


def extract_features(imgs, doplot=False):
    """
    The detection performance is not that good using pixel values. 
    The problem is that we are using the raw pixel values as features, so it is difficult for the classifier to 
    interpret larger structures of the face that might be important.  
    To fix the problem, we will extract features from the image using a set of filters.
    The filters are a sets of black and white boxes that respond to similar structures in the image.  
    After applying the filters to the image, the filter response map is aggregated over a 4x4 window.  
    Hence each filter produces a 5x5 feature response.  
    Since there are 4 filters, then the feature vector is 100 dimensions.
    """
    # the filter layout
    lay = [array([-1,1]), array([-1,1,-1]),  
               array([[1],[-1]]), array([[-1],[1],[-1]])]
    sc=8            # size of each filter patch
    poolmode = 'i'  # pooling mode (interpolate)
    cmode = 'same'  # convolution mode
    brick = ones((sc,sc))  # filter patch
    ks = []
    for l in lay:
        tmp = [brick*i for i in l]
        if (l.ndim==1):
            k = hstack(tmp)
        else:
            k = vstack(tmp)
        ks.append(k)

    # get the filter response size
    tmpimg = ndimage.zoom(imgs[0], 0.25)        
    fs = prod(tmpimg.shape)
    
    # get the total feature length
    fst = fs*len(ks)

    # filter the images
    X  = empty((len(imgs), fst))
    for i,img in enumerate(imgs):
        x = empty(fst)

        # for each filter
        for j,th in enumerate(ks):
            # filter the image
            imgk = signal.convolve(img, ks[j], mode=cmode)
            
            # do pooling
            mimg = ndimage.zoom(imgk, 0.25)
    
            # put responses into feature vector
            x[(j*fs):(j+1)*fs] = ravel(mimg)
               
            if (doplot):             
                plt.subplot(3,len(ks),j+1)
                plt.imshow(ks[j], cmap='gray', interpolation='nearest')
                plt.title("filter " + str(j))
                plt.subplot(3,len(ks),len(ks)+j+1)
                plt.imshow(imgk, cmap='gray', interpolation='nearest')
                plt.title("filtered image")
                plt.subplot(3,len(ks),2*len(ks)+j+1)
                plt.imshow(mimg, cmap='gray', interpolation='nearest')
                plt.title("image features")
        X[i,:] = x
    
    return X


def err_analysis(testY, predY, method="pixel_based"):
    # predY is the prediction from the classifier
    Pind = where(testY==1) # indicies for face
    Nind = where(testY==0) # indicies for non-face

    TP = count_nonzero(testY[Pind] == predY[Pind])
    FN = count_nonzero(testY[Pind] != predY[Pind])
    TN = count_nonzero(testY[Nind] == predY[Nind])
    FP = count_nonzero(testY[Nind] != predY[Nind])

    TPR = TP / (TP+FN)
    FPR = FP / (FP+TN)
    print("==============")
    print("Method : ", method)
    print("TP=", TP)
    print("FP=", FP)
    print("TN=", TN)
    print("FN=", FN)
    print("TPR=", TPR)
    print("FPR=", FPR)