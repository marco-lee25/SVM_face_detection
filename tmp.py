import IPython.core.display         
# setup output image format (Chrome works best)
IPython.core.display.set_matplotlib_formats("svg")
import matplotlib.pyplot as plt
import matplotlib
from numpy import *
from sklearn import *
import os
import zipfile
import fnmatch
from scipy import ndimage
from scipy import signal
from sklearn import preprocessing
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
import skimage.color
import skimage.exposure
import skimage.io
import skimage.util

from utils import image_montage, extract_features, load_dataset
root_dir = os.path.dirname(__file__)
random.seed(4487)

def detect(trainX, testX):
    clfs = {}
    # setup all the parameters and models
    exps = {
        'svm-lin': {
            'paramgrid': {'C': logspace(-2,3,10)},
            'clf': svm.SVC(kernel='linear') },
        'svm-rbf': {
            'paramgrid': {'C': logspace(-2,3,10), 'gamma': logspace(-4,3,10) },
            'clf': svm.SVC(kernel='rbf') },
        'svm-poly': {
            'paramgrid': {'C': logspace(-2,3,10), 'degree': [2, 3, 4] },
            'clf': svm.SVC(kernel='poly') },
    }
    # Normalization: sklearn.preprocessing.MinMaxScaler()
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))    # make scaling object
    trainXn = scaler.fit_transform(trainX)  # use training data to fit scaling parameters
    testXn  = scaler.transform(testX)  # apply scaling to test data

    for (name, exp) in exps.items():
        svc = exp['clf']
        clfs[name] = model_selection.GridSearchCV(svc, param_grid=exp['paramgrid'], cv=5, verbose=1, n_jobs=-1)
        clfs[name].fit(trainXn, trainY)
        predYtrain = {}
        predYtest  = {}
        for (name, clf) in clfs.items():
            predYtrain[name] = clfs[name].predict(trainXn)
            predYtest[name] = clfs[name].predict(testXn)
            trainingAccuracyScore = metrics.accuracy_score(predYtrain[name], trainY)
            testAccuracyScore = metrics.accuracy_score(predYtest[name], testY)
            print("Training Accuracy - "+ name + ": " + str(trainingAccuracyScore))
            print("Testing Accuracy - "+ name + ": " + str(testAccuracyScore))
    return predYtrain, predYtest, clfs


def err_analysis(testY, predY):
    # predY is the prediction from the classifier
    Pind = where(testY==1) # indicies for face
    Nind = where(testY==0) # indicies for non-face

    TP = count_nonzero(testY[Pind] == predY[Pind])
    FN = count_nonzero(testY[Pind] != predY[Pind])
    TN = count_nonzero(testY[Nind] == predY[Nind])
    FP = count_nonzero(testY[Nind] != predY[Nind])

    TPR = TP / (TP+FN)
    FPR = FP / (FP+TN)

    print("TP=", TP)
    print("FP=", FP)
    print("TN=", TN)
    print("FN=", FN)
    print("TPR=", TPR)
    print("FPR=", FPR)



if __name__ == "__main__":
    filename = 'faces.zip'
    imgdata, classes, imgsize = load_dataset(os.path.join(root_dir,filename), 4, 472)
    trainclass2start = sum(classes['train'])

    # plt.subplot(1,2,1)
    # plt.imshow(imgdata['train'][0], cmap='gray', interpolation='nearest')
    # plt.title("face sample")
    # plt.subplot(1,2,2)
    # plt.imshow(imgdata['train'][trainclass2start], cmap='gray', interpolation='nearest')
    # plt.title("non-face sample")
    # plt.show()


    # show a few images
    # plt.figure(figsize=(9,9))
    # plt.imshow(image_montage(imgdata['train'][::20]), cmap='gray', interpolation='nearest')
    # plt.show()


    # Each image is a 2d array, but the classifier algorithms work on 1d vectors. 
    # We convert all the images into 1d vectors by flattening.  
    # The result should be a matrix where each row is a flattened image.

    trainX = empty((len(imgdata['train']), prod(imgsize)))
    for i,img in enumerate(imgdata['train']):
        trainX[i,:] = ravel(img)
    trainY = asarray(classes['train'])  # convert list to numpy array
    print(trainX.shape)
    print(trainY.shape)

    testX = empty((len(imgdata['test']), prod(imgsize)))
    for i,img in enumerate(imgdata['test']):
        testX[i,:] = ravel(img)
    testY = asarray(classes['test'])  # convert list to numpy array
    print(testX.shape)
    print(testY.shape)


    # =========================================
    # Detection Using  Pixel Values
    # Train kernel SVM using either RBF or polynomia kernel classifiers to classify an image patch as face or non-face. 
    # Evaluate all classifiers on the test set.
    # Normalize the features and setup all the parameters and models.
    predYtrain, predYtest, clfs = detect(trainX, testX)

    # set variables for later
    predY = predYtest['svm-poly']
    #adaclf = clfs['ada'].best_estimator_
    svmclf_rbf = clfs['svm-rbf'].best_estimator_
    svmclf_poly = clfs['svm-poly'].best_estimator_
    #rfclf  = clfs['rf'].best_estimator_


    # Error analysis
    err_analysis(testY, predY)

    # For kernel SVM, we can look at the support vectors to see what the classifier finds difficult.
    # svmclf is the trained SVM classifier

    # Here we can see that there is only one false positive but so many false negatives, which means it predicts it to be non-face when it is supposed to predict it as a face. 
    # In addition we can see that the true positive rate is also less, which means it is not able to predict the faces properly

    print("num support vectors:", len(svmclf_poly.support_vectors_))
    si  = svmclf_poly.support_  # get indicies of support vectors

    # get all the patches for each support vector
    simg = [ imgdata['train'][i] for i in si ]

    # make montage
    outimg = image_montage(simg, maxw=20)

    plt.figure(figsize=(9,9))
    plt.imshow(outimg, cmap='gray', interpolation='nearest')
    plt.show()

    # =====================================
    # Detecction using Image Feature

    # new features
    img = imgdata['train'][0]
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title("image")
    plt.figure(figsize=(9,9))
    plt.show()

    extract_features([img], doplot=True)
    plt.show()

    # Extract image features on the training and test sets
    trainXf = extract_features(imgdata['train'])
    print(trainXf.shape)
    testXf = extract_features(imgdata['test'])
    print(testXf.shape)


    # Train AdaBoost and SVM classifiers on the image feature data.  Evaluate on the test set.
    ### Nomalization
    scalerf = preprocessing.MinMaxScaler(feature_range=(-1,1))    # make scaling object
    trainXfn = scalerf.fit_transform(trainXf)   # use training data to fit scaling parameters
    testXfn  = scalerf.transform(testXf)        # apply scaling to test data

    clfs2 = {}
    # setup all the parameters and models
    exps = {
        'svm-lin': {
            'paramgrid': {'C': logspace(-2,3,10)},
            'clf': svm.SVC(kernel='linear') },
        'svm-rbf': {
            'paramgrid': {'C': logspace(-2,3,10), 'gamma': logspace(-4,3,10) },
            'clf': svm.SVC(kernel='rbf') },
        'svm-poly': {
            'paramgrid': {'C': logspace(-2,3,10), 'degree': [2, 3, 4] },
            'clf': svm.SVC(kernel='poly') },       
        }

    for (name, exp) in exps.items():
        svc = exp['clf']
        clfs2[name] = model_selection.GridSearchCV(svc, param_grid=exp['paramgrid'], cv=5, verbose=1, n_jobs=-1)
        clfs2[name].fit(trainXfn, trainY)

    for (name, clf) in clfs2.items():
        print(clfs2[name].best_estimator_)


    # Calculate the training and test accuracy for the each classifier.
    predYtrain = {}
    predYtest  = {}
    for (name, clf) in clfs2.items():
        predYtrain[name] = clfs2[name].predict(trainXfn)
        predYtest[name] = clfs2[name].predict(testXfn)
        trainingAccuracyScore = metrics.accuracy_score(predYtrain[name], trainY)
        testAccuracyScore = metrics.accuracy_score(predYtest[name], testY)
        print("Training Accuracy - "+ name + ": " + str(trainingAccuracyScore))
        print("Testing Accuracy - "+ name + ": " + str(testAccuracyScore))

    testY = predYtest['svm-rbf']
    bestclf = clfs2['svm-rbf']


    # Error Analysis
    ft_poly_test_pred = predYtest['svm-poly']
    Pind = where(testY==1) # indicies for face
    Nind = where(testY==0) # indicies for non-face
    #print(Pind)
    TP = count_nonzero(testY[Pind] == ft_poly_test_pred[Pind])
    FN = count_nonzero(testY[Pind] != ft_poly_test_pred[Pind])
    TN = count_nonzero(testY[Nind] == ft_poly_test_pred[Nind])
    FP = count_nonzero(testY[Nind] != ft_poly_test_pred[Nind])

    TPR = TP / (TP+FN)
    FPR = FP / (FP+TN)

    print("TP =", TP)
    print("FP =", FP)
    print("TN =", TN)
    print("FN =", FN)
    print("TPR =", TPR)
    print("FPR =", FPR)


    # Test image
    fname = "nasa-small.png"
    testimg3 = skimage.io.imread(os.path.join(root_dir,fname))[:,:,:3]

    # convert to grayscale
    testimg = skimage.color.rgb2gray(testimg3)
    print(testimg.shape)
    plt.imshow(testimg, cmap='gray')
    plt.show()

    # step size for the sliding window
    step = 4

    # extract window patches with step size of 4
    patches = skimage.util.view_as_windows(testimg, (19,19), step=step)
    psize = patches.shape
    # collapse the first 2 dimensions
    patches2 = patches.reshape((psize[0]*psize[1], psize[2], psize[3]))
    print(patches2.shape )

    # histogram equalize patches (improves contrast)
    patches3 = empty(patches2.shape)
    for i in range(patches2.shape[0]):
        patches3[i,:,:] = skimage.exposure.equalize_hist(patches2[i,:,:])

    # extract features
    newXf = extract_features(patches3)

    testXfn = scalerf.transform(newXf)    
    prednewY = bestclf.predict(testXfn)

    # reshape prediction to an image
    imgY = prednewY.reshape(psize[0], psize[1])

    # zoom back to image size
    imgY2 = ndimage.zoom(imgY, step, output=None, order=0)
    # pad the top and left with half the window size
    imgY2 = vstack((zeros((9, imgY2.shape[1])), imgY2))
    imgY2 = hstack((zeros((imgY2.shape[0],9)), imgY2))
    # pad right and bottom to same size as image
    if (imgY2.shape[0] != testimg.shape[0]):
        imgY2 = vstack((imgY2, zeros((testimg.shape[0]-imgY2.shape[0], imgY2.shape[1]))))
    if (imgY2.shape[1] != testimg.shape[1]):
        imgY2 = hstack((imgY2, zeros((imgY2.shape[0],testimg.shape[1]-imgY2.shape[1]))))

    # show detections with image
    #detimg = dstack(((0.5*imgY2+0.5)*testimg, 0.5*testimg, 0.5*testimg))
    nimgY2 = 1-imgY2
    tmp = nimgY2*testimg
    detimg = dstack((imgY2+tmp, tmp, tmp))

    # show it!
    plt.figure(figsize=(9,9))
    plt.subplot(2,1,1)
    plt.imshow(imgY2, interpolation='nearest')
    plt.title('detection map')
    plt.subplot(2,1,2)
    plt.imshow(detimg)
    plt.title('image')
    plt.axis('image')

    plt.show()