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

from utils import image_montage, extract_features, load_dataset, err_analysis

root_dir = os.path.dirname(__file__)
random.seed(2024)

def detect(trainX, testX, method="pixel_based"):
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
    print("==============")
    print("Method : ", method)
    for (name, clf) in clfs.items():
        predYtrain[name] = clfs[name].predict(trainXn)
        predYtest[name] = clfs[name].predict(testXn)
        trainingAccuracyScore = metrics.accuracy_score(predYtrain[name], trainY)
        testAccuracyScore = metrics.accuracy_score(predYtest[name], testY)
        print("Training Accuracy - "+ name + ": " + str(trainingAccuracyScore))
        print("Testing Accuracy - "+ name + ": " + str(testAccuracyScore))
    return predYtrain, predYtest, clfs, scaler


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
    # # plt.show()
    # plt.savefig(os.path.join(root_dir, "imgs", "data_sample.png"))

    # # show a few images
    # plt.figure(figsize=(9,9))
    # plt.imshow(image_montage(imgdata['train'][::20]), cmap='gray', interpolation='nearest')
    # # plt.show()
    # plt.savefig(os.path.join(root_dir, "imgs", "train_data.png"))

    """
    Each image is a 2d array, but the classifier algorithms work on 1d vectors.
    We convert all the images into 1d vectors by flattening.  
    The result should be a matrix where each row is a flattened image.
    """
    trainX = empty((len(imgdata['train']), prod(imgsize)))
    for i,img in enumerate(imgdata['train']):
        trainX[i,:] = ravel(img)
    trainY = asarray(classes['train'])  # convert list to numpy array
    # print(trainX.shape)
    # print(trainY.shape)

    testX = empty((len(imgdata['test']), prod(imgsize)))
    for i,img in enumerate(imgdata['test']):
        testX[i,:] = ravel(img)
    testY = asarray(classes['test'])  # convert list to numpy array
    # print(testX.shape)
    # print(testY.shape)

    """
    ****** Detection Using Pixel Values ******
    Train kernel SVM using either RBF or polynomia kernel classifiers to classify an image patch as face or non-face. 
    Evaluate all classifiers on the test set.
    Normalize the features and setup all the parameters and models.
    """
    predYtrain, predYtest, clfs, _ = detect(trainX, testX, "pixel_based")
    # set variables for later
    predY = predYtest['svm-poly']
    #adaclf = clfs['ada'].best_estimator_
    svmclf_rbf = clfs['svm-rbf'].best_estimator_
    svmclf_poly = clfs['svm-poly'].best_estimator_
    #rfclf = clfs['rf'].best_estimator_
    # Error analysis
    err_analysis(testY, predY, "pixel_based")

    """
    For kernel SVM, we can look at the support vectors to see what the classifier finds difficult. svmclf is the trained SVM classifier.
    We can see that there is only one false positive but so many false negatives, which means it predicts it to be non-face when it is supposed to predict it as a face. 
    In addition we can see that the true positive rate is also less, which means it is not able to predict the faces properly
    """
    print("num support vectors:", len(svmclf_poly.support_vectors_))
    si  = svmclf_poly.support_  # get indicies of support vectors
    # get all the patches for each support vector
    simg = [imgdata['train'][i] for i in si ]
    # make montage
    outimg = image_montage(simg, maxw=20)
    plt.figure(figsize=(9,9))
    plt.imshow(outimg, cmap='gray', interpolation='nearest')
    plt.savefig(os.path.join(root_dir, "imgs", "sv_img.png"))

    """
    ****** Detecction using Image Feature ******
    The detection performance is not that good using pixel values. The problem is that we are using the raw pixel values as features, 
    so it is difficult for the classifier to interpret larger structures of the face that might be important.  
    To fix the problem, we will extract features from the image using a set of filters.

    The filters are a sets of black and white boxes that respond to similar structures in the image.  
    After applying the filters to the image, the filter response map is aggregated over a 4x4 window.  
    Hence each filter produces a 5x5 feature response.  
    Since there are 4 filters, then the feature vector is 100 dimensions.
    """
    # new features
    img = imgdata['train'][0]
    plt.figure(figsize=(9,9))
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title("image")
    plt.savefig(os.path.join(root_dir, "imgs", "target_img_for_filter.png"))
    
    plt.figure(figsize=(9,9))
    extract_features([img], doplot=True)
    plt.savefig(os.path.join(root_dir, "imgs", "img_filtered.png"))

    # Extract image features on the training and test sets
    trainXf = extract_features(imgdata['train'])
    # print(trainXf.shape)
    testXf = extract_features(imgdata['test'])
    # print(testXf.shape)

    # Train AdaBoost and SVM classifiers on the image feature data.  Evaluate on the test set.
    predYtrain, predYtest, clfs, scalerf = detect(trainXf, testXf, "feature_extraction")
    testY = predYtest['svm-rbf']
    bestclf = clfs['svm-rbf']
    ft_poly_test_pred = predYtest['svm-poly']

    # Error Analysis
    err_analysis(testY, ft_poly_test_pred, "feature_extraction")

    # ====================================================
    # Test image
    fname = "nasa-small.png"
    testimg3 = skimage.io.imread(os.path.join(root_dir,fname))[:,:,:3]

    # convert to grayscale
    testimg = skimage.color.rgb2gray(testimg3)
    # print(testimg.shape)
    # plt.imshow(testimg, cmap='gray')
    # plt.show()

    # step size for the sliding window
    step = 4

    # extract window patches with step size of 4
    patches = skimage.util.view_as_windows(testimg, (19,19), step=step)
    psize = patches.shape
    # collapse the first 2 dimensions
    patches2 = patches.reshape((psize[0]*psize[1], psize[2], psize[3]))

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
    plt.savefig(os.path.join(root_dir, "imgs", "test_img_result.png"))