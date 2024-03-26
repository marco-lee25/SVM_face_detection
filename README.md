# Face_detection

### Project Objective

Experiment with Support Vector Machine and Kernel Support Vector Machine classifiers to detect whether there is a face in a small image patch. And compare their performance.

### Dataset
The dataset contains face and non-face images with 19x19 pixel values.

| Data overview  | Face and non-face sample |
| ------------- | ------------- |
| <img src="https://github.com/marco-lee25/Face_detection/blob/main/imgs/train_data.png" width="400"/>  | <img src="https://github.com/marco-lee25/Face_detection/blob/main/imgs/data_sample.png" width="400"/>  |

### Support Vector Machine
This project mainly relied on the library Scikit-learn, and the SVM module provides 3 classes capable of performing binary and multi-class classification.

<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_001.png" width="600"/> 

_Image from_ https://scikit-learn.org/stable/modules/svm.html

### Error analysis
The accuracy only tells part of the classifier's performance. We can also look at the different types of errors that the classifier makes:
- _True Positive (TP)_: classifier correctly said face
- _True Negative (TN)_: classifier correctly said non-face
- _False Positive (FP)_: classifier said face, but not a face
- _False Negative (FN)_: classifier said non-face, but was a face

This is summarized in the following table:

<table>
<tr><th colspan=2 rowspan=2><th colspan=2 style="text-align: center">Actual</th></tr>
<tr>  <th>Face</th><th>Non-face</th></tr>
<tr><th rowspan=2>Prediction</th><th>Face</th><td>True Positive (TP)</td><td>False Positive (FP)</td></tr>
<tr>  <th>Non-face</th><td>False Negative (FN)</td><td>True Negative (TN)</td></tr>
</table>

We can then look at the _true positive rate_ and the _false positive rate_.
- _true positive rate (TPR)_: proportion of true faces that were correctly detected
- _false positive rate (FPR)_: proportion of non-faces that were mis-classified as faces.

  
### SVM on Pixel Values

Train the SVM classifiers directly based on the pixel.
```
Fitting 5 folds for each of 10 candidates, totalling 50 fits
Fitting 5 folds for each of 100 candidates, totalling 500 fits
Fitting 5 folds for each of 30 candidates, totalling 150 fits
Training Accuracy - svm-lin: 0.9833810888252149
Testing Accuracy - svm-lin: 0.6430084745762712
Training Accuracy - svm-rbf: 1.0
Testing Accuracy - svm-rbf: 0.6578389830508474
Training Accuracy - svm-poly: 1.0
Testing Accuracy - svm-poly: 0.6620762711864406
TP= 154
FP= 1
TN= 471
FN= 318
TPR= 0.326271186440678
FPR= 0.00211864406779661
```

**The detection performance is not that good using pixel values.**<br /> The problem is that we are using the raw pixel values as features, so it is difficult for the classifier to interpret larger structures of the face that might be important. We can see that there is only one false positive but so many false negatives, which means it predicts it to be non-face when it is supposed to predict it as a face. In addition, we can see that the true positive rate is also less, which means it is not able to predict the faces properly.

### SVM on Image Feature
To fix the problem of SVM using pixel values, we can extract features from the image using a set of filters.
The filters are sets of black and white boxes that respond to similar structures in the image.
After applying the filters to the image, the filter response map is aggregated over a 4x4 window.
Hence each filter produces a 5x5 feature response.
Since there are 4 filters, then the feature vector is 100 dimensions.






 
