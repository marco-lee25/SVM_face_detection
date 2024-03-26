# Face_detection_using_SVM

### Project Objective

Experiment with Support Vector Machine and Kernel Support Vector Machine classifiers to detect whether there is a face in a small image patch. And compare their performance.

### Dataset
The dataset contains face and non-face images with 19x19 pixel values.<br />
REMARK：This project only used part of the images as dataset is large.

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
###### Result
```
==============
Method :  pixel_based
Training Accuracy - svm-lin: 0.9833810888252149
Testing Accuracy - svm-lin: 0.6430084745762712
Training Accuracy - svm-rbf: 1.0
Testing Accuracy - svm-rbf: 0.6578389830508474
Training Accuracy - svm-poly: 1.0
Testing Accuracy - svm-poly: 0.6620762711864406
==============
Method :  pixel_based
TP= 154
FP= 1
TN= 471
FN= 318
TPR= 0.326271186440678
FPR= 0.00211864406779661
```
**The support vector** <br />
There are a total of 307 images are chosen to be the support vectors from the best classifier - _SVM-Poly_ <br />
<img src="https://github.com/marco-lee25/Face_detection/blob/main/imgs/sv_img.png" width="600"/>

**The detection performance is not that good using pixel values.** <br /> The problem is that we are using the raw pixel values as features, so it is difficult for the classifier to interpret larger structures of the face that might be important. We can see that there is only one false positive but so many false negatives, which means it predicts it to be non-face when it is supposed to predict it as a face. In addition, we can see that the true positive rate is also less, which means it is not able to predict the faces properly.

### SVM on Image Feature
To fix the problem of SVM using pixel values, we can extract features from the image using a set of filters.
The filters are sets of black and white boxes that respond to similar structures in the image.
After applying the filters to the image, the filter response map is aggregated over a 4x4 window.
Hence each filter produces a 5x5 feature response.
Since there are 4 filters, then the feature vector is 100 dimensions.

| Target image  | Feature extraced |
| ------------- | ------------- |
| <img src="https://github.com/marco-lee25/Face_detection/blob/main/imgs/target_img_for_filter.png" width="400"/>  | <img src="https://github.com/marco-lee25/Face_detection/blob/main/imgs/img_filtered.png" width="400"/>  |

Train the SVM classifiers on feature extracted from images.
###### Result
~~~
==============
Method :  feature_extraction
Training Accuracy - svm-lin: 0.9621776504297994
Testing Accuracy - svm-lin: 0.7129237288135594
Training Accuracy - svm-rbf: 0.9977077363896848
Testing Accuracy - svm-rbf: 0.7341101694915254
Training Accuracy - svm-poly: 0.9914040114613181
Testing Accuracy - svm-poly: 0.7542372881355932
==============
Method :  feature_extraction
TP= 226
FP= 32
TN= 665
FN= 21
TPR= 0.9149797570850202
FPR= 0.04591104734576758
~~~
### Performance on test data 
<img src="https://github.com/marco-lee25/Face_detection/blob/main/imgs/test_img_result.png" width="800"/>

Although it doesn't seem very accurate since we can see that it detected a few faces, it also had some false positives and didn't detect as many True positives as it should have.<br />
But please be reminded that this project only used one-quarter of the dataset, the performance could be improved if numbers of images increase. 

 ### Train your own SVM!
~~~
python main.py --data_subsample 4 --max_test 472 --random_seed --seed 2024 --method all

--data_subsample : The factor of sample the dataset
--max_test : The maximum number of test samples for each class
--random_seed :Fixing the random seed
--seed : The random seed
--method : ["all", "pixel_based", "feature_based"]
--test_img : Your own test_img, default using the nasa-small.png
~~~
