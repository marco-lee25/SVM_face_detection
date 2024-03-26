# Face_detection

### Project Objective

Experiment with Support Vector Machine and Kernel Support Vector Machine classifiers to detect whether there is a face in a small image patch. And compare their performance.

### Dataset
The dataset contains face and non-face images with 19x19 pixel values.

| Data overview  | Face and non-face sample |
| ------------- | ------------- |
| <img src="https://github.com/marco-lee25/Face_detection/blob/main/imgs/train_data.png" width="400"/>  | <img src="https://github.com/marco-lee25/Face_detection/blob/main/imgs/data_sample.png" width="400"/>  |

### SVM training
This project mainly relied on the library Scikit-learn, and the SVM module provides 3 classes capable of performing binary and multi-class classification.

<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_001.png" width="600"/>
_Image from_  
https://scikit-learn.org/stable/modules/svm.html


_This text is italicized_


###### SVM on Pixel Values
Directly train the SVM classifiers based on the pixel.


 
