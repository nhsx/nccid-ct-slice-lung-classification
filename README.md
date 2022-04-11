# Lung image classification using EfficientNet

A demo of a classification model to classify if a mid-coronal plane of the CT scan is covering and focusing on the whole area of the patient’s lung or not.

## The model

I built a 2D deep learning model and applied transfer learning for this classification task, using the convolution neural network architecture EfficientNetB1 (Tan and Le, 2019) as the central component of the algorithm.

I applied transfer learning and used the pre-trained ImageNet (Russakovsky et al., 2015) weight of EfficientNetB1 to train the classification model on our training set. The training set images are processed by a sequential layer for the image augmentation before feeding into the EfficientNetB1 component. The features extracted by EfficientNetB1 are then averaged out by a 2D global-average-pooling operation, before feeding into a fully-connected layer (with random dropout rate of 0.3). The output layer of the model uses a softmax activation function to give the probabilities of the two classes ('whole lung' and 'others').

## The code

The main script of the demo is `whole_lung_classifier_demo.ipynb`. The customised utility functions are in the `src` folder. 

## Model performance on the test set
This trained classification model perform well on the hold-out test set, providing a weighted accuracy of 98% and an AUC-ROC > 0.99.

## Notes for the model
In the script, `data_folder_path` is the path to the data folder that contains all the JPG images that we used in this demo. This data path also contained a labels file, `whole_lung_label.csv`.

Please refer to our related paper: Poon and Lemarchand (in prep.) for further information. If you found this script useful, please cite:

```
@misc{Poon_Lemarchand_II,
       author = {{Poon}, Sanson T.S. and {Lemarchand}, Fran\c{c}ois},
        title = "{Baseline deep learning model trained on the UK National COVID-19 Chest Imaging Database - II: chest CT images}",
         year = in prep.
}
```

### References:

- S.T.S. Poon and F. Lemarchand. Baseline deep learning model trained on the UK National COVID-19 Chest Imaging Database - II: chest CT images, in prep
- O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. Imagenet large scale visual recognition challenge, 2015
- M. Tan and Q. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019. 9
