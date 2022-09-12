Custom Orthotic Shell Prediction-Short Summary Report
================

**Created by:** [Jana Osea](https://www.linkedin.com/in/jana-osea/)

**Supervised by:** [Nick
Wilkinson](https://www.linkedin.com/in/nicholas-wilkinson/)

**Project Duration:** May 2, 20220-August 26, 2022

*Note*: In collaboration with the University of British Columbia Co-op
Program and [Two Tall Totems Studios](https://ttt.studio/).

## Table of Contents

[1. Summary](#1-summary)

[2. Problem Description](#2-problem-description)

[3. Data Description](#3-data-description)

[4. Methods](#4-methods)

[5. Results](#5-results)

[6. Conclusion, Limitations, and Future
Work](#6-conclusion-limitations-and-future-work)

[7. References](#7-references)

## 1 Summary

As of current, deep learning has been widely used in 2D image data and
has significantly improved image recognition. However, learning methods
to predict objects in the 3D spaces is still in its infancy and is
rapidly growing. The main objective of this project is to explore novel
methods to predict the 3D shape of a shell given a 3D input of a foot.
Aside from being a useful application for custom orthotic prediction,
this project is also unique because 3D object prediction is a new and
exciting field in machine learning.

After careful consideration of various machine learning methods, we
conclude that a Convolution Neural Network (CNN) is an effective method
to predict a shell output for a corresponding cleaned foot input. Each
foot triangular mesh is processed into 2D images and labelled into 8
categories (wide or narrow width, long or short length, or low or high
arch). The 2 CNN model achieves an accuracy of 0.71 for both the right
and left foot correspondingly. Post processing after classification
produces a general rendering of a sample shell related to the predicted
label.

## 2 Problem Description

The main objective is to explore novel methods to predict the 3D shape
of a shell given a 3D input of a foot. This is a supervised machine
learning problem that uses cleaned foot scans and corresponding shell
scans to train a model to predict a shell given a cleaned foot scan
input.

## 3 Data Description

There are 2 main data sources *Direct Mill* and *Direct Mill 2* which
both contain raw foot scans, cleaned foot scans, and predicted shells.
*Direct Mill* and *Direct Mill 2* has 899 and 817 foot scans with
corresponding bottom and top shells. Below is a list of the missing data
in each corresponding category.

Direct Mill:

-   Right foot clean scan: 22
-   Right foot bottom shell: 0
-   Right foot top shell: 0
-   Left foot clean scan: 6
-   Left foot bottom shell: 0
-   Left foot top shell: 0

Direct Mill 2:

-   Right foot clean scan: 0
-   Right foot bottom shell: 0
-   Right foot top shell: 0
-   Left foot clean scan: 0
-   Left foot bottom shell: 0
-   Left foot top shell: 0

There is also additional data through tabular data called *Order form
details - Direct mill* that provides information about weight of
patient, shell material, shell length, and heel depth cut.

## 4 Methods

We use a Convolutional Neural Network (CNN) implemented using
[Tensorflow Keras Sequential
Model](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
to classify images of feet into 8 classifications according to whether
they have a combination of narrow or wide width, short or long length,
or low or high arch.

### 4.1 Mesh Processing and Data Loading

The triangular mesh data is first processed using
[Trimesh](https://trimsh.org/trimesh.html) and
[Matplotlib](https://matplotlib.org/) to transform the data into a 2D
image as shown below. Each foot was arbitrarily classified as one of the
8 labels according to if the max x, y, z vertex is above or below the
50th quantile. The figure below is a right foot classified with a narrow
width, short length, and low arch.

![Right Foot Label 1](SUMMARY_files/right-foot-1.jpeg)

On the contrary, the figure below is a foot classified with a wide
width, long length, and high arch.

![Right Foot Label 8](SUMMARY_files/right-foot-2.jpeg)

After creating the labels for the data and transforming them into 2D
images, the images were loaded using [Tensorflow
Util](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)
with a batch size of 32 and image height and width of 224. The dataset
was split using a 80-20 split for validation and normalized using a
rescaling layer from \[0, 255\] to \[0, 1\].

Note: For information about the details of epochs and batch sizes,
please refer to
[this](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
article.

### 4.2 Convolutional Neural Network

The Keras Sequential model has three convolution blocks
[(Conv2D)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
each with a max pooling layer
[(MaxPooling2D)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).
The last layer is a dense fully-connected layer
[(Dense)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
with 128 units that is activated by a ReLU activation function (‘relu’).
This is a standard approach for most convolutional neural networks. The
model uses an Adam optimizer which is a stochastic gradient descent
method along with [sparse categorical cross
entropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy)
loss function since there are more than 2 categories. Once the CNN model
is trained, the model is tested with the validation data set. The model
predicts a category for each foot input. Post processing would then
choose an arbitrary shell triangular mesh that the most resembles the
input foot using a series of functions. There are 2 CNN models
generated, one for each of the top shell prediction for the left and
right foot correspondingly.

## 5 Results

We specifically examine the results from the right foot shell model.
Please note that the results are very similar in the model for the left
foot shell prediction which will not be examined here due to redundancy.

The training and validation accuracy and loss can be seen below where
the x-axis is each epoch. The margin of accuracy between the training
and validation is not significantly different. This is paralleled by the
similar margin between training and validation loss as shown in the
right figure. The model achieves an accuracy of 0.71 which achieves far
better accuracy compared to CNN models with 27 and 12 categories with
0.21 and 0.48 accuracy correspondingly.

![Accuracy](SUMMARY_files/train-acc.png)

## 6 Conclusion, Limitations and Future Work

### 6.1 Conclusion

The main objective is to explore novel methods to predict the 3D shape
of a shell given a 3D input of a foot. After careful consideration of
various methods to generate models using triangular mesh data, we
conclude that using a Convolution Neural Network (CNN) is an effective
method to predict a shell output for a corresponding cleaned foot input.
Each foot triangular mesh is processed into 2D images and labelled into
8 categories according to the max values of the X, Y, and Z vertices.
The 2D images and their corresponding labels are used as input into a
CNN with 3 convolution layers each with a corresponding 2D max pooling
and a final flattening layer with a RELU activation. The 2 CNN model
achieves an accuracy of 0.71 for both the right and left foot
correspondingly. Post processing after classification produces a general
rendering of a sample shell related to the predicted label.

### 6.2 Limitations and Future Work

Limitations exist in that each foot is categorized into a specific label
based on the max values of the X, Y, and Z vertices. There are instances
where the foot scans contain noise vertices produced by man-made error
or production error. Hence, certain max values are not robust in
determining the max values of the actual foot. Future work lies in
producing a more robust method to label each foot that is not based on
absolute max values for each vertex axis.

Another limitation lies in choosing the shell triangular mesh after the
model predicts a label for a corresponding foot input. Due to time
constraints, this process is primitive in that it only takes an
arbitrary shell from each category. Future work lies in generating a
more extensive method to produce a corresponding shell triangular mesh
for each label.

## 7 References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-tensorflow2015-whitepaper" class="csl-entry">

Abadi, Martín, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen,
Craig Citro, Greg S. Corrado, et al. 2015. “TensorFlow: Large-Scale
Machine Learning on Heterogeneous Systems.”
<https://www.tensorflow.org/>.

</div>

<div id="ref-ari2014matplotlib" class="csl-entry">

Ari, Niyazi, and Makhamadsulton Ustazhanov. 2014. “Matplotlib in
Python.” In *2014 11th International Conference on Electronics, Computer
and Computation (ICECCO)*, 1–6. IEEE.

</div>

<div id="ref-cherchi2019py3dviewer" class="csl-entry">

Cherchi, Gianmarco, Luca Pitzalis, Giovanni Laerte Frongia, and Riccardo
Scateni. 2019. “The Py3DViewer Project: A Python Library for Fast
Prototyping in Geometry Processing.” In *STAG*, 121–28.

</div>

</div>
