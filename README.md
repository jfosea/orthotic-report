# Custom Orthotic Shell Prediction

**Created by:** [Jana Osea](https://www.linkedin.com/in/jana-osea/)

**Supervised by:** [Nick Wilkinson](https://www.linkedin.com/in/nicholas-wilkinson/)

**Project Duration:** May 2, 2022-August 26, 2022

*Note*: In collaboration with the University of British Columbia Co-op Program and [Two Tall Totems Studios](https://ttt.studio/).


## Project Summary

As of current, deep learning has been widely used in 2D image data and has significantly improved image recognition. However, learning methods to predict objects in the 3D spaces is still in its infancy and is rapidly growing. The main objective of this project is to explore novel methods to predict the 3D shape of a shell given a 3D input of a foot. Aside from being a useful application for custom orthotic prediction, this project is also unique because 3D object prediction is a new and exciting field in machine learning.

After careful consideration of various machine learning methods, we conclude that a Convolution Neural Network (CNN) is an effective method to predict a shell output for a corresponding cleaned foot input. Each foot triangular mesh is processed into 2D images and labelled into 8 categories (wide or narrow width, long or short length, or low or high arch). The 2 CNN model achieves an accuracy of 0.71 for both the right and left foot correspondingly. Post processing after classification produces a general rendering of a sample shell related to the predicted label.


## Reports

Click [here](https://github.com/jfosea/orthotic-report/blob/main/SUMMARY_SHORT_REPORT.md) for the *short* summary of the project. 

Click [here](https://github.com/jfosea/orthotic-report/blob/main/SUMMARY_FULL_REPORT.md) for the *full* summary of the project. 


