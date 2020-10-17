# Digit-Recognizer
In this repository, I will introduce some algorithms for the famous dataset, handwritten digit recognization challenge, including Generalized Linear Model, Neural Networks etc.
The raw data is from kaggle data challenge, Digit Recognizer, the url is https://www.kaggle.com/c/digit-recognizer


I implemented LightGBM method here first. Its final score is approximately 0.96. Then I used Neural Network model. I used simple full connected model and CNN. The result is 0.9917. After this, I improved my CNN structure, and added the data augmentation part. This is essential improvement. And finally my score reached to 9.9970, after adapting the ensemble method for create 15 CNN models.

The main package used in this repo is Keras, Scikit-Learn, Pandas, Numpy.

I will try to do more about this dataset, and this repo may be updated every now and then XD.
