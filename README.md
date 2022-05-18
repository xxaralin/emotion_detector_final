# Emotion_detector
In this project we tried to find an answer to the question how well can machines recognize us and our emotions just by looking at us. In order to do that we got help from the machine learning and image processing fields. We found a data set that consists of images of 7 most common human emotions and corresponding expressions of human faces. We used this dataset to train our model and later used this trained model to make predictions on our captured images from our cameras. Since machine learning and artificial intelligence methods vary in each case, we tried a few different approaches and few different hyperparameters to achieve the best results. In conclusion we got the result of 80% accuracy for training data and 64% accuracy for test data. With different tries we acquired different results. For example we tried different numbers of layers and different numbers of neurons in our model and we acquired 98% accuracy on our training but only around 50% accuracy on our test which means that our model was overfit. Overfit means that our training was so specific that the model not only learned but also memorized the training images and it didn't recognize the test data. We also experienced that the number of epochs did not help our model after some point. We started the training with 50 epochs but we saw that after around 25 epochs the model stopped improving and so we reduced the number of epochs.\
\
With this project we learned how image processing can help our daily life and solve our problems. We learned how CNN model works, what it includes, what blocks and layers do. We also gain experience with different libraries that help us get the best results. We now know which steps to take when creating a model and training it with different variables.

# How To Run The Code
As we mentioned before our project includes 2 python files, a big dataset including 2 folders named train and test (each including 7 classes for each emotion that we are using), one .h5 file that saves the weights of the trained model and one .xml file that we use for detecting the faces from the images. In order to train the model from the start, the file ’Emotion_little_vgg.h5’ is not needed, our training code saves it itself. However, in order to use the pretrained model, it is necessary. \
\
To train our model the dataset is needed as well. After downloading the dataset, opening the classification.py file and running it will start the training process. \
To run the classification.py file the command below should be run in the right directory.
```
python classification.py
```
In the beginning and after the training, program will print out and display some results and graphs. After the training is over and the results are shown, classification.py file can be closed. \
\
Next step is to open and run the facial_expression.py file. \
to run this file the command below should be run in the right directory.
```
python facial_expression.py
```
When running this file, a pop-up window will appear and with this window laptop camera will open. Our program is designed to access the camera and capture image from the live footage. Once this window is open program will predict the emotion from the expression dynamically. The window can be closed by pressing “q”.
 

