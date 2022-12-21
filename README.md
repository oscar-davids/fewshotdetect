# Few-Shot Object Detection

Humans are able to learn to recognize new objects even from a few examples. In contrast, training deep-learning-based object detectors require huge amounts of annotated data. To avoid the need to acquire and annotate these huge amounts of data, few-shot object detection aims to learn from a few object instances of new categories in the target domain.

We fine-tune TensorFlow RetinaNet architecture on very few examples of a novel class after initializing from a pre-trained COCO checkpoint through a simple web UI(http://localhost:5555/) and deploy the resulting model into our interactive video environment.

# Installation and Runing

 - Linux with Python >= 3.9

 - pip install tensorflow==2.8.0

 - pip install -r requirements.txt

 - python run.py


# Step by Step for training new model and deploy

- prepare a few images and labeling data for training.
  we can use online labeling tools such as [makesense](https://www.makesense.ai/), here we need CSV format label data, so have to export a csv file with "Single CSV file" option.
  ![image](https://user-images.githubusercontent.com/54097108/208912017-7cb15185-22a6-4d98-b58c-970d4fa4c608.png)

- input model name in simple web UI form.
- choose the images used in the labeling step.
- choose label data CSV file got in the labeling step.
- click "Trainig" button and it will take about 5 minutes.
- once finished training we can see the model name in "Select model" combo box. Select the model name in this combo box and click "Deploy" button.

  ![image](https://user-images.githubusercontent.com/54097108/208911022-41369416-46e8-4e2d-8aac-7f07d43cfaf3.png)

if finished all steps then you can push the test video stream to our interactive video server.

# References

 - https://github.com/ucbdrive/few-shot-object-detection
 - https://github.com/tensorflow/models
