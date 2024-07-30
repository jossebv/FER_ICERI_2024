# Face Emotion Recognition (FER) - ICERI 2024

This project consists on the creation of a system capable of recognizing emotions by detecting the face of a person. The system works by extracting the landmarks of the subject and classifying them into different emotions by using a model previously trained.

## Installation

For installing the necessary dependencies in your local machine run:

```console
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Directory

The directory contains the following:

-   **Training of models on RAVDESS**. The subdirectories `src.data_loader`, `src.data_utils`, `src.models_definitions` and `src.training` contains the necessary for training a system from scratch using the RAVDESS dataset. The dataset is not included on the directory because of its size.

-   **Record a custom dataset**. The dataset can be recorded by running the script `src.record_dataset`. In the script you can configure the classes to be recorded, as well as the name of the dataset and the number of instances to record. The recording process is not destructive, which means that the new images recorded will be added to the existing ones.

    For interacting with the system camera, the script initializes a object using cv2. At the moment, it is only available for using with `CVCamera` class. This class allows for two parameters, the resolution and the index of the camera to use. This index depends on the system that is being used and the number of cameras connected. For the default one set it to 0. Future implementations will make also available to use the Raspberry Pi camera.

-   **Training a system over the custom dataset**. Once the dataset has been recorded, the `src.notebooks.FER_2024.ipynb` can be used to train and analyse a system using that dataset. It contains all the necessary for the process. We recommend running it on collab as it will use the GPU to speed up the process.

-   **Run the demo**. Finally, once you trained your custom model, run the `src.run_demo.py` script. This file will run the model you just trained to detect the emotion you are doing. The file must be first configured by adding the desired classes to train and the path to the model you just trained. Remember that you must also check that the program is preprocessing the landmarks as same as you did when training the model.

## Demo

The current version of the demo allows for the recognition of 4 different emotions: ["Angry", "Happy", "Sad", "Surprise"]. To recognise them, the default model is a CNN pretrained on RAVDESS and finetuned with the images that can be found on `data/my_faces_dataset`. To run the demo execute the following command from the root of the directory:

```console
$ source venv/bin/activate
$ python3 src/run_demo.py
```
