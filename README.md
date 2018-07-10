# Optical Sketch Recognition

OSR uses a deep convoluted neural network implemented in tensorflow to identify sketches of graphs

## Overview
-ogr.py runs a desktop sketching application that classifys the graph

-ogr_web.py starts a webserver that can be sent an encoded sketch in a POST request and returns the classification

-ogr_trainer.py trains the CNN and saves it to models

-create_corpus.py is a program that produces a set of ~500 labelled examples to train the CNN on

-ogr_cam.py takes a picture of a graph from the webcam and feeds it into the recognition software. It's temperamental!

## Prerequisites
To run the trainer you require _tensorflow_, _Tkinter_ and _numpy_.
To run the image recognition you also require _opencv).
