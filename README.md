**Optical Sketch Recognition**

OSR uses a deep convoluted neural network implemented in tensorflow to identify sketches of graphs

*ogr.py* runs a desktop sketching application that classifys the graph

*ogr_web.py* starts a webserver that can be sent an encoded sketch in a POST request and returns the classification

*ogr_trainer.py* trains the CNN and saves it to models

*create_corpus.py* is a program that produces a set of ~500 labelled examples to train the CNN on
