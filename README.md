Optical Sketch Recogniser

OSR uses a deep convoluted neural network to identify sketches of graphs although inaccurate contain all the features that allow a human to identify a sketched graph.

ogr.py runs a sketching application that classifys the graph

ogr_web.py starts a webserver that can be sent an encoded sketch in a POST request and returns the classification

ogr_trainer.py trains the CNN and saves it to models

create_corpus.py is a program that produces a set of ~500 labelled examples to train the CNN with