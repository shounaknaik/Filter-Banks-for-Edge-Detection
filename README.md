1) To run the Phase 1 code simply run:

`python3 Wrapper.py`

This will run the Pb-Lite algorithm on all the 10 images and will also create appropriate images.

2) To run the Phase 2 code:

i) Install torchsummary package by `pip install torchsummary`. I have used this package to summarize the model architecture and parameters.
ii) Run `python3 Train.py --NumEpochs=30 --MiniBatchSize=200`. This will run the training process.
iii) Run `python3 Test.py`. All these commands would create all images shown in the figure.

