# BNN-Inference-Based-on-CPP
In this demo,we binarize both activations and weights,and the network is organized as B->A->C->P to avoid information loss in Pooling layer,
Besides,applying BN layer before Activation can decrease the quantization error.During Inference,we fuse the bn and activation layer as a threshold function,
this can decline the computation time,We inference this model on python and c++，and get no precision loss.
To run this demo:
1.python train.py
(than the parameter will save to the current path,ofcourse you can change the path)
2.compile the test.cpp and just run it
(test.cpp will read param from a default path "f:\\nn\\filename.bin",also you can change this path)
