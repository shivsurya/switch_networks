#!/usr/bin/env sh

/home/babu/caffe/build/tools/caffe train -gpu 0 -solver /home/babu/caffe/examples/mnist/lenet_multistep_solver.prototxt | tee /home/babu/caffe/examples/mnist/mnist_run_multistep.log
