### SwiDeN

This projects implements a switch layer to existing CAFFE implementation that can direct data in two ways:<br />
1. From one bottom blob to one of the multiple top blobs.<br />
2. Select a blob from multiple bottom blobs and copt to a single top blob.<br />(Use case: Our ACMMM 2016 paper describing Switching Deep Networks([SwiDeN](https://arxiv.org/abs/1607.08764)))<br />
### License

This code is released under the MIT License (Please refer to the LICENSE file for details).

### Dependencies and Installation

2. To install this version of CAFFE , install all the dependencies for CAFFE and then execute the following:
  
   ```bash
   $ git clone https://github.com/val-iisc/swiden.git
   $ cd swiden/caffe/
   $ make all 
   $ make matcaffe #if you have a MATLAB installation and want to link CAFFE to MATLAB
   $ make pycaffe  #if you want to link CAFFE to python
