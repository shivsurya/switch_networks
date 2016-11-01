%configure caffe
clear all;

if exist('./+caffe', 'dir')
  addpath('..');
end

caffe.set_mode_gpu();
  gpu_id = 1;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
mean_image=zeros(1,1,3);
res=caffe.io.read_mean('/home/babu/caffe/matlab/+caffe/ResNet_mean.binaryproto');
load('/home/babu/Documents/mean_image.mat');

for i=1:3
   mean_image(1,1,i)=mean2(res(:,:,4-i));
end