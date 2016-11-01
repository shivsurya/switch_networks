#include <vector>
#include <iostream>

#include "caffe/layers/switch_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Let's move the selector to the last bottom
template <typename Dtype>
void SwitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //Read switch parameter 
  switch_type_ = this->layer_param_.switch_param().switch_type();
  // Check that the dimensions of bottoms are all the same
  DCHECK(switch_type_==1|| switch_type_==2);
  if(switch_type_==1)//switch 0 implementation
 { 
  for (int i = 1; i < bottom.size() - 1; ++i) {
    CHECK_EQ(bottom[i]->num(),  bottom[0]->num());
    CHECK_EQ(bottom[i]->channels(), bottom[0]->channels());
    CHECK_EQ(bottom[i]->height(), bottom[0]->height());
    CHECK_EQ(bottom[i]->width(), bottom[0]->width());
 }
 }
  // Check the selector dimensions
  const int selector_ind = bottom.size() - 1;
  CHECK_EQ(bottom[selector_ind]->num(), bottom[0]->num());
  CHECK_EQ(bottom[selector_ind]->channels(), 1);
  CHECK_EQ(bottom[selector_ind]->height(), 1);
  CHECK_EQ(bottom[selector_ind]->width(), 1);
}

template <typename Dtype>
void SwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize with the first blob.
 if(switch_type_==1)
 { 
 	top[0]->ReshapeLike(*bottom[0]);
	num_elem_ = top[0]->channels() * top[0]->height() * top[0]->width();;
 }
 else
 {
	num_elem_ = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
  	vector<int> top_num(top.size()); // stores the size(num) of top blobs

 	 const int selector_ind = bottom.size() - 1;
  	for (int n = 0; n < bottom[selector_ind]->num(); n++) {
  	  int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));
    	 CHECK_LT(index, top.size()) << "Index exceeds the number of top blobs.";
    	 top_num[index]++;
  	}

  	int bottom_channels = bottom[0]->channels();
  	int bottom_height = bottom[0]->height();
  	int bottom_width = bottom[0]->width();


  	for (int i = 0; i < top.size(); ++i) {
    	CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        	"allow in-place computation.";
    	if (top_num[i] > 0) {
     	 top[i]->Reshape(top_num[i], bottom_channels, bottom_height, bottom_width);
      	CHECK_EQ(num_elem_ * top_num[i], top[i]->count()) << "Layer reshaping failed.";
    	}
    	else
      	top[i]->Reshape(1, bottom_channels, bottom_height, bottom_width);
  	}
 
 }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int selector_ind = bottom.size() - 1;
 if(switch_type_==1)
 { 
  Dtype* top_data = top[0]->mutable_cpu_data();

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));
    DCHECK(floor(index) == index) << "Index should be an integer";
    DCHECK_GE(index, 0) << "Index should be greater than 0";
    DCHECK_LT(index, selector_ind)
        << "Index should be less than " << selector_ind;
    const Dtype* bottom_data = bottom[index]->cpu_data();
    caffe_copy(num_elem_, bottom_data + bottom[index]->offset(n),
          top_data + top[0]->offset(n));
  }
 }
 else
 {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  vector<int> count_top_num(top.size());  // keeps count of the top_num

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));

    Dtype* top_data = top[index]->mutable_cpu_data();
    int top_offset = count_top_num[index];
    count_top_num[index]++;

    caffe_copy(num_elem_, bottom_data + bottom[0]->offset(n),
        top_data + top[index]->offset(top_offset));
  }

 }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int selector_ind = bottom.size() - 1;
 // const int num_elem = top[0]->channels() * top[0]->height() * top[0]->width();
 if(switch_type_==1)
 { 
 const Dtype* top_diff = top[0]->cpu_diff();

 CHECK(!propagate_down[selector_ind]) <<" Layer cannot backpropagate to selector inputs.";

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));
    Dtype* bottom_diff = bottom[index]->mutable_cpu_diff();
    caffe_copy(num_elem_, top_diff+top[0]->offset(n),
        bottom_diff + bottom[index]->offset(n));
  }
 }
 else
 {
  CHECK(!propagate_down[0]) <<" Layer cannot backpropagate to bottom";

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  vector<int> count_top_num(top.size());  // keeps count of the top_num

  const int selector_ind = bottom.size() - 1;
  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));

    const Dtype* top_diff = top[index]->cpu_diff() + top[index]->offset(count_top_num[index]);
    count_top_num[index]++;

    caffe_copy(num_elem_, top_diff, bottom_diff + bottom[0]->offset(n));
  }

 }
}

#ifdef CPU_ONLY
STUB_GPU(SwitchLayer);
#endif

INSTANTIATE_CLASS(SwitchLayer);
REGISTER_LAYER_CLASS(Switch);
}  // namespace caffe
