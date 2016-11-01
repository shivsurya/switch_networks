#include <vector>

#include "caffe/layers/switch_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/im2col_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void SwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int selector_ind = bottom.size() - 1;
  if(switch_type_==1)
  { 
  Dtype* top_data = top[0]->mutable_gpu_data();

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));
    DCHECK(floor(index) == index) << "Index should be an integer";
    DCHECK_GE(index, 0) << "Index should be greater than 0";
    DCHECK_LT(index, selector_ind)
        << "Index should be less than " << selector_ind;
    const Dtype* bottom_data = bottom[index]->gpu_data();
    caffe_copy(num_elem_, bottom_data+bottom[index]->offset(n),
          top_data+top[0]->offset(n));
  }
  }
  else
  {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  vector<int> count_top_num(top.size());  // keeps count of the top_num

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));

    Dtype* top_data = top[index]->mutable_gpu_data();
    int top_offset = count_top_num[index];
    count_top_num[index]++;


    caffe_copy(num_elem_, bottom_data + bottom[0]->offset(n),
        top_data + top[index]->offset(top_offset));
  }
    
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int selector_ind = bottom.size() - 1;
  if(switch_type_==1)
  {
  const Dtype* top_diff = top[0]->gpu_diff();

  CHECK(!propagate_down[selector_ind])<<"Layer cannot backpropagate to selector inputs.";

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));
    Dtype* bottom_diff = bottom[index]->mutable_gpu_diff();
    caffe_copy(num_elem_, top_diff+top[0]->offset(n),
        bottom_diff + bottom[index]->offset(n));
  }
  }
  else
  {
  CHECK(!propagate_down[0])<<"Bottom layer cannot be propagated to.";

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  vector<int> count_top_num(top.size());  // keeps count of the top_num

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));

    const Dtype* top_diff = top[index]->gpu_diff() +
                      top[index]->offset(count_top_num[index]);
    count_top_num[index]++;

    caffe_copy(num_elem_, top_diff, bottom_diff + bottom[0]->offset(n));
  }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SwitchLayer);

}  // namespace caffe
