#ifndef CAFFE_SWITCH_LAYER_HPP_
#define CAFFE_SWITCH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {
/**
 * @brief Takes some inputs Blob%s and a selector and copies the data
 *        based on the value of the selector blob to the top blob.
 *        The selector values should integer and within [0,n-1]
 */
template <typename Dtype>
class SwitchLayer : public Layer<Dtype> {
 public:
  explicit SwitchLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Switch"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MinNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 //type of switch to be implemented. Route a blob to multiple top blobs vs Route one of the bottom blob // to a single top blob
  Dtype switch_type_;
  Dtype num_elem_;
};

}// namespace caffe

#endif  // CAFFE_SWITCH_LAYER_HPP_ 
