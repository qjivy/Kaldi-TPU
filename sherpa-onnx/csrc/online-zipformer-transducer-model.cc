// sherpa-onnx/csrc/online-zipformer-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer-transducer-model.h"

#include <assert.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cvi-utils.h"
#include "cviruntime.h"
#include "onnx-to-cvi.h"
#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_WARNING),
      config_(config),
      sess_opts_(GetSessionOptions(config)),
      allocator_{} {
  { InitEncoder(config.transducer.encoder); }

  { InitDecoder(config.transducer.decoder); }

  { InitJoiner(config.transducer.joiner); }
}

void OnlineZipformerTransducerModel::InitEncoder(
    const std::string &model_path) {
  CVI_NN_RegisterModel(model_path.c_str(), &encoder_sess_);
  GetInputOutPutInfo(encoder_sess_);

  // set meta data
  encoder_dims_ = {384, 384, 384, 384, 384};
  attention_dims_ = {192, 192, 192, 192, 192};
  num_encoder_layers_ = {2, 4, 3, 2, 4};
  cnn_module_kernels_ = {31, 31, 31, 31, 31};
  left_context_len_ = {64, 32, 16, 8, 32};
  T_ = 39;
  decode_chunk_len_ = 32;
}

void OnlineZipformerTransducerModel::InitDecoder(
    const std::string &model_path) {
  CVI_NN_RegisterModel(model_path.c_str(), &decoder_sess_);
  GetInputOutPutInfo(decoder_sess_);

  // set meta data
  vocab_size_ = 6254;
  context_size_ = 2;
}

void OnlineZipformerTransducerModel::InitJoiner(const std::string &model_path) {
  CVI_NN_RegisterModel(model_path.c_str(), &joiner_sess_);
  GetInputOutPutInfo(joiner_sess_);
  // set meta data
  // joiner_dim = 512
}

void OnlineZipformerTransducerModel::ReleaseModels() {
  CVI_NN_CleanupModel(encoder_sess_);
  CVI_NN_CleanupModel(decoder_sess_);
  CVI_NN_CleanupModel(joiner_sess_);
  TPU_LOG_INFO("release CVI models");
}

OnlineZipformerTransducerModel::~OnlineZipformerTransducerModel() {
  ReleaseModels();
}

std::vector<Ort::Value> OnlineZipformerTransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());
  int32_t num_encoders = static_cast<int32_t>(num_encoder_layers_.size());

  std::vector<const Ort::Value *> buf(batch_size);

  std::vector<Ort::Value> ans;
  ans.reserve(states[0].size());

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][i];
    }
    auto v = Cat<int64_t>(allocator_, buf, 1);  // (num_layers, 1)
    ans.push_back(std::move(v));
  }

  // cached_avg
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders + i];
    }
    auto v = Cat(allocator_, buf, 1);  // (num_layers, 1, encoder_dims)
    ans.push_back(std::move(v));
  }

  // cached_key
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 2 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 3 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 4 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_conv1
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 5 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator_, buf, 1);
    ans.push_back(std::move(v));
  }

  // cached_conv2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 6 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator_, buf, 1);
    ans.push_back(std::move(v));
  }

  return ans;
}

std::vector<std::vector<Ort::Value>>
OnlineZipformerTransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  assert(states.size() == num_encoder_layers_.size() * 7);

  int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[1];
  int32_t num_encoders = num_encoder_layers_.size();

  std::vector<std::vector<Ort::Value>> ans;
  ans.resize(batch_size);

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    auto v = Unbind<int64_t>(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_avg
  for (int32_t i = num_encoders; i != 2 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_key
  for (int32_t i = 2 * num_encoders; i != 3 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val
  for (int32_t i = 3 * num_encoders; i != 4 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val2
  for (int32_t i = 4 * num_encoders; i != 5 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv1
  for (int32_t i = 5 * num_encoders; i != 6 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv2
  for (int32_t i = 6 * num_encoders; i != 7 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  return ans;
}

std::vector<Ort::Value> OnlineZipformerTransducerModel::GetEncoderInitStates() {
  // Please see
  // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py#L673
  // for details

  int32_t n = static_cast<int32_t>(encoder_dims_.size());
  std::vector<Ort::Value> cached_len_vec;
  std::vector<Ort::Value> cached_avg_vec;
  std::vector<Ort::Value> cached_key_vec;
  std::vector<Ort::Value> cached_val_vec;
  std::vector<Ort::Value> cached_val2_vec;
  std::vector<Ort::Value> cached_conv1_vec;
  std::vector<Ort::Value> cached_conv2_vec;

  cached_len_vec.reserve(n);
  cached_avg_vec.reserve(n);
  cached_key_vec.reserve(n);
  cached_val_vec.reserve(n);
  cached_val2_vec.reserve(n);
  cached_conv1_vec.reserve(n);
  cached_conv2_vec.reserve(n);

  for (int32_t i = 0; i != n; ++i) {
    {
      std::array<int64_t, 2> s{num_encoder_layers_[i], 1};
      auto v =
          Ort::Value::CreateTensor<int64_t>(allocator_, s.data(), s.size());
      Fill<int64_t>(&v, 0);
      cached_len_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 3> s{num_encoder_layers_[i], 1, encoder_dims_[i]};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_avg_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i]};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_key_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i] / 2};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_val_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i] / 2};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_val2_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], 1, encoder_dims_[i],
                               cnn_module_kernels_[i] - 1};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_conv1_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], 1, encoder_dims_[i],
                               cnn_module_kernels_[i] - 1};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_conv2_vec.push_back(std::move(v));
    }
  }

  std::vector<Ort::Value> ans;
  ans.reserve(n * 7);

  for (auto &v : cached_len_vec) ans.push_back(std::move(v));
  for (auto &v : cached_avg_vec) ans.push_back(std::move(v));
  for (auto &v : cached_key_vec) ans.push_back(std::move(v));
  for (auto &v : cached_val_vec) ans.push_back(std::move(v));
  for (auto &v : cached_val2_vec) ans.push_back(std::move(v));
  for (auto &v : cached_conv1_vec) ans.push_back(std::move(v));
  for (auto &v : cached_conv2_vec) ans.push_back(std::move(v));

  return ans;
}

void dumpmem(const char *name, CVI_TENSOR * tensor){
printf("[DUMP] name: %s type: %s count: %ld memsize: %ld\n",name, formatToStr(tensor->fmt),tensor->count,tensor->mem_size);
for(int i=0;i<tensor->mem_size;i+=16)
{
printf("%p: ",&tensor->sys_mem[i]);
printf(" 0x%02x%02x%02x%02x ",
static_cast<int>(tensor->sys_mem[i+3]),
static_cast<int>(tensor->sys_mem[i+2]),
static_cast<int>(tensor->sys_mem[i+1]),
static_cast<int>(tensor->sys_mem[i+0]));

printf(" 0x%02x%02x%02x%02x ",
static_cast<int>(tensor->sys_mem[i+7]),
static_cast<int>(tensor->sys_mem[i+6]),
static_cast<int>(tensor->sys_mem[i+5]),
static_cast<int>(tensor->sys_mem[i+4]));

printf(" 0x%02x%02x%02x%02x ",
static_cast<int>(tensor->sys_mem[i+11]),
static_cast<int>(tensor->sys_mem[i+10]),
static_cast<int>(tensor->sys_mem[i+9]),
static_cast<int>(tensor->sys_mem[i+8]));

printf(" 0x%02x%02x%02x%02x \n",
static_cast<int>(tensor->sys_mem[i+15]),
static_cast<int>(tensor->sys_mem[i+14]),
static_cast<int>(tensor->sys_mem[i+13]),
static_cast<int>(tensor->sys_mem[i+12]));
}
printf("\n");
}
static int cntenc=0;
std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineZipformerTransducerModel::RunEncoder(Ort::Value features,
                                           std::vector<Ort::Value> states,
                                           Ort::Value /* processed_frames */) {
  cntenc++;
  std::vector<Ort::Value> encoder_inputs;
  encoder_inputs.reserve(1 + states.size());

  encoder_inputs.push_back(std::move(features));
  for (auto &v : states) {
    encoder_inputs.push_back(std::move(v));
  }

  // get input / output tensors
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_num, output_num;
  CVI_NN_GetInputOutputTensors(encoder_sess_, &input_tensors, &input_num,
                               &output_tensors, &output_num);
  printf("[encoder]:[%d] input num: %d, output num: %d\n",cntenc, input_num, output_num);
  dumpmem("Encoder orig input: ",input_tensors);
  LoadOrtValuesToCviTensors(encoder_inputs, input_tensors, input_num);
  dumpmem("Encoder loaded input: ",input_tensors);
  CVI_NN_Forward(encoder_sess_, input_tensors, input_num, output_tensors,
                 output_num);
  dumpmem("Encoder output: ",output_tensors);
  std::vector<Ort::Value> next_states =
      GetOrtValuesFromCviTensors(output_tensors + 1, output_num - 1);

  return {std::move(GetOrtValueFromCviTensor(output_tensors[0])),
          std::move(next_states)};
}

static int cntdec=0;
Ort::Value OnlineZipformerTransducerModel::RunDecoder(
    Ort::Value decoder_input) {
    cntdec++;
  // get input / output tensors
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_num, output_num;
  CVI_NN_GetInputOutputTensors(decoder_sess_, &input_tensors, &input_num,
                               &output_tensors, &output_num);
  printf("[Decoder]:[%d] input num: %d, output num: %d\n",cntdec, input_num, output_num);
  dumpmem("Decoder orig input: ",input_tensors);
  std::vector<Ort::Value> temp = {};
  temp.push_back(std::move(decoder_input));
  LoadOrtValuesToCviTensors(temp, input_tensors, input_num);
  dumpmem("Decoder loaded input: ",input_tensors);
  CVI_NN_Forward(decoder_sess_, input_tensors, input_num, output_tensors,
                 output_num);
  dumpmem("Decoder output: ",output_tensors);
  return std::move(GetOrtValueFromCviTensor(output_tensors[0]));
}

static int cntjoin=0;
Ort::Value OnlineZipformerTransducerModel::RunJoiner(Ort::Value encoder_out,
                                                     Ort::Value decoder_out) {
  cntjoin++;
  // get input / output tensors
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_num, output_num;
  CVI_NN_GetInputOutputTensors(joiner_sess_, &input_tensors, &input_num,
                               &output_tensors, &output_num);
  printf("[Joiner]:[%d] input num: %d, output num: %d\n",cntjoin, input_num, output_num);
  dumpmem("Join orig input: ",input_tensors);
  std::vector<Ort::Value> temp = {};
  temp.push_back(std::move(encoder_out));
  temp.push_back(std::move(decoder_out));
  LoadOrtValuesToCviTensors(temp, input_tensors, input_num);
  dumpmem("Join loaded input: ",input_tensors);
  CVI_NN_Forward(joiner_sess_, input_tensors, input_num, output_tensors,
                 output_num);
  dumpmem("Join output: ",output_tensors);
  return std::move(GetOrtValueFromCviTensor(output_tensors[0]));
}

}  // namespace sherpa_onnx
