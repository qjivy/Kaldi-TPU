#include "sherpa-onnx/csrc/onnx-to-cvi.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

#include "cvi-utils.h"

Ort::Value GetOrtValueFromCviTensor(CVI_TENSOR &cvi_tensor) {
  // 打印数据类型
  TPU_LOG_INFO("CVI_TENSOR datatype: %s\n", formatToStr(cvi_tensor.fmt));
  // 根据CVI_TENSOR的数据类型确定Ort的数据类型
  ONNXTensorElementDataType type;
  switch (cvi_tensor.fmt) {
    case CVI_FMT_FP32:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      break;
    // case CVI_FMT_INT32:
    //   type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    //   break;
    // case CVI_FMT_UINT32:
    //   type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    //   break;
    // case CVI_FMT_BF16:
    //   type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    //   break;
    // case CVI_FMT_INT16:
    //   type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    //   break;
    // case CVI_FMT_UINT16:
    //   type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    //   break;
    // case CVI_FMT_INT8:
    //   type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    //   break;
    // case CVI_FMT_UINT8:
    //   type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    //   break;
    default:
      throw std::runtime_error("Unsupported data type for conversion.");
  }
  Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  //   const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const
  //   int64_t* shape, size_t shape_len, ONNXTensorElementDataType type
  int64_t *dim64 = new int64_t[CVI_DIM_MAX];
  for (int i = 0; i < cvi_tensor.shape.dim_size; ++i) {
    dim64[i] = static_cast<int64_t>(cvi_tensor.shape.dim[i]);
  }
  // todo: check data type cast
  auto ort_value = Ort::Value::CreateTensor(
      memory_info, (void *)CVI_NN_TensorPtr(&cvi_tensor), cvi_tensor.mem_size, dim64,
      cvi_tensor.shape.dim_size, type);
  return ort_value;
}

void memcheck1(const char *name, const void *ptr,size_t n){

printf("[CHECKMEMCPY] %s ptr: %p size: %ld\n",name,ptr,n);
  const unsigned char *p = (const unsigned char *)ptr;
    size_t i;

    for (i = 0; i < n; i+=16) {
printf("%p: ",&p[i]);
printf(" 0x%02x%02x%02x%02x ", p[i+3], p[i+2], p[i+1], p[i+0]);
printf(" 0x%02x%02x%02x%02x ", p[i+7], p[i+6], p[i+5], p[i+4]);
printf(" 0x%02x%02x%02x%02x ", p[i+11], p[i+10], p[i+9], p[i+8]);
printf(" 0x%02x%02x%02x%02x \n", p[i+15], p[i+14], p[i+13], p[i+12]);
}

}



void memcheck(const char *name, const void *ptr,size_t n){
/*
printf("[CHECKMEMCPY] %s ptr: %p size: %ld\n",name,ptr,n);
  const unsigned char *p = (const unsigned char *)ptr;
    size_t i;

    for (i = 0; i < n; i+=16) {
printf("%p: ",&p[i]);
printf(" 0x%02x%02x%02x%02x ", p[i+3], p[i+2], p[i+1], p[i+0]);
printf(" 0x%02x%02x%02x%02x ", p[i+7], p[i+6], p[i+5], p[i+4]);
printf(" 0x%02x%02x%02x%02x ", p[i+11], p[i+10], p[i+9], p[i+8]);
printf(" 0x%02x%02x%02x%02x \n", p[i+15], p[i+14], p[i+13], p[i+12]);
}
*/
}

void ConvertOrtValueToCviTensor(Ort::Value &ort_value, CVI_TENSOR &cvi_tensor) {
  // assert ort_value中数据的数量和cvi_tensor的数据的数量一致
  TPU_LOG_INFO("ort_value count: %ld, cvi_tensor count: %ld\n", ort_value.GetTensorTypeAndShapeInfo().GetElementCount(), cvi_tensor.count);
  assert(cvi_tensor.count == ort_value.GetTensorTypeAndShapeInfo().GetElementCount());
  auto type = ort_value.GetTensorTypeAndShapeInfo().GetElementType();
  if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && cvi_tensor.fmt == CVI_FMT_FP32) {
    printf("in ConvertOrtValueToCviTensor copyI64ToFp32\n");
    copyI64ToFp32((int64_t*)(ort_value.GetTensorMutableRawData()), (float *)CVI_NN_TensorPtr(&cvi_tensor), cvi_tensor.count);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && cvi_tensor.fmt == CVI_FMT_UINT16) {
    printf("in ConvertOrtValueToCviTensor copyI64ToU16\n");
    copyI64ToU16((int64_t*)(ort_value.GetTensorMutableRawData()), (uint16_t *)CVI_NN_TensorPtr(&cvi_tensor), cvi_tensor.count);
  } else {
    printf("in ConvertOrtValueToCviTensor memcpy, src:%p, dst:%p size:%lld\n",ort_value.GetTensorMutableRawData(),CVI_NN_TensorPtr(&cvi_tensor),cvi_tensor.mem_size);
    if (fmt_size_map.at(cvi_tensor.fmt) == ortvalue_size_map.at(type)) {
      memcpy(CVI_NN_TensorPtr(&cvi_tensor), ort_value.GetTensorMutableRawData(), cvi_tensor.mem_size);
      memcheck("src: ",ort_value.GetTensorMutableRawData(),cvi_tensor.mem_size);
      memcheck("dst: ",CVI_NN_TensorPtr(&cvi_tensor),cvi_tensor.mem_size);
    } else {
      TPU_LOG_INFO("ort_value element size: %d, cvi_tensor element size: %ld\n", ortvalue_size_map.at(type), cvi_tensor.mem_size);
      throw std::runtime_error("Unsupported data type for conversion.");
    }
  }
}

void DUMPCviTensors(CVI_TENSOR *target_cvi_tensors, const int &num) {
  int idx = 0;
  //for (auto &ort_value : ort_value_list) {
  for (idx=0;idx<num;idx++) {
    auto &cvi_tensor = target_cvi_tensors[idx];
    memcheck1("input: ",CVI_NN_TensorPtr(&cvi_tensor),cvi_tensor.mem_size);
  }
}


void LoadOrtValuesToCviTensors(std::vector<Ort::Value> &ort_value_list, CVI_TENSOR *target_cvi_tensors, const int &num) {
  assert(num == (int)(ort_value_list.size()));
  int idx = 0;
  for (auto &ort_value : ort_value_list) {
  /*
    printf("LoadOrtValuesToCviTensors ort%d\n Print1D:\n",idx);
    sherpa_onnx::Print1D(&ort_value); 
    printf("Print2D\n");
    sherpa_onnx::Print2D(&ort_value); 
    printf("Print3D\n");
    sherpa_onnx::Print3D(&ort_value); 
    */
    auto &cvi_tensor = target_cvi_tensors[idx];
    ConvertOrtValueToCviTensor(ort_value, cvi_tensor);
    idx += 1;
  }
}

std::vector<Ort::Value> GetOrtValuesFromCviTensors(CVI_TENSOR *cvi_tensors, const int num) {
  std::vector<Ort::Value> target_ort_value_list;
  target_ort_value_list.reserve(num);
  for (int i = 0; i < num; i++) {
    auto &cvi_tensor = cvi_tensors[i];
    target_ort_value_list.push_back(GetOrtValueFromCviTensor(cvi_tensor));
  }
  return target_ort_value_list;
}
