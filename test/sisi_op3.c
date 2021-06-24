// tvm target: c -keys=cpu -link-params=0
#include llvm/ADT/ArrayRef.h
#include dtu/factor/factor.h
#include dtu/runtime/task_context.h
#include gtets/gtest.h
void* __tvm_module_ctx = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_contrib_dense_pack_add_nn_relu(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* T_relu = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)128, 2, 32);
    if (compute == NULL) {
      return -1;
    }
    void* compute_global = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)64, 2, 32);
    if (compute_global == NULL) {
      return -1;
    }
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 2; ++y_inner_outer_x_inner_outer_fused) {
      ((float16*)((float*)compute_global + (0)))[0] = ((float16)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
      for (int32_t k_outer = 0; k_outer < 784; ++k_outer) {
        ((float16*)((float*)compute_global + (0)))[0] = (((float16*)((float*)compute_global + (0)))[0] + (((float16)(((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)], ((float*)placeholder)[(k_outer)])) * ((float16*)((float*)placeholder1 + ((((ax1_outer_ax0_outer_fused * 25088) + (y_inner_outer_x_inner_outer_fused * 12544)) + (k_outer * 16)))))[0]));
      }
      ((float16*)((float*)compute + ((y_inner_outer_x_inner_outer_fused * 16))))[0] = ((float16*)((float*)compute_global + (0)))[0];
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      float16 _1 = ((float16*)((float*)compute + ((ax1_inner_outer * 16))))[0] + ((float16*)((float*)placeholder2 + (((ax1_outer_ax0_outer_fused * 32) + (ax1_inner_outer * 16)))))[0];
      float16 _2 = (float16)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
      ((float16*)((float*)T_relu + (((ax1_outer_ax0_outer_fused * 32) + (ax1_inner_outer * 16)))))[0] = ((_1) > (_2) ? (_1) : (_2));
    }
    if (TVMBackendFreeWorkspace(1, dev_id, compute_global) != 0) {
      return -1;
    }
    if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {
      return -1;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_batch_flatten(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* tensor = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t ax1_outer = 0; ax1_outer < 49; ++ax1_outer) {
    ((float16*)((float*)tensor + ((ax1_outer * 16))))[0] = ((float16*)((float*)placeholder + ((ax1_outer * 16))))[0];
  }
  return 0;
}

