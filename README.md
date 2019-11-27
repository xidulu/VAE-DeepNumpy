# VAE-DeepNumpy
Implementation of Variational Autoencoder with DeepNumpy

`MXNET_USE_FUSION=0 python np_vae.py` 

No idea why `MXNET_USE_FUSION=0` is necessary,
simply following the back trace:
```
Traceback (most recent call last):
  File "np_vae.py", line 120, in <module>
    train(net, n_epoch, print_period, train_set, test_set)
  File "np_vae.py", line 103, in train
    epoch, float(epoch_loss), float(epoch_val_loss)))
  File "/home/ubuntu/mxnet_master_develop/python/mxnet/numpy/multiarray.py", line 797, in __float__
    return float(self.item())
  File "/home/ubuntu/mxnet_master_develop/python/mxnet/numpy/multiarray.py", line 836, in item
    return self.asnumpy().item(*args)
  File "/home/ubuntu/mxnet_master_develop/python/mxnet/ndarray/ndarray.py", line 2552, in asnumpy
    ctypes.c_size_t(data.size)))
  File "/home/ubuntu/mxnet_master_develop/python/mxnet/base.py", line 278, in check_call
    raise MXNetError(py_str(_LIB.MXGetLastError()))
mxnet.base.MXNetError: [11:55:41] /home/ubuntu/mxnet_master_develop/src/operator/fusion/fused_op.cu:605: Check failed: compileResult == NVRTC_SUCCESS (6 vs. 0) : NVRTC Compilation failed. Please set environment variable MXNET_USE_FUSION to 0.
_FusedOpOutHelper__backward_log__FusedOpOutHelper__copy__FusedOpOutHelper_negative__FusedOpOutHelper__backward_log__FusedOpOutHelper__copy_add_n__FusedOpOutHelper__backward_Activation_kernel.cu(782): error: launch_bounds attribute is not allowed on __device__ functions
```
