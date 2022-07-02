# Deepstream Inference
## copy engine file to deepstream folder
```shell
cp path-to-tensorrt-engine deepstream
e.g cp yolov6s.trt deepstream
```
## deepstream infer
```shell
cd deepstream
export CUDA_VER=11.4 # for dGPU   CUDA_VER=11.4 for Jetson
make -j32
cd ..
# video inference , output file: output_yolov6.mp4
deepstream-app -c deepstream_app_config_yoloV6.txt

```