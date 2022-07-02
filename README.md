 # INTRODUCTION
 TensorRT Inference for Yolov6.
  
 # Platform
 - NVIDIA T4
 - tensorrt8.0
 - deepstream6.0
 
 # TensorRT Inference Command

 ## FP32 Inference
```shell
python deploy/TENSORRT/onnx-tensorrt.py --image_path path to Inference Image Folder(only support folder) 
--result_path path to Inference Result
--onnx path to Yolov6 onnx file 
--engine path to generate Yolov6 TensorRT engine 
```
 ## FP16 Inference
```shell
python deploy/TENSORRT/onnx-tensorrt.py --image_path path to Inference Image Folder(only support folder) 
--result_path path to Inference Result
--onnx path to Yolov6 onnx file 
--engine path to generate Yolov6 TensorRT engine 
--half
```
 ## INT8 Inference
 ```shell
python deploy/TENSORRT/onnx-tensorrt.py --image_path path to Inference Image Folder(only support folder) 
--result_path path to Inference Result
--onnx path to Yolov6 onnx file 
--engine path to generate Yolov6 TensorRT engine 
--int8
```
 
 ## Inference Demo
![image1.jpg](./assert/image1.jpg)
![image2.jpg](./assert/image2.jpg)
![image3.jpg](./assert/image3.jpg)

 # TenosrRT COCO2017 Val Benchmark

 ## FP32 
```shell
python deploy/DEEPSTREAM/tensorrt_dynamic/eval_yolov6.py 
--image_path path to COCO Eval Image Folder(only support folder) 
--annotations path to COCO Annotations --onnx path to Yolov6 onnx file 
--engine path to generate Yolov6 TensorRT engine 
```
 ## FP16 
```shell
python deploy/DEEPSTREAM/tensorrt_dynamic/eval_yolov6.py 
--image_path path to COCO Eval Image Folder(only support folder)
--annotations path to COCO Annotations --onnx path to Yolov6 onnx file 
--engine path to generate Yolov6 TensorRT engine 
--half
```
 ## INT8 
 ```shell
python deploy/DEEPSTREAM/tensorrt_dynamic/eval_yolov6.py 
--image_path path to COCO Eval Image Folder(only support folder)
--annotations path to COCO Annotations --onnx path to Yolov6 onnx file
--engine path to generate Yolov6 TensorRT engine  
--int8 
--calib_data_path path to calibration Image Folder 
--calib_file_path path to calibration table 
```
## Yolov6s Benchmark

| YOLOv6s(640) FP32|     AP              | area        | maxDet                |
| ---------------- | ------------------- | ----------- | --------------------- |    
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.414
|Average Precision|  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.610
|Average Precision|  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.439
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.575
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.552
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.341
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.612
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.722

--------------
| YOLOv6s(640) FP16|     AP              | area        | maxDet                |
| ---------------- | ------------------- | ----------- | --------------------- |    
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.413 |
|Average Precision|  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.610|
|Average Precision| (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.439|
|Average Precision| (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224|
|Average Precision| (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458|
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.574|
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333|
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522|
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.552|
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.342|
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.612|
|Average Recall|     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.722|

-------
| YOLOv6s(640) INT8|     AP              | area        | maxDet                |
| ---------------- | ------------------- | ----------- | --------------------- |   
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.318|
 |Average Precision|  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.492|
 |Average Precision|  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.337|
 |Average Precision|  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163|
 |Average Precision|  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.359
 |Average Precision|  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.450|
 |Average Recall|     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278|
 |Average Recall|     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.441|
 |Average Recall|     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.468|
 |Average Recall|     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.266|
 |Average Recall|     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.522|
 |Average Recall|     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.619|


 # Related Repository
 - https://github.com/Linaom1214/tensorrt-python
 - https://github.com/spacewalk01/tensorrt-yolov6
 - https://github.com/DataXujing/YOLOv6
 - https://github.com/xzacrhhh/yolov6-trt
