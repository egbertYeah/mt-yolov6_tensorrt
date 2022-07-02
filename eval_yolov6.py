"""eval_yolo.py
This script is for evaluating mAP (accuracy) of YOLO models.
"""


import os
import sys
import json
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorrt as trt
from tqdm import tqdm
# Utility functions
import utils.inference as inference_utils  # TRT/TF inference wrappers
import utils.data_processing as data_utils
import ctypes
from utils.yolo_classes import yolo_cls_to_ssd



HOME = os.environ['HOME']
VAL_IMGS_DIR = HOME + '/data/coco/images/val2017'
VAL_ANNOTATIONS = HOME + '/data/coco/annotations/instances_val2017.json'


def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of YOLO model'
    parser = argparse.ArgumentParser(description=desc)
    # parser.add_argument('--sofile', type=str, default='../nvdsinfer_custom_impl_Yolov6/libnvdsinfer_custom_impl_Yolov6.so', 
    #                         help='dynamic shared object file path')
    parser.add_argument('--onnx', type=str, default='../yolov6s.onnx', help='Yolov6 onnx file path')
    parser.add_argument('--image_path', type=str, default=None, help='Yolov6 Inference Image path')
    parser.add_argument('--annotations', type=str, default=None, help='Yolov6 Inference Image Result path')
    parser.add_argument('--engine', type=str, default='../yolov6s.trt', help='Yolov6 tensorrt engine file path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true',default=False, help='FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='INT8 precision export')
    parser.add_argument('--calib_data_path', type=str, default=None, help='Yolov6 Inference Image path')
    parser.add_argument('--calib_file_path', type=str, default=None, help='Yolov6 Inference Image Result path')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold for inference.')
    parser.add_argument('--num_classes', type=int, default=80, help='dataset class number.')
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.image_path):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.isfile(args.annotations):
        sys.exit('%s is not a valid file' % args.annotations)

def main():
    args = parse_args()
    check_args(args)

    results_file = 'results_yolov6.json'

    # add tensorrt plugin
    # ctypes.cdll.LoadLibrary(args.sofile)
   # Precision command line argument -> TRT Engine datatype
    TRT_PRECISION_TO_DATATYPE = {
        8: trt.DataType.INT8,
        16: trt.DataType.HALF,
        32: trt.DataType.FLOAT
    }
    # datatype: float 32
    trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[32]

    if(args.half):
        trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[16]
    if(args.int8):
        trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[8]

    # Create a pre-processor object by specifying the required input resolution for YOLOv6
    preprocessor = data_utils.PreprocessYOLO(args.img_size)
    max_batch_size = args.batch_size
    trt_inference_static_wrapper = inference_utils.TRTInference(
        args.engine, args.onnx,
        trt_engine_datatype, max_batch_size, calib_data_path=args.calib_data_path, calib_file_path=args.calib_file_path, input_shape_HW=args.img_size
    )

    output_shapes = [ (1, 85, int(args.img_size[0]/8), int(args.img_size[1]/8)), 
                      (1, 85, int(args.img_size[0]/16), int(args.img_size[1]/16)), 
                      (1, 85, int(args.img_size[0]/32), int(args.img_size[1]/32))]

    # post processing
    postprocessor = data_utils.PostprocessYOLO(args.conf_thres, args.iou_thres,
                                                args.img_size, args.num_classes, 1)

    jpgs = [j for j in os.listdir(args.image_path) if j.endswith('.jpg')]
    
    """Run detection on each jpg and write results to file."""
    results = []
    for jpg in tqdm(jpgs, desc='Processing'):
        input_image_path = os.path.join(args.image_path, jpg)
        # Load an image from the specified input path, and return it together with  a pre-processed version
        image_raw, input_data = preprocessor.process(input_image_path)
        # Store the shape of the original input image in WH format, we will need it for later
        orig_image_w, orig_image_h = image_raw.size
        shape_orig_HW = (orig_image_h, orig_image_w)

        image_id = int(jpg.split('.')[0].split('_')[-1])
        trt_outputs = trt_inference_static_wrapper.infer(input_data, output_shapes)
        boxes, classes, scores = postprocessor.process(trt_outputs, shape_orig_HW)
        # Draw the bounding boxes onto the original input image and save it as a PNG file
        if(boxes is None): continue
        for box, conf, cls in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            obj_w = xmax - xmin
            obj_h = ymax - ymin
            x = float(xmin)
            y = float(ymin)
            w = float(obj_w + 1)
            h = float(obj_h + 1)
            cls = int(yolo_cls_to_ssd[int(cls)])
            
            results.append({'image_id': image_id,
                            'category_id': cls,
                            'bbox': [x, y, w, h],
                            'score': float(conf)})
    with open(results_file, 'w') as f:
        f.write(json.dumps(results, indent=4))
    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(args.annotations)
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()