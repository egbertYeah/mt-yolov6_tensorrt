from ast import arg
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
# Utility functions
import utils.inference as inference_utils  # TRT/TF inference wrappers
import utils.data_processing as data_utils
import ctypes
import argparse
from PIL import ImageDraw, ImageFont
import os
from utils.yolo_classes import yolo_cls_to_ssd
import cv2

colors = np.random.randint(0, 255, size=(len(data_utils.ALL_CATEGORIES))).tolist()


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    # get a font
    fnt = ImageFont.truetype("./utils/Arial.ttf", 20)
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        xmin, ymin, xmax, ymax = box
        left = max(0, np.floor(xmin ).astype(int))
        top = max(0, np.floor(ymin).astype(int))
        right = min(image_raw.width, np.floor(xmax).astype(int))
        bottom = min(image_raw.height, np.floor(ymax).astype(int))
        bbox_color = colors[int(category)]
        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 25), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color, font=fnt)

    return image_raw

# def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):

#     mask_img = image.copy()
#     det_img = image.copy()

#     img_height, img_width = image.shape[:2]
#     size = min([img_height, img_width]) * 0.0006
#     text_thickness = int(min([img_height, img_width]) * 0.001)

#     # Draw bounding boxes and labels of detections
#     for box, score, class_id in zip(boxes, scores, class_ids):

#         color = colors[class_id]

#         x1, y1, x2, y2 = box.astype(int)

#         # Draw rectangle
#         cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

#         # Draw fill rectangle in mask image
#         cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

#         label = class_names[class_id]
#         caption = f'{label} {int(score*100)}%'
#         (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                                       fontScale=size, thickness=text_thickness)
#         th = int(th * 1.2)

#         cv2.rectangle(det_img, (x1, y1),
#                       (x1 + tw, y1 - th), color, -1)
#         cv2.rectangle(mask_img, (x1, y1),
#                       (x1 + tw, y1 - th), color, -1)
#         cv2.putText(det_img, caption, (x1, y1),
#                     cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

#         cv2.putText(mask_img, caption, (x1, y1),
#                     cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

#     return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sofile', type=str, default='../nvdsinfer_custom_impl_Yolov6/libnvdsinfer_custom_impl_Yolov6.so', 
    #                         help='dynamic shared object file path')
    parser.add_argument('--onnx', type=str, default='../yolov6s.onnx', help='Yolov6 onnx file path')
    parser.add_argument('--image_path', type=str, default=None, help='Yolov6 Inference Image path')
    parser.add_argument('--result_path', type=str, default=None, help='Yolov6 Inference Image Result path')
    parser.add_argument('--engine', type=str, default='../yolov6s.trt', help='Yolov6 tensorrt engine file path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='INT8 precision export')
    parser.add_argument('--calib_data_path', type=str, default=None, help='Yolov6 Inference Image path')
    parser.add_argument('--calib_file_path', type=str, default=None, help='Yolov6 Inference Image Result path')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold for inference.')
    parser.add_argument('--num_classes', type=int, default=80, help='dataset class number.')
    args = parser.parse_args()
    print(args)
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
    image_file_list = os.listdir(args.image_path)
    image_file_list.sort(key=lambda name:name)
    for image_path in image_file_list:
        input_image_path = os.path.join(args.image_path, image_path)
        # Load an image from the specified input path, and return it together with  a pre-processed version
        image_raw, input_data = preprocessor.process(input_image_path)
        # Store the shape of the original input image in WH format, we will need it for later
        orig_image_w, orig_image_h = image_raw.size
        shape_orig_HW = (orig_image_h, orig_image_w)

        trt_outputs = trt_inference_static_wrapper.infer(input_data, output_shapes)
        
        # print("output layer1 : {}".format(trt_outputs[0].shape))
        # print("output layer2 : {}".format(trt_outputs[1].shape))
        # print("output layer3 : {}".format(trt_outputs[2].shape))

        boxes, classes, scores = postprocessor.process(trt_outputs, shape_orig_HW)
        # Draw the bounding boxes onto the original input image and save it as a PNG file
        if(boxes is None): continue

        obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, data_utils.ALL_CATEGORIES)
        output_image_path = os.path.join(args.result_path, image_path)
        obj_detected_img.save(output_image_path)
        print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
       