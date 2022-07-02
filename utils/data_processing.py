#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

import math
from PIL import Image
import numpy as np
import os


# YOLOv3-608 has been trained with these 80 categories from COCO:
# Lin, Tsung-Yi, et al. "Microsoft COCO: Common Objects in Context."
# European Conference on Computer Vision. Springer, Cham, 2014.

def load_label_categories(label_file_path):
    categories = [line.rstrip('\n') for line in open(label_file_path)]
    return categories

LABEL_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coco_labels.txt')
ALL_CATEGORIES = load_label_categories(LABEL_FILE_PATH)

# Let's make sure that there are 80 classes, as expected for the COCO data set:
CATEGORY_NUM = len(ALL_CATEGORIES)
assert CATEGORY_NUM == 80


class PreprocessYOLO(object):
    """A simple class for loading images with PIL and reshaping them to the specified
    input resolution for YOLOv3-608.
    """

    def __init__(self, yolo_input_resolution):
        """Initialize with the input resolution for YOLOv3, which will stay fixed in this sample.

        Keyword arguments:
        yolo_input_resolution -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.yolo_input_resolution = yolo_input_resolution

    def process(self, input_image_path):
        """Load an image from the specified input path,
        and return it together with a pre-processed version required for feeding it into a
        YOLOv3 network.

        Keyword arguments:
        input_image_path -- string path of the image to be loaded
        """
        # image_raw, image_resized = self._load_and_resize(input_image_path)
        image_raw, image_resized = self._load_letterbox(input_image_path)
        image_preprocessed = self._shuffle_and_normalize(image_resized)
        return image_raw, image_preprocessed
    
    def _load_letterbox(self, input_image_path):
        image_raw = Image.open(input_image_path)
        image_w, image_h = image_raw.size       # 原始图像尺寸
        # Expecting yolo_input_resolution in (height, width) format, adjusting to PIL
        # convention (width, height) in PIL:
        new_resolution = (
            self.yolo_input_resolution[1],
            self.yolo_input_resolution[0])      # 目标图像的尺度
        # scale ratio
        r = min(new_resolution[0] / image_w , new_resolution[1] / image_h)   #

        nw  = int(r * image_w)   # 
        nh  = int(r * image_h)

        image_resized = image_raw.resize((nw, nh), resample=Image.BICUBIC)
        # 生成灰色图像
        new_image = Image.new("RGB", new_resolution, (114, 114, 114))
        # 贴图
        new_image.paste(image_resized, ((new_resolution[0] - nw)//2, (new_resolution[1] - nh)//2))
        # new_image.save("resize_pad.jpg")
        new_image = np.array(new_image, dtype=np.float32, order='C')
        return image_raw, new_image

    def _load_and_resize(self, input_image_path):
        """Load an image from the specified path and resize it to the input resolution.
        Return the input image before resizing as a PIL Image (required for visualization),
        and the resized image as a NumPy float array.

        Keyword arguments:
        input_image_path -- string path of the image to be loaded
        """

        image_raw = Image.open(input_image_path)
        # Expecting yolo_input_resolution in (height, width) format, adjusting to PIL
        # convention (width, height) in PIL:
        new_resolution = (
            self.yolo_input_resolution[1],
            self.yolo_input_resolution[0])
        image_resized = image_raw.resize(
            new_resolution, resample=Image.BICUBIC)
        image_resized = np.array(image_resized, dtype=np.float32, order='C')
        return image_raw, image_resized

    def _shuffle_and_normalize(self, image):
        """Normalize a NumPy array representing an image to the range [0, 1], and
        convert it from HWC format ("channels last") to NCHW format ("channels first"
        with leading batch dimension).

        Keyword arguments:
        image -- image as three-dimensional NumPy float array, in HWC format
        """
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.array(image, dtype=np.float32, order='C')
        return image


class PostprocessYOLO(object):
    """Class for post-processing the three outputs tensors from YOLOv3-608."""

    def __init__(self,
                 obj_threshold,
                 nms_threshold,
                 yolo_input_resolution,
                 num_classes,
                 anchors):
        """Initialize with all values that will be kept when processing several frames.
        Assuming 3 outputs of the network in the case of (large) YOLOv3.

        Keyword arguments:
        yolo_masks -- a list of 3 three-dimensional tuples for the YOLO masks
        yolo_anchors -- a list of 9 two-dimensional tuples for the YOLO anchors
        object_threshold -- threshold for object coverage, float value between 0 and 1
        nms_threshold -- threshold for non-max suppression algorithm,
        float value between 0 and 1
        input_resolution_yolo -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """

        self.object_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.input_resolution_yolo = yolo_input_resolution
        self.na = anchors
        self.no = num_classes + 5 # number of outputs per anchor
        self.nc = num_classes       # number of classes

    def process(self, outputs, resolution_raw):
        """Take the YOLOv3 outputs generated from a TensorRT forward pass, post-process them
        and return a list of bounding boxes for detected object together with their category
        and their confidences in separate lists.

        Keyword arguments:
        outputs -- outputs from a TensorRT engine in NCHW format
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        """
        outputs_reshaped = list()
        for output in outputs:
            outputs_reshaped.append(self._reshape_output(output))

        boxes, categories, confidences = self._process_yolo_output(
            outputs_reshaped, resolution_raw)

        return boxes, categories, confidences

    def _reshape_output(self, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (height,width,3,85) dimensionality after further reshaping.

        Keyword argument:
        output -- an output from a TensorRT engine after inference
        """
        bs, _, ny, nx = output.shape
        output = np.reshape(output, (bs, self.na, self.no, ny, nx))
        
        return np.transpose(output,[0, 1, 3, 4, 2])

    def _process_yolo_output(self, outputs_reshaped, resolution_raw):
        """Take in a list of three reshaped YOLO outputs in (height,width,3,85) shape and return
        return a list of bounding boxes for detected object together with their category and their
        confidences in separate lists.

        Keyword arguments:
        outputs_reshaped -- list of three reshaped YOLO outputs as NumPy arrays
        with shape (bs,anchors,height, widht,85)
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        """

        # E.g. in YOLOv3-608, there are three output tensors, which we associate with their
        # respective masks. Then we iterate through all output-mask pairs and generate candidates
        # for bounding boxes, their corresponding category predictions and their confidences:
        boxes, categories, confidences = list(), list(), list()
        for output in outputs_reshaped:
            box, category, confidence = self._process_feats(output)     # decode

            box, category, confidence = self._filter_boxes(box, category, confidence)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)

        boxes = np.concatenate(boxes)
        categories = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # Using the candidates from the previous (loop) step, we apply the non-max suppression
        # algorithm that clusters adjacent bounding boxes to a single bounding box:
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self._nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return None, None, None

        boxes = np.concatenate(nms_boxes)
        categories = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)
        
        # xywh -> x1y1x2y2
        boxes = self.xywh2xyxy(boxes)
        # rescale 
        boxes = self.rescale(resolution_raw, boxes, self.input_resolution_yolo)

        return boxes, categories, confidences

    def _process_feats(self, output_reshaped):
        """Take in a reshaped YOLO output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.

        Keyword arguments:
        output_reshaped -- reshaped YOLO output as NumPy arrays with shape (height,width,3,85)
        mask -- 2-dimensional tuple with mask specification for this output
        """

        # Two in-line functions required for calculating the bounding box
        # descriptors:
        def sigmoid(value):
            """Return the sigmoid of the input."""
            return 1.0 / (1.0 + math.exp(-value))

        def exponential(value):
            """Return the exponential of the input."""
            return math.exp(value)

        # Vectorized calculation of above two functions:
        sigmoid_v = np.vectorize(sigmoid)
        exponential_v = np.vectorize(exponential)

        bs, na, grid_h, grid_w, no = output_reshaped.shape
        # decode 
        box_xy = output_reshaped[..., :2]                       # [1, 1, 80, 80, 2]
        box_wh = exponential_v(output_reshaped[..., 2:4])       # [1, 1, 80, 80, 2]
        # box_wh = output_reshaped[..., 2:4]       # [1, 1, 80, 80, 2]
        box_confidence = output_reshaped[..., 4]                # [1, 1, 80, 80, 1]
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = output_reshaped[..., 5:]   # [1, 1, 80, 80, 80]
        # generate grid
        col, row = np.meshgrid(range(grid_h), range(grid_w))

        col = col.reshape(bs, 1, grid_h, grid_w, 1).repeat(na, axis=1)
        row = row.reshape(bs, 1, grid_h, grid_w, 1).repeat(na, axis=1)

        grid = np.concatenate((col, row), axis=-1)          # [1, 80, 80, 1, 2]
        stride = [self.input_resolution_yolo[0] / grid_h, self.input_resolution_yolo[1] / grid_w]
        # for y in range(grid_h):
        #     for x in range(grid_w):
        #         for b in range(na):
        #             box_xy[0][b][y][x][0] = (box_xy[0][b][y][x][0] + x) * stride[1]
        #             box_xy[0][b][y][x][1] = (box_xy[0][b][y][x][1] + y) * stride[0]
        #             box_wh[0][b][y][x][0] = math.exp(box_wh[0][b][y][x][0]) * stride[1]
        #             box_wh[0][b][y][x][1] = math.exp(box_wh[0][b][y][x][1]) * stride[0]
        # scale height , width
       

        box_xy += grid
        box_xy *= np.array(stride)
        box_wh *= np.array(stride)

        boxes = np.concatenate((box_xy, box_wh), axis=-1)
        boxes = np.reshape(boxes, (bs, -1, 4))
        box_confidence = np.reshape(box_confidence, (bs, -1, 1))
        box_class_probs = np.reshape(box_class_probs, (bs, -1, self.nc))

        # boxes: centroids, box_confidence: confidence level, box_class_probs:
        # class confidence
        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Take in the unfiltered bounding box descriptors and discard each cell
        whose score is lower than the object threshold set during class initialization.

        Keyword arguments:
        boxes -- bounding box coordinates with shape (height,width,3,4); 4 for
        x,y,height,width coordinates of the boxes
        box_confidences -- bounding box confidences with shape (height,width,3,1); 1 for as
        confidence scalar per element
        box_class_probs -- class probabilities with shape (height,width,3,CATEGORY_NUM)

        """
        box_scores = box_confidences * box_class_probs      # [bs, -1, num_classes]
        box_classes = np.argmax(box_scores, axis=-1)        # [bs, -1]
        box_class_scores = np.max(box_scores, axis=-1)      # [bs, -1]
        pos = np.where(box_class_scores >= self.object_threshold)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).

        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep

    def xywh2xyxy(self, x):
        # Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def rescale(self, ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        # ori_shape: height, width
        # target_shape: height, width
        
        ratio = min(target_shape[0] / ori_shape[0], target_shape[1]/ori_shape[1])
        padding = ( target_shape[1] - ori_shape[1] * ratio) / 2, \
                    (target_shape[0] - ori_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]      # width
        boxes[:, [1, 3]] -= padding[1]      # height
        boxes[:, :4] /= ratio

        boxes[:, 0].clip(0, target_shape[1])  # x_center
        boxes[:, 1].clip(0, target_shape[0])  # y_center
        boxes[:, 2].clip(0, target_shape[1])  # width
        boxes[:, 3].clip(0, target_shape[0])  # height

        return boxes
