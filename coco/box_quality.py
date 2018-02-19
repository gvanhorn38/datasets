"""
Basic bounding box quality control reporting for COCO style datasets.
This tool will print out Visipedia Annotation Tool urls that can be used
to examine the images that have possible problems.
"""

import argparse
import json

import numpy as np

try:
    from object_detection.utils import np_box_ops
except:

    pass

def clamp_boxes_to_image(dataset):
    """ Modifies the box annotations so that they are clamped to the image dimensions.
    Returns:
        The modified dataset.
    """

    image_dict = {image['id'] : image for image in dataset['images']}

    modified_count = 0
    for anno in dataset['annotations']:
        if 'bbox' in anno:

            image = image_dict[anno['image_id']]
            image_w = image['width']
            image_h = image['height']

            x, y, w, h = anno['bbox']
            x2 = x + w
            y2 = y + h

            x = np.clip(x, a_min=0, a_max=image_w)
            x2 = np.clip(x2, a_min=0, a_max=image_w)
            y = np.clip(y, a_min=0, a_max=image_h)
            y2 = np.clip(y2, a_min=0, a_max=image_h)

            clamped_bbox = [x, y, x2 - x, y2 - y]
            if not np.allclose(clamped_bbox, anno['bbox']):
                modified_count += 1

            anno['bbox'] = clamped_bbox

    return dataset, modified_count


def check_for_small_boxes(dataset, percentage_of_image_threshold=0.001):
    """ Unusually small boxes can be a sign of mistakes. These can arise due to annotation
    interfaces that draw boxes on mouse clicks.

    Returns:
        A list of annotations whose bounding boxes have a normalized area less than `percentage_of_image_threshold`
    """

    image_dict = {image['id'] : image for image in dataset['images']}

    small_annos = []
    for anno in dataset['annotations']:
        if 'bbox' in anno:
            image = image_dict[anno['image_id']]
            image_area = float(image['width'] * image['height'])
            x, y, w, h = anno['bbox']
            if ( (w * h) / image_area ) < percentage_of_image_threshold:
                small_annos.append(anno)

    return small_annos

def check_for_big_boxes(dataset, percentage_of_image_threshold=0.95):
    """ Unusually large boxes can be a sign of mistakes.
    Returns:
        A list of annotations whose bounding boxes have a normalized area more than `percentage_of_image_threshold`
    """

    image_dict = {image['id'] : image for image in dataset['images']}

    large_annos = []
    for anno in dataset['annotations']:
        if 'bbox' in anno:
            image = image_dict[anno['image_id']]
            image_area = float(image['width'] * image['height'])
            x, y, w, h = anno['bbox']
            if ( (w * h) / image_area ) > percentage_of_image_threshold:
                large_annos.append(anno)

    return large_annos

def check_for_duplicate_annotations(dataset, iou_threshold=0.9):
    """ Unusually high IOU values between boxes can be a sign of duplicate annotations on the same instance.
    Returns:
        A list of image ids that may contain duplicate annotations
    """

    image_id_to_annos = {image['id'] : [] for image in dataset['images']}
    for anno in dataset['annotations']:
        if 'bbox' in anno:
            x, y, w, h = anno['bbox']
            x2 = x + w
            y2 = y + h
            image_id_to_annos[anno['image_id']].append([y, x, y2, x2])

    image_ids_with_duplicates = []
    for image_id, annos in image_id_to_annos.iteritems():
        if len(annos) > 1:
            boxes = np.array(annos)
            iou_mat = np_box_ops.iou(boxes, boxes)
            np.fill_diagonal(iou_mat, -1)
            if np.any(iou_mat > iou_threshold):
                image_ids_with_duplicates.append(image_id)

    return image_ids_with_duplicates


def parse_args():

    parser = argparse.ArgumentParser(description='Report images that should be reviewed for quality control.')

    parser.add_argument('--dataset', dest='dataset_fp',
                          help='Path to a COCO style dataset.', type=str,
                          required=True)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.dataset_fp) as f:
        dataset = json.load(f)

    print "Clamping boxes to image dimensions"
    dataset, num_modified = clamp_boxes_to_image(dataset)
    print "Clamping modified %d boxes" % (num_modified,)
    print

    print "Searching for small boxes"
    small_annos = check_for_small_boxes(dataset)
    small_image_ids = list(set([anno['image_id'] for anno in small_annos]))
    print "Found %d small boxes across %d images" % (len(small_annos), len(small_image_ids))
    print

    print "Searching for large boxes"
    large_annos = check_for_big_boxes(dataset)
    large_image_ids = list(set([anno['image_id'] for anno in large_annos]))
    print "Found %d large boxes across %d images" % (len(large_annos), len(large_image_ids))
    print

    print "Searching for duplicate boxes"
    dup_image_ids = check_for_duplicate_annotations(dataset)
    print "Found %d images with possible duplicate boxes" % (len(dup_image_ids),)
    print

    print "VAT Edit URLS"
    print
    print "Images with small boxes"
    print "/edit_task/?image_ids=%s" % (",".join(map(str, small_image_ids)),)
    print
    print "Images with large boxes"
    print "/edit_task/?image_ids=%s" % (",".join(map(str, large_image_ids)),)
    print
    print "Images with possible duplicate boxes"
    print "/edit_task/?image_ids=%s" % (",".join(map(str, dup_image_ids)),)
    print

if __name__ == '__main__':
    main()