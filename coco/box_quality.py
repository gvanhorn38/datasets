"""
Basic bounding box quality control reporting for COCO style datasets.
This tool will print out Visipedia Annotation Tool urls that can be used
to examine the images that have possible problems.

The object_detection code base needs to be on your PYTHONPATH. This can
be found here: https://github.com/tensorflow/models/tree/master/research/object_detection

"""

import argparse
import json

import numpy as np

try:
    from object_detection.utils import np_box_list
    from object_detection.utils import np_box_list_ops
    from object_detection.utils import np_box_ops
except:
    print("WARNING: Failed to import `object_detection`")
    np_box_ops=None
    pass

def check_for_annotations_without_boxes(dataset, remove_boxes=False):
    """ Check for annotations that do not have the `bbox` property.
    Returns:
        A list of annotations without bounding boxes.
    """

    annos_without_box = [anno for anno in dataset['annotations'] if 'bbox' not in anno]

    if remove_boxes:
        dataset['annotations'] = [anno for anno in dataset['annotations'] if 'bbox' in anno]

    return annos_without_box

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


def check_for_small_boxes(dataset, percentage_of_image_threshold=0.0001, remove_boxes=False):
    """ Unusually small boxes can be a sign of mistakes. These can arise due to annotation
    interfaces that draw boxes on mouse clicks.

    Returns:
        A list of annotations whose bounding boxes have a normalized area less than `percentage_of_image_threshold`
    """

    image_dict = {image['id'] : image for image in dataset['images']}

    small_annos = []
    g2g_annos = []
    for anno in dataset['annotations']:
        if 'bbox' in anno:
            image = image_dict[anno['image_id']]
            image_area = float(image['width'] * image['height'])
            x, y, w, h = anno['bbox']
            if ( (w * h) / image_area ) < percentage_of_image_threshold:
                small_annos.append(anno)
            else:
                g2g_annos.append(anno)
        else:
            g2g_annos.append(anno)

    if remove_boxes:
        dataset['annotations'] = g2g_annos

    return small_annos

def check_for_big_boxes(dataset, percentage_of_image_threshold=0.95, remove_boxes=False):
    """ Unusually large boxes can be a sign of mistakes.
    Returns:
        A list of annotations whose bounding boxes have a normalized area more than `percentage_of_image_threshold`
    """

    image_dict = {image['id'] : image for image in dataset['images']}

    large_annos = []
    g2g_annos = []
    for anno in dataset['annotations']:
        if 'bbox' in anno:
            image = image_dict[anno['image_id']]
            image_area = float(image['width'] * image['height'])
            x, y, w, h = anno['bbox']
            if ( (w * h) / image_area ) > percentage_of_image_threshold:
                large_annos.append(anno)
            else:
                g2g_annos.append(anno)
        else:
            g2g_annos.append(anno)

    if remove_boxes:
        dataset['annotations'] = g2g_annos

    return large_annos

def check_for_duplicate_annotations(dataset, iou_threshold=0.9, remove_boxes=False):
    """ Unusually high IOU values between boxes can be a sign of duplicate annotations on the same instance.
    Returns:
        A list of image ids that may contain duplicate annotations
    """
    if np_box_ops is None:
        print("WARNING: `np_box_ops` failed to import, can't check for duplicates.")
        return []

    image_id_to_annos = {image['id'] : [] for image in dataset['images']}
    for anno in dataset['annotations']:
        if 'bbox' in anno:
            image_id_to_annos[anno['image_id']].append(anno)

    image_ids_with_duplicates = []
    anno_ids_to_remove = set()
    for image_id, annos in image_id_to_annos.items():
        if len(annos) > 1:

            boxes = []
            ids = []
            for anno in annos:
                x, y, w, h = anno['bbox']
                x2 = x + w
                y2 = y + h
                boxes.append([y, x, y2, x2])
                ids.append(anno['id'])

            boxes = np.array(boxes).astype(np.float)
            scores = np.ones(len(ids), dtype=np.float32)
            ids = np.array(ids)

            try:
                boxlist = np_box_list.BoxList(boxes)
            except:
                print(image_id)
                print(annos)
                print(boxes)
                raise
            boxlist.add_field("ids", ids)
            boxlist.add_field("scores", scores)

            nms = np_box_list_ops.non_max_suppression(boxlist, iou_threshold=iou_threshold)

            selected_ids = nms.get_field("ids").tolist()
            removed_ids = [aid for aid in ids if aid not in selected_ids]
            if len(removed_ids):
                anno_ids_to_remove.update(removed_ids)
                image_ids_with_duplicates.append(image_id)

    if remove_boxes:
        dataset['annotations'] = [anno for anno in dataset['annotations'] if anno['id'] not in anno_ids_to_remove]

    return image_ids_with_duplicates

def check_for_no_boxes(dataset):
    """ Return the ids of the images that do not have any bounding boxes.
    """
    image_ids_with_boxes = set([anno['image_id'] for anno in dataset['annotations'] if 'bbox' in anno])
    image_ids = set([image['id'] for image in dataset['images']])
    image_ids_without_boxes = image_ids.difference(image_ids_with_boxes)

    return list(image_ids_without_boxes)

def parse_args():

    parser = argparse.ArgumentParser(description='Print VAT urls containing images that should be reviewed for quality control.')

    parser.add_argument('--dataset', dest='dataset_fp',
                          help='Path to a COCO style dataset.', type=str,
                          required=True)

    parser.add_argument('--small_threshold', dest='small_threshold',
                        help='Boxes whose normalized area is less than this value will be removed. Set to 0 to turn off.',
                        required=False, type=float, default=0.0001)

    parser.add_argument('--large_threshold', dest='large_threshold',
                        help='Boxes whose normalized area is more than this value will be removed. Set to 1 to turn off.',
                        required=False, type=float, default=0.999)

    parser.add_argument('--iou_threshold', dest='iou_threshold',
                        help='NMS is applied to the boxes with this threshold. Set to 1 to turn off.',
                        required=False, type=float, default=0.9)

    parser.add_argument('--output', dest='output_fp',
                          help='If provided, then the boxes marked for review will be removed from the dataset and the modified dataset will be saved at this file path.', type=str,
                          required=False, default=None)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.dataset_fp) as f:
        dataset = json.load(f)

    if args.output_fp is not None:
        print("Problem boxes will be removed and the new dataset will be saved at:")
        print(args.output_fp)
        print()
        remove_boxes = True

        original_anno_count = len(dataset['annotations'])

    else:
        remove_boxes = False

    print("Checking for annotations without boxes")
    annos_without_boxes = check_for_annotations_without_boxes(dataset, remove_boxes=False)
    image_ids_with_annos_without_boxes = list(set([anno['image_id'] for anno in annos_without_boxes]))
    print("Found %d annotations without boxes across %d images" % (len(annos_without_boxes), len(image_ids_with_annos_without_boxes)))
    print()

    print("Clamping boxes to image dimensions")
    dataset, num_modified = clamp_boxes_to_image(dataset)
    print("Clamping modified %d boxes" % (num_modified,))
    print()

    print("Searching for small boxes whose normalized area is less than %f" % (args.small_threshold,))
    small_annos = check_for_small_boxes(dataset, args.small_threshold, remove_boxes)
    small_image_ids = list(set([anno['image_id'] for anno in small_annos]))
    print("Found %d small boxes across %d images" % (len(small_annos), len(small_image_ids)))
    print()

    print("Searching for large boxes whose normalized area is greater than %f" % (args.large_threshold,))
    large_annos = check_for_big_boxes(dataset, args.large_threshold, remove_boxes)
    large_image_ids = list(set([anno['image_id'] for anno in large_annos]))
    print("Found %d large boxes across %d images" % (len(large_annos), len(large_image_ids)))
    print()

    print("Searching for duplicate boxes whose iou is greater than %f" % (args.iou_threshold,))
    dup_image_ids = check_for_duplicate_annotations(dataset, args.iou_threshold, remove_boxes)
    print("Found %d images with possible duplicate boxes" % (len(dup_image_ids),))
    print()

    print("Searching for images with no boxes")
    image_ids_without_boxes = check_for_no_boxes(dataset)
    print("Found %d images with no boxes" % (len(image_ids_without_boxes),))
    print()

    print("VAT Edit URLS")
    print()
    print("Images with annotations with no boxes")
    print("/edit_task/?image_ids=%s" % (",".join(map(str, image_ids_with_annos_without_boxes)),))
    print()
    print("Images with small boxes")
    print("/edit_task/?image_ids=%s" % (",".join(map(str, small_image_ids)),))
    print()
    print("Images with large boxes")
    print("/edit_task/?image_ids=%s" % (",".join(map(str, large_image_ids)),))
    print()
    print("Images with possible duplicate boxes")
    print("/edit_task/?image_ids=%s" % (",".join(map(str, dup_image_ids)),))
    print()
    print("Images with no boxes")
    print("/edit_task/?image_ids=%s" % (",".join(map(str, image_ids_without_boxes)),))
    print()

    # Save the modified dataset
    if args.output_fp is not None:
        print("Saving the modified dataset")
        mod_anno_count = len(dataset['annotations'])
        print("%d - %d = %d final annotations." % (original_anno_count, original_anno_count - mod_anno_count, mod_anno_count))

        with open(args.output_fp, 'w') as f:
            json.dump(dataset, f)

        if len(image_ids_without_boxes):
            print("WARNING: you have images without bounding boxes in the saved dataset.")

if __name__ == '__main__':
    main()