"""
Build a dataset using the 2014 Train/Val object instance data.

Only the people instances are stored. No crowd label is used. 
"""
import json
import random

import numpy as np

def build(coco_annotation_file, image_path_prefix, 
          category_ids=None,
          bbox_minimum_area=None, 
          store_keypoints=False, expand_boxes_to_include_keypoints=False,
          store_crowds=False,
          remap_category_ids=True,
          single_class=False,
          canonical_image_dim_for_area_computation=800,
          include_empty_images=True):
    """Construct a tfrecords json data structure.
    Args:
        coco_annotation_file: (str) path to the coco annotation file
        image_path_prefix: (str) prefix to use to construct the path to the image file
        category_ids: ([ints]) list of category ids to include. If None, then all categories are included
        bbox_minimum_area: (int) Filter out annotations that have bounding boxes with an area less than this
        store_keypoints: (bool) If True, then keypoints annotations will be stored
        expand_bbox_to_include_keypoints: (bool) If True, then a bounding box will be expanded if a keypoint annotations falls outside of it.
        store_crowds: (bool) If True, then crowd annotations are stored.
        remap_category_ids: (bool) If true, then the provided ids will be remapped to labels in the range [0, # categories]. If false then
            the provided ids will be used as the labels
        single_class: (bool) If true then a `class.label` and a `class.text` field will be provided for all images.
        canonical_image_dim_for_area_computation: (int) When computing the area of an annotation using the bbox area, the box 
            will be scaled for this image size. Set to 0 to use the per image area. 
        include_empty_images: (bool): If true, then images with no annotations will be included in the dataset.
    
    Returns:
        list : A list that can be passed to create_tfrecords,create() 
        dict : A dict mapping coco lables to integers in the range [0:num_classes-1]
    """
    # Load in the coco annotations
    with open(coco_annotation_file)  as f:
        coco_data = json.load(f)
    
    dataset_info = coco_data['info']
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    if category_ids is None:
        category_ids = [category['id'] for category in categories]
    
    if store_keypoints:
        category_id_to_num_parts = {}
        for category in categories:
            if category['id'] in category_ids:
                category_id_to_num_parts[category['id']] = len(category['keypoints'])
    
    image_id_to_image = {image['id'] : image for image in images}

    category_ids_set = set(category_ids)

    category_id_to_category = {category['id'] : category for category in categories}
    if remap_category_ids:
        category_id_to_label = {category_id : label for label, category_id in enumerate(category_ids)}
    else:
        category_id_to_label = {category_id : category_id for category_id in category_ids}


    num_too_small_annos = 0

    # Create the tfrecords json format
    dataset = {}
    for anno in annotations:
        
        category_id = anno['category_id']
        if category_id not in category_ids_set:
            continue
        
        is_crowd = anno['iscrowd'] if 'iscrowd' in anno else False
        if is_crowd and not store_crowds:
            continue
        
        image_id = anno['image_id']
        image = image_id_to_image[image_id]
        image_filename = image['file_name']
        image_width = float(image['width'])
        image_height = float(image['height'])
    
        x1, y1, w, h = anno['bbox']
        x2 = x1 + w
        y2 = y1 + h
    
        # Normalize the bbox coordinates
        bbox_x1 = x1 / image_width
        bbox_x2 = x2 / image_width
        bbox_y1 = y1 / image_height
        bbox_y2 = y2 / image_height

        # Restrict the bounding box to be in the image plane
        bbox_x1 = np.clip(bbox_x1, 0.0, 1.0)
        bbox_x2 = np.clip(bbox_x2, 0.0, 1.0)
        bbox_y1 = np.clip(bbox_y1, 0.0, 1.0)
        bbox_y2 = np.clip(bbox_y2, 0.0, 1.0)

        # Make sure the coordinates have not been flipped
        if bbox_x2 < bbox_x1:
            t = bbox_x1
            bbox_x1 = bbox_x2
            bbox_x2 = t
        if bbox_y2 < bbox_y1:
            t = bbox_y1
            bbox_y1 = bbox_y2
            bbox_y2 = t
        
        parts_x = None
        parts_y = None
        parts_v = None
        if store_keypoints and 'keypoints' in anno:
            
            parts_x = []
            parts_y = []
            parts_v = []

            num_parts = category_id_to_num_parts[category_id]
      
            parts_min_x = 1.
            parts_min_y = 1.
            parts_max_x = 0.
            parts_max_y = 0.
      
            for i in range(num_parts):
                part_index = i*3
                x, y, v = anno['keypoints'][part_index:part_index + 3]
                
                # Normalize the part coordinates
                px = x / image_width
                py = y / image_height

                # Restrict the parts to be in the image plane
                px = np.clip(px, 0.0, 1.0)
                py = np.clip(py, 0.0, 1.0)

                if v > 0:
                    parts_min_x = min(parts_min_x, px)
                    parts_min_y = min(parts_min_y, py)
                    parts_max_x = max(parts_max_x, px)
                    parts_max_y = max(parts_max_y, py)
                
                parts_x.append(px)
                parts_y.append(py)
                parts_v.append(v)

            # Extend the given bounding box to contain the part bounding box
            # We could put some padding here... 
            if expand_boxes_to_include_keypoints:
                bbox_x1 = max(min(parts_min_x, bbox_x1), 0.0)
                bbox_x2 = min(max(parts_max_x, bbox_x2), 1.0)
                bbox_y1 = max(min(parts_min_y, bbox_y1), 0.0)
                bbox_y2 = min(max(parts_max_y, bbox_y2), 1.0)


        # Should we check for bad annotations? 
        if bbox_minimum_area != None:
            bbox_w = bbox_x2 - bbox_x1
            bbox_h = bbox_y2 - bbox_y1
            if (bbox_w * image_width) * (bbox_h * image_height) < bbox_minimum_area:
                num_too_small_annos += 1
                continue
     
        # Is this the first time we are seeing this image id?
        if image_id not in dataset:
            image_path = str("%s/%s" % (image_path_prefix, image_filename))
            dataset[image_id] = { 
                "filename" : image_path,
                "id" : str(image_id),
                "width" : image_width,
                "height" : image_height,
                "object" : { 
                    "bbox" : {
                        "xmin" : [],
                        "xmax" : [],
                        "ymin" : [],
                        "ymax" : [],
                        "score" : [],
                        "label" : [],
                        "text" : [],
                        "conf" : []
                    },
                    "parts" : {
                        "x" : [],
                        "y" : [],
                        "v" : [],
                        "score" : []
                    },
                    "area" : [], # segmentation area
                    "id" : [], # annotation id
                    "count" : 0
                }
            }

            if single_class:
                dataset[image_id]["class"] = {
                    "label" : category_id_to_label[category_id],
                    "text" : category_id_to_category[category_id]['name']
                }
    
        object_instance = dataset[image_id]["object"]
    
        object_instance["bbox"]["xmin"] += [bbox_x1]
        object_instance["bbox"]["xmax"] += [bbox_x2]
        object_instance["bbox"]["ymin"] += [bbox_y1]
        object_instance["bbox"]["ymax"] += [bbox_y2]
        object_instance["bbox"]["score"] += [1]
        object_instance["bbox"]["label"] += [category_id_to_label[category_id]]
        object_instance["bbox"]["text"] += [category_id_to_category[category_id]['name']]
        
        if "area" in anno:
            object_instance["area"] += [anno["area"]]
        else:
            bbox_w = bbox_x2 - bbox_x1
            bbox_h = bbox_y2 - bbox_y1
            iw = image_width
            ih = image_height
            if canonical_image_dim_for_area_computation > 0:
                if image_width > image_height:
                    iw = canonical_image_dim_for_area_computation
                    ih = (canonical_image_dim_for_area_computation / float(image_width)) * image_height
                else:
                    ih = canonical_image_dim_for_area_computation
                    iw = (canonical_image_dim_for_area_computation / float(image_height)) * image_width
            object_instance["area"] += [(bbox_w * iw) * (bbox_h * ih)]
        
        object_instance["id"] += [anno["id"]]
        object_instance["count"] += 1

        if parts_x != None:
            object_instance['parts']['x'] += parts_x
            object_instance['parts']['y'] += parts_y
            object_instance['parts']['v'] += parts_v
            object_instance['parts']['score'] += [1] * num_parts 

    # See if there are any empty images:
    num_empty_images = 0
    for image in images:
        if image['id'] not in dataset:
            num_empty_images += 1

            if include_empty_images:
                
                image_id = image['id']
                image_filename = image['file_name']
                image_width = float(image['width'])
                image_height = float(image['height'])
                image_path = str("%s/%s" % (image_path_prefix, image_filename))

                dataset[image_id] = { 
                "filename" : image_path,
                "id" : str(image_id),
                "width" : image_width,
                "height" : image_height,
                "object" : { 
                    "bbox" : {
                        "xmin" : [],
                        "xmax" : [],
                        "ymin" : [],
                        "ymax" : [],
                        "score" : [],
                        "label" : [],
                        "text" : [],
                        "conf" : []
                    },
                    "parts" : {
                        "x" : [],
                        "y" : [],
                        "v" : [],
                        "score" : []
                    },
                    "area" : [], # segmentation area
                    "id" : [], # annotation id
                    "count" : 0
                }
            }

    dataset = dataset.values()
    max_bboxes = max(image['object']['count'] for image in dataset)
  
    print "Number of images: %d" % (len(dataset),)
    print "Maximum number of bboxes in an image: %d" % (max_bboxes,)
    
    print "Excluded %d annotations due to boxes being too small." % (num_too_small_annos,)
    print "Found %d empty images, and %s them." % (num_empty_images, "included" if include_empty_images else "excluded")

    return dataset, category_id_to_label


