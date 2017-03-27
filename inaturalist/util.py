# the 2017 inat dataset has a similar format to the COCO format

import json
import random

import numpy as np

def build(inat_annotation_file, image_path_prefix, 
          category_ids=None):
    """Construct a tfrecords json data structure.
    Args:
        coco_annotation_file: (str) path to the coco annotation file
        image_path_prefix: (str) prefix to use to construct the path to the image file
        category_ids: ([ints]) list of category ids to include. If None, then all categories are included
    
    Returns:
        list : A list that can be passed to create_tfrecords,create() 
    """
    # Load in the coco annotations
    with open(inat_annotation_file)  as f:
        coco_data = json.load(f)
    
    dataset_info = coco_data['info']
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    if category_ids is None:
        category_ids = [category['id'] for category in categories]
    
    image_id_to_image = {image['id'] : image for image in images}

    category_ids_set = set(category_ids)

    category_id_to_category = {category['id'] : category for category in categories}

    # Create the tfrecords json format
    dataset = {}
    for anno in annotations:
        
        category_id = anno['category_id']
        if category_id not in category_ids_set:
            continue
        
        category = category_id_to_category[category_id]

        image_id = anno['image_id']
        image = image_id_to_image[image_id]
        image_filename = image['file_name']
        image_width = float(image['width'])
        image_height = float(image['height'])
        
        image_path = str("%s/%s" % (image_path_prefix, image_filename))
        dataset[image_id] = { 
            "filename" : image_path,
            "id" : str(image_id),
            "width" : image_width,
            "height" : image_height,
            "class" : {
                "label" : category['id'],
                "text" : category['name']
            },
            "object" : { 
                "bbox" : {
                    "xmin" : [0],
                    "xmax" : [1],
                    "ymin" : [0],
                    "ymax" : [1],
                    "score" : [1],
                    "label" : [category['id']],
                    "text" : [category['name']],
                    "conf" : [1]
                },
                "id" : [anno['id']], # annotation id
                "count" : 1
            }
        }

    dataset = dataset.values()
  
    print "Number of images: %d" % (len(dataset),)
  
    return dataset