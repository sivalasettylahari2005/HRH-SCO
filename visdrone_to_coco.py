import os
import cv2
import json
from tqdm import tqdm

def convert_visdrone_to_coco(data_dir, output_file):
    images_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    
    categories = [
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "bicycle"},
        {"id": 4, "name": "car"},
        {"id": 5, "name": "van"},
        {"id": 6, "name": "truck"},
        {"id": 7, "name": "tricycle"},
        {"id": 8, "name": "awning-tricycle"},
        {"id": 9, "name": "bus"},
        {"id": 10, "name": "motor"}
    ]
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    ann_id = 1
    image_list = sorted(os.listdir(images_dir))
    
    for i, img_name in enumerate(tqdm(image_list)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width, _ = img.shape
        
        image_id = img_name.split('.')[0]
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        ann_path = os.path.join(annotations_dir, image_id + '.txt')
        if not os.path.exists(ann_path):
            continue
            
        with open(ann_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                
                # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                try:
                    x, y, w, h, score, cat, trunc, occ = map(int, parts[:8])
                except ValueError:
                    continue
                
                if cat < 1 or cat > 10:
                    continue
                
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
                
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)
    print(f"Finished. Saved to {output_file}")

if __name__ == "__main__":
    # Convert Train
    convert_visdrone_to_coco(
        'data/Visdrone2019/unzipped_train/VisDrone2019-DET-train', 
        'data/Visdrone2019/train_coco.json'
    )
    # Convert Val (to ensure consistency)
    convert_visdrone_to_coco(
        'data/Visdrone2019/unzipped_val/VisDrone2019-DET-val', 
        'data/Visdrone2019/val_coco.json'
    )
