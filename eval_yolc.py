from mmdet.apis import init_detector
import torch
from inference_YOLC import inference_detector, inference_detector_with_LSM
import numpy as np
import cv2
import mmcv
import os
import json
from mmdet.datasets import build_dataset

from pycocotools.coco import COCO

from models import *
from VisDrone_Dataset import VisDroneDataset


visual = False


def inference(model1, model2, img, img_name, num_cls = 10, saved_crop = 2, crop=True):
    # data = dict(img=img)
    subregion_coord, coarse_result = inference_detector_with_LSM(model1, img)

    img_np = cv2.imread(img)
    img_draw = img_np.copy()
    final_result = []
    cluster_regions = []
    
    for i in range(num_cls):
        final_result.append([])

    if crop:
        areas = []
        for item in subregion_coord:
            x,y,w,h = item
            areas.append(w*h)
        areas = np.array(areas)
        idx = areas.argsort()[::-1]
        if len(idx) > saved_crop:
            idx = idx[:saved_crop]

        subregion_coord = subregion_coord[idx]

        for i in range(len(subregion_coord)):
            x, y, w, h = subregion_coord[i]
            x = int(x)
            y = int(y)

            bboxes = [x, y, x+w, y+h]
            box_scale_ratio = 1.2         # box scale factor

            w_half = (bboxes[2] - bboxes[0]) * 0.5
            h_half = (bboxes[3] - bboxes[1]) * 0.5
            x_center = (bboxes[2] + bboxes[0]) * 0.5
            y_center = (bboxes[3] + bboxes[1]) * 0.5

            w_half *= box_scale_ratio
            h_half *= box_scale_ratio
            w_img, h_img = img_np.shape[1], img_np.shape[0]

            boxes_scaled = [0, 0, 0, 0]
            boxes_scaled[0] = min(max(x_center - w_half, 0), w_img - 1)
            boxes_scaled[2] = min(max(x_center + w_half, 0), w_img - 1)
            boxes_scaled[1] = min(max(y_center - h_half, 0), h_img - 1)
            boxes_scaled[3] = min(max(y_center + h_half, 0), h_img - 1)
            
            cluster_regions.append(boxes_scaled)
            boxes_scaled = [int(i) for i in boxes_scaled]

            img_scale_ratio = 1.5   # image scale factor
            w_new = boxes_scaled[2] - boxes_scaled[0]
            h_new = boxes_scaled[3] - boxes_scaled[1]
            
            if w_new <= 0 or h_new <= 0:
                continue

            img_crop = img_np[boxes_scaled[1]:boxes_scaled[3], boxes_scaled[0]:boxes_scaled[2]]
            img_resize = cv2.resize(img_crop, (int(w_new * img_scale_ratio), int(h_new * img_scale_ratio)))
            result_refine = inference_detector(model2, img_resize)
        
            for j in range(len(result_refine)):
                for item in result_refine[j]:
                    new_item = item.copy()
                    new_item[0:4] = new_item[0:4] / img_scale_ratio
                    new_item[0:2] += boxes_scaled[0:2]
                    new_item[2:4] += boxes_scaled[0:2]
                    final_result[j].append(new_item)

    # fuse coarse and refined results
    for i in range(len(coarse_result)):
        cls_result = coarse_result[i]
        for item in cls_result:
            x1, y1, x2, y2, score = item
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            in_cluster = False
            for boxes_scaled in cluster_regions:
                if boxes_scaled[0] <= x_center <= boxes_scaled[2] and boxes_scaled[1] <= y_center <= boxes_scaled[3]:
                    in_cluster = True
                    break
            if not in_cluster:
                final_result[i].append(item)

    # Convert to numpy arrays
    for i in range(len(final_result)):
        if len(final_result[i]) > 0:
            final_result[i] = np.stack(final_result[i])
        else:
            final_result[i] = np.zeros((0, 5), dtype=np.float32)

    return final_result




if __name__ == '__main__':
    dataset_anno = 'data/Visdrone2019/val_coco.json'
    dataset_root = 'data/Visdrone2019/unzipped_val/VisDrone2019-DET-val/images'

    classes = ('pedestrian', "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")

    config_file1 = './configs/yolc.py'
    config_file2 = './configs/yolc.py'

    checkpoint_file = 'yolc.pth'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model1 = init_detector(config_file1, checkpoint_file, device=device)
    model1.eval()

    model2 = model1
    saved_crop = 2

    # Standard loop to eval all images
    coco = COCO(dataset_anno)
    size = len(list(coco.imgs.keys()))
    results = []
    
    checkpoint_path = 'results_checkpoint.pkl'
    import pickle
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Resuming from checkpoint: {len(results)} images already processed.")
    
    prog_bar = mmcv.ProgressBar(size)
    for _ in range(len(results)):
        prog_bar.update()

    import time
    start_time = time.time()
    ids = coco.getImgIds()
    
    for i in range(len(results), size):
        img_id = ids[i]
        img_info = coco.loadImgs([img_id])[0]
        img_name = img_info['file_name']
        img = os.path.join(dataset_root, img_name)
        
        final_result = inference(model1, model2, img, img_name, num_cls=len(classes), saved_crop = 2)
        results.append(final_result)
        
        # Save checkpoint every 10 images
        if (i + 1) % 10 == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(results, f)
        
        prog_bar.update()
        time.sleep(2) # Give CPU room to breathe
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / size
    print(f"\nTotal inference time: {total_time:.2f} s")
    print(f"Average time per image: {avg_time:.2f} s")
    print(f"Efficiency (FPS): {1/avg_time:.2f}")
    

    eval_kwargs = dict(interval=1, metric='bbox')
    kwargs = {}
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox', **kwargs))
    test_config = dict(
        type='VisDroneDataset',
        classes=classes,
        ann_file=dataset_anno,
        img_prefix=dataset_root,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                #scale_factor=[1.0, 1.25, 1.5],
                #flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomCenterCropPad',
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=['logical_or', 31],
                        test_pad_add_pix=1),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg', 'border'),
                        keys=['img'])
                ])
        ])
    
    dataset = build_dataset(test_config)
    metric = dataset.evaluate(results, **eval_kwargs)
    print(metric)
