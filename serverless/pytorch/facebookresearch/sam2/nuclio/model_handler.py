# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # self.sam_checkpoint = "./sam2_hiera_tiny.pt"
        # self.model_cfg = "sam2_hiera_t.yaml"
        
        # self.sam_checkpoint = "./sam2_hiera_small.pt"
        # self.model_cfg = "sam2_hiera_s.yaml"
        
        # self.sam_checkpoint = "./sam2_hiera_base_plus.pt"
        # self.model_cfg = "sam2_hiera_b+.yaml"
        
        self.sam_checkpoint = "./sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"
        
        self.predictor = SAM2ImagePredictor(build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device))

    # def handle(self, image, pos_points, neg_points, bboxs):
    #     pos_points, neg_points = list(pos_points), list(neg_points)
        
    #     select_points = np.array(pos_points + neg_points) if len(pos_points) or len(neg_points) else None
    #     points_labels = np.array([1]*len(pos_points) + [0]*len(neg_points)) if len(pos_points) or len(neg_points) else None
    #     bboxs = np.array(bboxs) if len(bboxs) else None
        
    #     print(f'select_points: {select_points}')
    #     print(f'points_labels: {points_labels}')
    #     print(f'bboxs: {bboxs}')
        
    #     with torch.inference_mode():
    #         self.predictor.set_image(np.array(image))
    #         masks, scores, _ = self.predictor.predict(
    #             point_coords=select_points,
    #             point_labels=points_labels,
    #             box=bboxs,
    #             multimask_output=False,
    #         )
            
    #         sorted_ind = np.argsort(scores)[::-1]
    #         best_mask = masks[sorted_ind][0]
    #         return best_mask
        
    def handle(self, image, pos_points, neg_points, bboxs):
        pos_points, neg_points = list(pos_points), list(neg_points)
        
        select_points = torch.Tensor(pos_points + neg_points) if len(pos_points) or len(neg_points) else None
        points_labels = torch.Tensor([1]*len(pos_points) + [0]*len(neg_points)) if len(pos_points) or len(neg_points) else None
        bboxs = torch.Tensor(bboxs) if len(bboxs) else None
        
       # print(f'select_points: {select_points}')
       # print(f'points_labels: {points_labels}')
       # print(f'bboxs: {bboxs}')
        
        with torch.inference_mode():
            self.predictor.set_image(np.array(image))
            masks, scores, _ = self.predictor.predict(
                point_coords=select_points,
                point_labels=points_labels,
                box=bboxs,
                multimask_output=False,
            )
            
            sorted_ind = np.argsort(scores)[::-1]
            best_mask = masks[sorted_ind][0]
            return best_mask

