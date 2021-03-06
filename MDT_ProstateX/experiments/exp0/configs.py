#!/usr/bin/env python
# Copyright 2021 Oscar J. Pellicer-Valero
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# CHANGELOG
#
# This file has been modified to be able to handle ProstateX imaging data
# Additional functionality has beeen added:
#  - self.n_blocks now defines the actual architecture of the model, rather than its name
#  - self.match_iou sets a general match_iou to be used by default for other configurations
#  - self.keep_all_epochs If true, all epochs (not only best k) are kept
#  - self.test_subset use 'train', 'val' or 'test' data for testing  
#  - self.drop_channels_test list of channels to drop (set to 0) at test time
#  - self.test_checkpoints use a temporal ensemble of a list of epochs with
# self.test_checkpoints= steps(25, self.num_epochs, self.test_n_epochs), 
# or use instead the best validation cps with self.test_checkpoints=None
#  - self.use_prostatex_test Use prostatex test set too with masked classification loss?
#  - self.modify_class_target_fn Custom dict preprocessing function to transform classes
#  - self.droppable_channels= [[1], [2], [3], [4]] Custom augmentation: drop a channel with some prob
#  - self.channel_drop_p= 0.05 Per channel probability of droping it
#  - self.sc= 1.7 Anchor scaling factor. By default it was 2
#  - self.base_xy= 4 Base anchor size in x or y
#  - self.base_z= self.base_xy / 3 Base anchor size in z

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from default_configs import DefaultConfigs

class configs(DefaultConfigs):

    def __init__(self, server_env=False):
        
        #Change some things if debugging
        #Set always to false, it is set to true by others
        self.debugging= False

        #########################
        #    Preprocessing      #
        #########################

        #Already done!
        self.root_dir = r'../../../'
        #self.raw_data_dir = os.path.join(self.root_dir, './')
        self.pp_dir = os.path.join(self.root_dir, 'out')
        self.target_spacing = (0.5, 0.5, 3.)

        #########################
        #         I/O           #
        #########################


        # one out of [2, 3]. dimension the model operates in.
        self.dim = 3

        # one out of ['mrcnn', 'retina_net', 'retina_unet', 'detection_unet', 'ufrcnn'].
        # Only retina_unet can be used with the current model setup
        self.model = 'retina_unet'
        self.model_path= '././'

        DefaultConfigs.__init__(self, self.model, server_env, self.dim)

        # int [0 < dataset_size]. select n patients from dataset for prototyping. If None, all data is used.
        self.select_prototype_subset = None

        # path to preprocessed data.
        self.pp_name = 'out'
        self.input_df_name = 'info_df.pickle'
        self.pp_data_path = os.path.join(self.root_dir, self.pp_name)
        self.pp_test_data_path = self.pp_data_path #change if test_data in separate folder.

        # settings for deployment in cloud.
        if server_env:
            # path to preprocessed data.
            self.pp_name = ''
            self.crop_name = ''
            self.pp_data_path = ''
            self.pp_test_data_path = self.pp_data_path
            self.select_prototype_subset = None

        #########################
        #      Data Loader      #
        #########################
        
        #Use a single CV split
        self.n_cv_splits = 5

        # select modalities from preprocessed data
        self.channels = list(range(8))
        self.n_channels = len(self.channels)

        # patch_size to be used for training. pre_crop_size is the patch_size before data augmentation.
        self.pre_crop_size_2D = [160, 160]
        self.patch_size_2D = [160, 160]
        self.pre_crop_size_3D = [160, 160, 24]
        self.patch_size_3D = [160, 160, 24]
        self.patch_size = self.patch_size_2D if self.dim == 2 else self.patch_size_3D
        self.pre_crop_size = self.pre_crop_size_2D if self.dim == 2 else self.pre_crop_size_3D

        # ratio of free sampled batch elements before class balancing is triggered
        # (>0 to include "empty"/background patches.)
        self.batch_sample_slack = 0.2

        # set 2D network to operate in 3D images.
        self.merge_2D_to_3D_preds = self.dim == 2

        # feed +/- n neighbouring slices into channel dimension. set to None for no context.
        self.n_3D_context = 3
        if self.n_3D_context is not None and self.dim == 2:
            self.n_channels *= (self.n_3D_context * 2 + 1)


        #########################
        #      Architecture      #
        #########################

        self.start_filts = 48 if self.dim == 2 else 15
        self.end_filts = self.start_filts * 4 if self.dim == 2 else self.start_filts * 2
        self.n_blocks= [3, 7, 21, 3]#[3, 4, 23, 3] 'resnet101'
        self.norm = 'batch_norm' # one of None, 'instance_norm', 'batch_norm'
        # 0 for no weight decay
        self.weight_decay = 0

        # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (=default = 'kaiming_uniform')
        self.weight_init = None

        #########################
        #  Schedule / Selection #
        #########################

        self.num_epochs = 115 #99
        self.num_train_batches = 120 if self.dim == 2 else 120
        self.batch_size = 20 if self.dim == 2 else 6

        self.do_validation = True
        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is more accurate, while the latter is faster (depending on volume size)
        self.val_mode = 'val_sampling' # one of 'val_sampling' , 'val_patient'
        if self.val_mode == 'val_patient':
            self.max_val_patients = 50  # if 'None' iterates over entire val_set once.
        if self.val_mode == 'val_sampling':
            self.num_val_batches = 50

        self.optimizer = "Adam"

        # set dynamic_lr_scheduling to True to apply LR scheduling with below settings.
        self.dynamic_lr_scheduling = False
        self.lr_decay_factor = 0.25
        self.scheduling_patience = np.ceil(16000 / (self.num_train_batches * self.batch_size))
        self.scheduling_criterion = 'malignant_ap'
        self.scheduling_mode = 'min' if "loss" in self.scheduling_criterion else 'max'

        #########################
        #   Testing / Plotting  #
        #########################
        
        # General matching IOU
        self.match_iou= 1e-5
        
        # set the top-n-epochs to be saved for temporal averaging in testing.
        self.save_n_models = 5
        self.test_n_epochs = 5
        
        # If true, all epochs (not only best k) are kept
        self.keep_all_epochs= False
        
        # use 'train', 'val' or 'test' data for testing
        self.test_subset= 'test'
        
        # Drop some channels during test?
        #T2:[0], B500,B800+,ADC:[1,2,3], ktrans:[4], Perf:[5,6,7]
        self.drop_channels_test= []
        
        # use a temporal ensemble of a list of epochs, or None, to use instead the best validation cps
        def steps(start,end,n):
            if n<2: raise Exception("Behaviour not defined for n<2")
            step = (end-start)/float(n-1)
            return [int(round(start+x*step)) for x in range(n)]
        self.test_checkpoints= None #steps(25, self.num_epochs, self.test_n_epochs)

        # set a minimum epoch number for saving in case of instabilities in the first phase of training.
        self.min_save_thresh = 0 if self.dim == 2 else 0
        self.scan_det_thresh= False #Used in evaluator > return_metrics

        self.report_score_level = ['patient', 'rois']  # choose list from 'patient', 'rois'
        self.class_dict = {1:'benign', 2:'GGG_1', 3:'GGG_2', 4:'GGG_3'}  # 0 is background.
        self.patient_class_of_interest = 3  # patient metrics are only plotted for one class.
        self.ap_match_ious = [self.match_iou]  # list of ious to be evaluated for ap-scoring.

        # criteria to average over for saving epochs.
        self.model_selection_criteria = ['GGG_%d_%s'%(g, metric) for g in range(1, 4) for metric in ['ap']] + ['benign_ap']
        self.min_det_thresh = 0.  # minimum confidence value to select predictions for evaluation.
        
        #Use prostatex test set too with masked classification loss?
        self.use_prostatex_test= True
        
        # evaluates average precision per image and averages over images. instead computing one ap over data set.
        self.per_patient_ap = False

        # threshold for clustering 2D box predictions to 3D Cubes. Overlap is computed in XY.
        self.merge_3D_iou = self.match_iou
        
        #Custom dict preprocessing function to transform classes 10,1 -> 0, 2,3,4,5 -> 1, 20 -> 20
        #Please note that class 20 is a very special class for which bce loss is masked!
        #correspondance_dict= {0:0, 10:0, 1:0, 2:1, 3:1, 4:1, 5:1, 20:20}
        correspondance_dict= {0:0, 10:0, 1:1, 2:2, 3:3, 4:3, 5:3, 20:20} 
        def modify_class_target(class_targets):
            return [correspondance_dict[target] for target in class_targets]
        self.modify_class_target_fn = modify_class_target

        # threshold for clustering predictions together (wcs = weighted cluster scoring).
        # needs to be >= the expected overlap of predictions coming from one model (typically NMS threshold).
        # if too high, preds of the same object are separate clusters.
        self.wcs_iou = 1e-5

        self.plot_prediction_histograms = False
        self.plot_stat_curves = False
        
        # if True, test data lies in a separate folder and is not part of the cross validation.
        self.hold_out_test_set = True
        
        # if hold_out_test_set provided, ensemble predictions over models of all trained cv-folds.
        # implications for hold-out test sets: if True, evaluate folds separately on the test set, aggregate only the
        # evaluations. if False, aggregate the raw predictions across all folds, then evaluate.
        self.ensemble_folds = True

        #########################
        #   Data Augmentation   #
        #########################

        self.da_kwargs={
        'do_elastic_deform': True,
        'alpha':(0., 300.),
        'sigma':(20., 40.),
        'do_rotation':True,
        'angle_x': (-np.pi/30., np.pi/30),
        'angle_y': (-np.pi/20., np.pi/20) if self.dim == 2 else (0., 0.),  #must be 0!!
        'angle_z': (-np.pi/20., np.pi/20),
        'do_scale': True,
        'scale':(1/1.15, 1.15),
        'random_crop':False,
        'rand_crop_dist':  (self.patch_size[0] / 2. - 3, self.patch_size[1] / 2. - 3),
        'border_mode_data': 'constant',
        'border_cval_data': 0,
        'order_data': 3
        }
        
        self.test_aug= False #Seems to be true by default
        self.mirror_axes= (1,)
            
        #Custom augmentation: drop a channel with some prob
        #Do not drop T2 nor final channels & consider perfusion channels as one
        # 0:T2, 1:B500, 2:B800, 3:ADC, 4:ktrans,
        # 5:Prostate mask, 6:CZ mask, 7: PZ mask
        self.droppable_channels= [[1], [2], [3], [4]]
        self.channel_drop_p= 0.05

        #########################
        #   Add model specifics #
        #########################

        {'detection_unet': self.add_det_unet_configs,
         'mrcnn': self.add_mrcnn_configs,
         'ufrcnn': self.add_mrcnn_configs,
         'retina_net': self.add_mrcnn_configs,
         'retina_unet': self.add_mrcnn_configs,
         'retina_unet_coral': self.add_mrcnn_configs,
        }[self.model]()


    def add_det_unet_configs(self):

        self.learning_rate = [1e-4] * self.num_epochs

        # aggregation from pixel perdiction to object scores (connected component). One of ['max', 'median']
        self.aggregation_operation = 'max'

        # max number of roi candidates to identify per batch element and class.
        self.n_roi_candidates = 10 if self.dim == 2 else 30

        # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
        self.seg_loss_mode = 'dice_wce'

        # if <1, false positive predictions in foreground are penalized less.
        self.fp_dice_weight = 1 if self.dim == 2 else 1

        self.wce_weights = [0.3, 1, 1]
        self.detection_min_confidence = self.min_det_thresh

        # if 'True', loss distinguishes all classes, else only foreground vs. background (class agnostic).
        self.class_specific_seg_flag = True
        self.num_seg_classes = 3 if self.class_specific_seg_flag else 2
        self.head_classes = self.num_seg_classes

    def add_mrcnn_configs(self):
        
        def get_cyclic_lr(iteration, base_lr=8e-5, max_lr=3.5e-4, step_size=8, mode='triangular3'):
            if mode == 'triangular':
                scale_fn = lambda x: 1.
                scale_mode = 'cycle'
            elif mode == 'triangular2':
                scale_fn = lambda x: 1/(2.**(x-1))
                scale_mode = 'cycle'
            elif mode == 'triangular3':
                scale_fn = lambda x: 1/(1.5**(x-1))
                scale_mode = 'cycle'
            elif mode == 'exp_range':
                scale_fn = lambda x: gamma**(x)
                scale_mode = 'iterations'

            cycle = np.floor(1+iteration/(2*step_size))
            x = np.abs(iteration/step_size - 2*cycle + 1)
            if scale_mode == 'cycle':
                return base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))*scale_fn(cycle)
            else:
                return base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))*scale_fn(iteration)

        # learning rate is a list with one entry per epoch.
        self.learning_rate = [get_cyclic_lr(i) for i in np.arange(0, self.num_epochs)] #[1e-4] * self.num_epochs
        
        # disable the re-sampling of mask proposals to original size for speed-up.
        # since evaluation is detection-driven (box-matching) and not instance segmentation-driven (iou-matching),
        # mask-outputs are optional.
        self.return_masks_in_val = True
        self.return_masks_in_test = False

        # set number of proposal boxes to plot after each epoch.
        self.n_plot_rpn_props = 5 if self.dim == 2 else 30

        # number of classes for head networks: n_foreground_classes + 1 (background)
        self.head_classes = len(self.class_dict) + 1

        # seg_classes here refers to the first stage classifier (RPN)
        self.num_seg_classes = 2  # foreground vs. background

        # feature map strides per pyramid level are inferred from architecture.
        self.backbone_strides = {'xy': [4, 8, 16, 32], 'z': [1, 2, 4, 8]}

        # anchor scales are chosen according to expected object sizes in data set. Default uses only one anchor scale
        # per pyramid level. (outer list are pyramid levels (corresponding to BACKBONE_STRIDES), inner list are scales per level.)
        self.sc= 1.7 #Anchor scaling factor. By default it was 2
        self.base_xy= 4
        self.base_z= self.base_xy / 3
        self.rpn_anchor_scales = {'xy': [[self.base_xy*self.sc**0], [self.base_xy*self.sc**1], 
                                         [self.base_xy*self.sc**2], [self.base_xy*self.sc**3]], 
                                  'z':  [[self.base_z*self.sc**0],  [self.base_z*self.sc**1], 
                                         [self.base_z*self.sc**2],  [self.base_z*self.sc**3]]}


        # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
        self.pyramid_levels = [0, 1, 2, 3]

        # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
        self.n_rpn_features = 512 if self.dim == 2 else 128

        # anchor ratios and strides per position in feature maps.
        self.rpn_anchor_ratios = [1] #[0.5, 1, 2]
        self.rpn_anchor_stride = 1

        # Threshold for first stage (RPN) non-maximum suppression (NMS):  LOWER == HARDER SELECTION
        self.rpn_nms_threshold = 0.7 if self.dim == 2 else 0.7

        # loss sampling settings.
        self.rpn_train_anchors_per_image = 6  #per batch element
        self.train_rois_per_image = 6 #per batch element
        self.roi_positive_ratio = 0.5
        self.anchor_matching_iou = 0.7

        # factor of top-k candidates to draw from  per negative sample (stochastic-hard-example-mining).
        # poolsize to draw top-k candidates from will be shem_poolsize * n_negative_samples.
        self.shem_poolsize = 10

        self.pool_size = (7, 7) if self.dim == 2 else (7, 7, 3)
        self.mask_pool_size = (14, 14) if self.dim == 2 else (14, 14, 5)
        self.mask_shape = (28, 28) if self.dim == 2 else (28, 28, 10)

        self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.window = np.array([0, 0, self.patch_size[0], self.patch_size[1], 0, self.patch_size_3D[2]])
        self.scale = np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1],
                               self.patch_size_3D[2], self.patch_size_3D[2]])
        if self.dim == 2:
            self.rpn_bbox_std_dev = self.rpn_bbox_std_dev[:4]
            self.bbox_std_dev = self.bbox_std_dev[:4]
            self.window = self.window[:4]
            self.scale = self.scale[:4]

        # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
        self.pre_nms_limit = 3000 if self.dim == 2 else 6000

        # n_proposals to be selected after NMS per batch element. too high numbers blow up memory if "detect_while_training" is True,
        # since proposals of the entire batch are forwarded through second stage in as one "batch".
        self.roi_chunk_size = 2500 if self.dim == 2 else 600
        self.post_nms_rois_training = 500 if self.dim == 2 else 75
        self.post_nms_rois_inference = 500

        # Final selection of detections (refine_detections)
        self.model_max_instances_per_batch_element = 10 if self.dim == 2 else 30  # per batch element and class.
        self.detection_nms_threshold = 1e-5  # needs to be > 0, otherwise all predictions are one cluster.
        self.model_min_confidence = 0.25

        if self.dim == 2:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride))]
                 for stride in self.backbone_strides['xy']])
        else:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride)),
                  int(np.ceil(self.patch_size[2] / stride_z))]
                 for stride, stride_z in zip(self.backbone_strides['xy'], self.backbone_strides['z']
                                             )])

        if self.model == 'ufrcnn':
            self.operate_stride1 = True
            self.class_specific_seg_flag = True
            self.num_seg_classes = 3 if self.class_specific_seg_flag else 2
            self.frcnn_mode = True

        if self.model.startswith('retina'):
            # implement extra anchor-scales according to retina-net publication.
            self.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (self.sc ** (1 / 3)), ii[0] * (self.sc ** (2 / 3))] for ii in
                                            self.rpn_anchor_scales['xy']]
            self.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (self.sc ** (1 / 3)), ii[0] * (self.sc ** (2 / 3))] for ii in
                                           self.rpn_anchor_scales['z']]
            self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3

            self.n_rpn_features = 256 if self.dim == 2 else 64

            # pre-selection of detections for NMS-speedup. per entire batch.
            self.pre_nms_limit = 10000 if self.dim == 2 else 50000

            # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
            self.anchor_matching_iou = 0.5

            # if 'True', seg loss distinguishes all classes, else only foreground vs. background (class agnostic).
            self.num_seg_classes = 3 if self.class_specific_seg_flag else 2

            if self.model.startswith('retina_unet'):
                self.operate_stride1 = True
