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
# CHANGELOG:
#
# This file has been modified to be able to load ProstateX imaging data
# The main changes have been aplied to get_train_generators
# It includes a dictionary (ss_v2) with the different data subsets: train_fix (used always 
# for training), train_val (used for training and cross-validation), and test (used for test)

'''
Data Loader for prostate lesion segmentation. 
This dataloader expects preprocessed data per patient in .npy or .npz files and
a pandas dataframe in the same directory containing the meta-info e.g. labels, foregound slice-ids.
'''


import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import pickle
import time
import subprocess

ss_v2= {
    'train_fix': ['ProstateX-0204', 'ProstateX-0205', 'ProstateX-0206', 'ProstateX-0207', 'ProstateX-0208', 'ProstateX-0209', 'ProstateX-0210', 'ProstateX-0211', 'ProstateX-0212', 'ProstateX-0213', 'ProstateX-0214', 'ProstateX-0215', 'ProstateX-0216', 'ProstateX-0217', 'ProstateX-0218', 'ProstateX-0219', 'ProstateX-0220', 'ProstateX-0221', 'ProstateX-0222', 'ProstateX-0223', 'ProstateX-0224', 'ProstateX-0225', 'ProstateX-0226', 'ProstateX-0227', 'ProstateX-0228', 'ProstateX-0229', 'ProstateX-0230', 'ProstateX-0231', 'ProstateX-0232', 'ProstateX-0233', 'ProstateX-0234', 'ProstateX-0235', 'ProstateX-0236', 'ProstateX-0237', 'ProstateX-0238', 'ProstateX-0239', 'ProstateX-0240', 'ProstateX-0241', 'ProstateX-0242', 'ProstateX-0243', 'ProstateX-0244', 'ProstateX-0245', 'ProstateX-0246', 'ProstateX-0247', 'ProstateX-0248', 'ProstateX-0249', 'ProstateX-0250', 'ProstateX-0251', 'ProstateX-0252', 'ProstateX-0253', 'ProstateX-0254', 'ProstateX-0255', 'ProstateX-0256', 'ProstateX-0257', 'ProstateX-0258', 'ProstateX-0259', 'ProstateX-0260', 'ProstateX-0261', 'ProstateX-0262', 'ProstateX-0263', 'ProstateX-0264', 'ProstateX-0265', 'ProstateX-0266', 'ProstateX-0267', 'ProstateX-0268', 'ProstateX-0269', 'ProstateX-0270', 'ProstateX-0271', 'ProstateX-0272', 'ProstateX-0273', 'ProstateX-0274', 'ProstateX-0275', 'ProstateX-0276', 'ProstateX-0277', 'ProstateX-0278', 'ProstateX-0279', 'ProstateX-0280', 'ProstateX-0281', 'ProstateX-0282', 'ProstateX-0283', 'ProstateX-0284', 'ProstateX-0285', 'ProstateX-0286', 'ProstateX-0287', 'ProstateX-0288', 'ProstateX-0289', 'ProstateX-0290', 'ProstateX-0291', 'ProstateX-0292', 'ProstateX-0293', 'ProstateX-0294', 'ProstateX-0295', 'ProstateX-0296', 'ProstateX-0297', 'ProstateX-0298', 'ProstateX-0299', 'ProstateX-0300', 'ProstateX-0301', 'ProstateX-0302', 'ProstateX-0303', 'ProstateX-0304', 'ProstateX-0305', 'ProstateX-0306', 'ProstateX-0307', 'ProstateX-0308', 'ProstateX-0309', 'ProstateX-0310', 'ProstateX-0311', 'ProstateX-0312', 'ProstateX-0313', 'ProstateX-0314', 'ProstateX-0315', 'ProstateX-0316', 'ProstateX-0317', 'ProstateX-0318', 'ProstateX-0319', 'ProstateX-0320', 'ProstateX-0321', 'ProstateX-0322', 'ProstateX-0323', 'ProstateX-0324', 'ProstateX-0326', 'ProstateX-0327', 'ProstateX-0328', 'ProstateX-0329', 'ProstateX-0330', 'ProstateX-0332', 'ProstateX-0333', 'ProstateX-0334', 'ProstateX-0335', 'ProstateX-0336', 'ProstateX-0337', 'ProstateX-0338', 'ProstateX-0339', 'ProstateX-0340', 'ProstateX-0341', 'ProstateX-0342', 'ProstateX-0343', 'ProstateX-0344', 'ProstateX-0345'], 
     'train_val': ['ProstateX-0000', 'ProstateX-0002', 'ProstateX-0004', 'ProstateX-0005', 'ProstateX-0006', 'ProstateX-0007', 'ProstateX-0008', 'ProstateX-0009', 'ProstateX-0011', 'ProstateX-0012', 'ProstateX-0013', 'ProstateX-0014', 'ProstateX-0015', 'ProstateX-0016', 'ProstateX-0017', 'ProstateX-0019', 'ProstateX-0020', 'ProstateX-0021', 'ProstateX-0023', 'ProstateX-0024', 'ProstateX-0025', 'ProstateX-0027', 'ProstateX-0028', 'ProstateX-0029', 'ProstateX-0030', 'ProstateX-0031', 'ProstateX-0033', 'ProstateX-0035', 'ProstateX-0037', 'ProstateX-0038', 'ProstateX-0040', 'ProstateX-0041', 'ProstateX-0042', 'ProstateX-0043', 'ProstateX-0044', 'ProstateX-0046', 'ProstateX-0047', 'ProstateX-0049', 'ProstateX-0050', 'ProstateX-0051', 'ProstateX-0053', 'ProstateX-0054', 'ProstateX-0056', 'ProstateX-0058', 'ProstateX-0059', 'ProstateX-0060', 'ProstateX-0063', 'ProstateX-0064', 'ProstateX-0065', 'ProstateX-0066', 'ProstateX-0067', 'ProstateX-0068', 'ProstateX-0069', 'ProstateX-0070', 'ProstateX-0071', 'ProstateX-0072', 'ProstateX-0075', 'ProstateX-0078', 'ProstateX-0080', 'ProstateX-0081', 'ProstateX-0082', 'ProstateX-0083', 'ProstateX-0084', 'ProstateX-0085', 'ProstateX-0086', 'ProstateX-0087', 'ProstateX-0088', 'ProstateX-0089', 'ProstateX-0090', 'ProstateX-0091', 'ProstateX-0092', 'ProstateX-0093', 'ProstateX-0094', 'ProstateX-0095', 'ProstateX-0096', 'ProstateX-0097', 'ProstateX-0098', 'ProstateX-0099', 'ProstateX-0100', 'ProstateX-0101', 'ProstateX-0102', 'ProstateX-0103', 'ProstateX-0104', 'ProstateX-0105', 'ProstateX-0106', 'ProstateX-0107', 'ProstateX-0108', 'ProstateX-0109', 'ProstateX-0110', 'ProstateX-0111', 'ProstateX-0112', 'ProstateX-0116', 'ProstateX-0117', 'ProstateX-0120', 'ProstateX-0121', 'ProstateX-0122', 'ProstateX-0123', 'ProstateX-0124', 'ProstateX-0125', 'ProstateX-0126', 'ProstateX-0128', 'ProstateX-0129', 'ProstateX-0131', 'ProstateX-0132', 'ProstateX-0133', 'ProstateX-0135', 'ProstateX-0136', 'ProstateX-0140', 'ProstateX-0141', 'ProstateX-0143', 'ProstateX-0144', 'ProstateX-0145', 'ProstateX-0148', 'ProstateX-0149', 'ProstateX-0150', 'ProstateX-0151', 'ProstateX-0152', 'ProstateX-0153', 'ProstateX-0155', 'ProstateX-0156', 'ProstateX-0158', 'ProstateX-0160', 'ProstateX-0162', 'ProstateX-0163', 'ProstateX-0165', 'ProstateX-0167', 'ProstateX-0168', 'ProstateX-0169', 'ProstateX-0171', 'ProstateX-0172', 'ProstateX-0173', 'ProstateX-0174', 'ProstateX-0175', 'ProstateX-0176', 'ProstateX-0177', 'ProstateX-0178', 'ProstateX-0179', 'ProstateX-0180', 'ProstateX-0181', 'ProstateX-0182', 'ProstateX-0183', 'ProstateX-0184', 'ProstateX-0185', 'ProstateX-0186', 'ProstateX-0187', 'ProstateX-0188', 'ProstateX-0189', 'ProstateX-0190', 'ProstateX-0191', 'ProstateX-0192', 'ProstateX-0193', 'ProstateX-0194', 'ProstateX-0195', 'ProstateX-0196', 'ProstateX-0197', 'ProstateX-0198', 'ProstateX-0199', 'ProstateX-0201', 'ProstateX-0203'],  
    'test': ['ProstateX-0001', 'ProstateX-0003', 'ProstateX-0010', 'ProstateX-0018', 'ProstateX-0022', 'ProstateX-0026', 'ProstateX-0032', 'ProstateX-0034', 'ProstateX-0036', 'ProstateX-0039', 'ProstateX-0045', 'ProstateX-0048', 'ProstateX-0052', 'ProstateX-0055', 'ProstateX-0057', 'ProstateX-0061', 'ProstateX-0062', 'ProstateX-0073', 'ProstateX-0074', 'ProstateX-0076', 'ProstateX-0077', 'ProstateX-0079', 'ProstateX-0113', 'ProstateX-0114', 'ProstateX-0115', 'ProstateX-0118', 'ProstateX-0119', 'ProstateX-0127', 'ProstateX-0130', 'ProstateX-0134', 'ProstateX-0137', 'ProstateX-0138', 'ProstateX-0139', 'ProstateX-0142', 'ProstateX-0146', 'ProstateX-0147', 'ProstateX-0154', 'ProstateX-0157', 'ProstateX-0159', 'ProstateX-0161', 'ProstateX-0164', 'ProstateX-0166', 'ProstateX-0170', 'ProstateX-0200', 'ProstateX-0202']
       }


# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates
from custom_transform import RandomChannelDeleteTransform

import utils.dataloader_utils as dutils
import utils.exp_utils as utils

def get_train_generators(cf, logger):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. 
    (inner loop test set)
    """
        
    #Generate info_df.pickle when training for the first time
    files = [os.path.join(cf.pp_dir, f) for f in os.listdir(cf.pp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)
    df.to_pickle(os.path.join(cf.pp_dir, 'info_df.pickle'))
    print ("Aggregated meta info to df with length", len(df))
    
    #load data
    all_data = load_dataset(cf, logger)
    splits_file = os.path.join(cf.exp_dir, 'fold_ids.pickle')
    ss= ss_v2
        
    #Get subset idx
    pids= list(pd.read_pickle(os.path.join(cf.pp_dir, 'info_df.pickle')).pid.values)

    #Regenerate fold_ids.pickle?
    if not os.path.isfile(splits_file) and not cf.created_fold_id_pickle:           
        #Get subsets
        prostatex_train_fix, prostatex_train_val, prostatex_test= ss['train_fix'] if cf.use_prostatex_test else [], \
                                                                  ss['train_val'], ss['test']
        #Build samples for all folds     
        from sklearn.model_selection import KFold

        prostatex_kf= KFold(n_splits=cf.n_cv_splits, random_state=5, shuffle=True)
        prostatex_kf.get_n_splits(prostatex_train_val)

        fg= []
        for i, (prostatex_train_index, prostatex_val_index) in \
            enumerate(prostatex_kf.split(prostatex_train_val)) :
            fg.append([
                list(np.array(prostatex_train_val)[prostatex_train_index]) + prostatex_train_fix,
                list(np.array(prostatex_train_val)[prostatex_val_index]),
                prostatex_test,
                i
            ])

        #Save it
        pickle.dump(fg, open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'wb'))
        cf.created_fold_id_pickle = True
    else:
        with open(splits_file, 'rb') as handle:
            fg = pickle.load(handle)

    train_pids, val_pids, test_pids, _ = fg[cf.fold]
    
    print('Train IDs:', train_pids)
    print('\nValidation IDs:', val_pids)
    print('\nTest IDs:', test_pids)

    train_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in train_pids)}
    val_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in val_pids)}

    logger.info("data set loaded with: {} train / {} val / {} test patients".format(len(train_pids), len(val_pids), len(test_pids)))
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, is_training=True)
    batch_gen['val_sampling'] = create_data_gen_pipeline(val_data, cf=cf, is_training=False)
    if cf.val_mode == 'val_patient':
        batch_gen['val_patient'] = PatientBatchIterator(val_data, cf=cf)
        batch_gen['n_val'] = len(val_pids) if cf.max_val_patients is None else min(len(val_pids), cf.max_val_patients)
    else:
        batch_gen['n_val'] = cf.num_val_batches

    return batch_gen

def get_test_generator(cf, logger):
    """
    wrapper function for creating the test batch generator pipeline.
    selects patients according to cv folds (generated by first run/fold of experiment)
    """
    with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
        fold_list = pickle.load(handle)
    train_pids, val_pids, test_pids, _ = fold_list[cf.fold]
    pp_name = None
    if cf.test_subset == 'train':
        pids= train_pids 
        logger.warn('INFO: using train set for testing')
    elif cf.test_subset == 'val':
        pids= val_pids 
        logger.warn('INFO: using validation set for testing')
    elif cf.test_subset == 'test':
        pids= test_pids
        logger.warn('INFO: using test set for testing')
    else:
        raise ValueError('cf.test_subset must be either train, val or test')
    print('Pids:', pids)    
    
    test_data = load_dataset(cf, logger, pids, pp_data_path=cf.pp_test_data_path, pp_name=pp_name)
    logger.info("data set loaded with: {} test patients".format(len(pids)))
    batch_gen = {}
    batch_gen['test'] = PatientBatchIterator(test_data, cf=cf)
    batch_gen['n_test'] = len(pids) if cf.max_test_patients=="all" else \
        min(cf.max_test_patients, len(pids))
    return batch_gen

def load_dataset(cf, logger, subset_pids=None, pp_data_path=None, pp_name=None):
    """
    loads the dataset. if deployed in cloud also copies and unpacks the data to the working directory.
    :param subset_pids: subset pids to be loaded from the dataset. used e.g. for testing to only load the test folds.
    :return: data: dictionary with one entry per patient (in this case per patient-breast, since they are treated as
    individual images for training) each entry is a dictionary containing respective meta-info as well as paths to the preprocessed
    numpy arrays to be loaded during batch-generation
    """
    if pp_data_path is None:
        pp_data_path = cf.pp_data_path
    if pp_name is None:
        pp_name = cf.pp_name
    if cf.server_env:
        copy_data = True
        target_dir = os.path.join(cf.data_dest, pp_name)
        if not os.path.exists(target_dir):
            cf.data_source_dir = pp_data_path
            os.makedirs(target_dir)
            subprocess.call('rsync -av {} {}'.format(
                os.path.join(cf.data_source_dir, cf.input_df_name), os.path.join(target_dir, cf.input_df_name)), shell=True)
            logger.info('created target dir and info df at {}'.format(os.path.join(target_dir, cf.input_df_name)))

        elif subset_pids is None:
            copy_data = False

        pp_data_path = target_dir


    p_df = pd.read_pickle(os.path.join(pp_data_path, cf.input_df_name))

    if cf.select_prototype_subset is not None:
        prototype_pids = p_df.pid.tolist()[:cf.select_prototype_subset]
        p_df = p_df[p_df.pid.isin(prototype_pids)]
        logger.warning('WARNING: using prototyping data subset!!!')

    if subset_pids is not None:
        p_df = p_df[p_df.pid.isin(subset_pids)]
        logger.info('subset: selected {} instances from df'.format(len(p_df)))

    if cf.server_env:
        if copy_data:
            copy_and_unpack_data(logger, p_df.pid.tolist(), cf.fold_dir, cf.data_source_dir, target_dir)

    class_targets = p_df['class_target'].tolist()
    pids = p_df.pid.tolist()
    imgs = [os.path.join(pp_data_path, '{}_img.npy'.format(pid)) for pid in pids]
    segs = [os.path.join(pp_data_path,'{}_rois.npy'.format(pid)) for pid in pids]

    data = OrderedDict()
    # for the experiment conducted here, malignancy scores are binarized following self.cf.modify_class_target_fn
    for ix, pid in enumerate(pids):
        targets = cf.modify_class_target_fn(class_targets[ix])
        data[pid] = {'data': imgs[ix], 'seg': segs[ix], 'pid': pid, 'class_target': targets}
        data[pid]['fg_slices'] = p_df['fg_slices'].tolist()[ix]
        #data[pid]['class_unknown']= 

    return data

def create_data_gen_pipeline(patient_data, cf, is_training=True):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """

    # create instance of batch generator as first element in pipeline.
    data_gen = BatchGenerator(patient_data, batch_size=cf.batch_size, cf=cf)

    # add transformations to pipeline.
    my_transforms = []
    if is_training:
        mirror_transform = Mirror(axes=cf.mirror_axes) #np.arange(cf.dim)
        my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'])

        my_transforms.append(spatial_transform)
        my_transforms.append(RandomChannelDeleteTransform(cf.droppable_channels, cf.channel_drop_p))
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, get_rois_from_seg_flag=False, class_specific_seg_flag=cf.class_specific_seg_flag))
    all_transforms = Compose(my_transforms)
    if cf.debugging:
        multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    else:
        multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    return multithreaded_generator


class BatchGenerator(SlimDataLoaderBase):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, y, x(, z)) / seg (b, 1, y, x(, z)) / pids / class_target
    """
    def __init__(self, data, batch_size, cf):
        super(BatchGenerator, self).__init__(data, batch_size)
        
        self.cf = cf
        self.crop_margin = np.array(self.cf.patch_size)/8. #min distance of ROI center to edge of cropped_patch.
        self.p_fg = 0.5

    def generate_train_batch(self):

        batch_data, batch_segs, batch_pids, batch_targets, batch_patient_labels = [], [], [], [], []
        class_targets_list =  [v['class_target'] for (k, v) in self._data.items()]

        #I am turning this off, because it is problematic with my class 20
        if False: #self.cf.head_classes > 2:
            # samples patients towards equilibrium of foreground classes on a roi-level (after randomly sampling the ratio "batch_sample_slack).
            batch_ixs = dutils.get_class_balanced_patients(
                class_targets_list, self.batch_size, self.cf.head_classes - 1, slack_factor=self.cf.batch_sample_slack)
        else:
            batch_ixs = np.random.choice(len(class_targets_list), self.batch_size)

        patients = list(self._data.items())

        for b in batch_ixs:
            patient = patients[b][1]

            # data shape: from (c, z, y, x) to (c, y, x, z).
            data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(3, 1, 2, 0))
            seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(3, 1, 2, 0))
            batch_pids.append(patient['pid'])
            batch_targets.append(patient['class_target'])

            if self.cf.dim == 2:
                # draw random slice from patient while oversampling slices containing foreground objects with p_fg.
                if len(patient['fg_slices']) > 0:
                    fg_prob = self.p_fg / len(patient['fg_slices'])
                    bg_prob = (1 - self.p_fg) / (data.shape[3] - len(patient['fg_slices']))
                    slices_prob = [fg_prob if ix in patient['fg_slices'] else bg_prob for ix in range(data.shape[3])]
                    slice_id = np.random.choice(data.shape[3], p=slices_prob)
                else:
                    slice_id = np.random.choice(data.shape[3])

                # if set to not None, add neighbouring slices to each selected slice in channel dimension.
                if self.cf.n_3D_context is not None:
                    padded_data = dutils.pad_nd_image(data[0], [(data.shape[-1] + (self.cf.n_3D_context*2))], mode='constant')
                    padded_slice_id = slice_id + self.cf.n_3D_context
                    data = (np.concatenate([padded_data[..., ii][np.newaxis] for ii in range(
                        padded_slice_id - self.cf.n_3D_context, padded_slice_id + self.cf.n_3D_context + 1)], axis=0))
                else:
                    data = data[..., slice_id]
                seg = seg[..., slice_id]

            # pad data if smaller than pre_crop_size.
            if np.any([data.shape[dim + 1] < ps for dim, ps in enumerate(self.cf.pre_crop_size)]):
                new_shape = [np.max([data.shape[dim + 1], ps]) for dim, ps in enumerate(self.cf.pre_crop_size)]
                data = dutils.pad_nd_image(data, new_shape, mode='constant')
                seg = dutils.pad_nd_image(seg, new_shape, mode='constant')

            # crop patches of size pre_crop_size, while sampling patches containing foreground with p_fg.
            crop_dims = [dim for dim, ps in enumerate(self.cf.pre_crop_size) if data.shape[dim + 1] > ps]
            if len(crop_dims) > 0:
                fg_prob_sample = np.random.rand(1)
                # with p_fg: sample random pixel from random ROI and shift center by random value.
                if fg_prob_sample < self.p_fg and np.sum(seg) > 0:
                    seg_ixs = np.argwhere(seg == np.random.choice(np.unique(seg)[1:], 1))
                    roi_anchor_pixel = seg_ixs[np.random.choice(seg_ixs.shape[0], 1)][0]
                    assert seg[tuple(roi_anchor_pixel)] > 0
                    # sample the patch center coords. constrained by edges of images - pre_crop_size /2. And by
                    # distance to the desired ROI < patch_size /2.
                    # (here final patch size to account for center_crop after data augmentation).
                    sample_seg_center = {}
                    for ii in crop_dims:
                        low = np.max((self.cf.pre_crop_size[ii]//2, roi_anchor_pixel[ii] - (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
                        high = np.min((data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2,
                                       roi_anchor_pixel[ii] + (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
                        # happens if lesion on the edge of the image. dont care about roi anymore,
                        # just make sure pre-crop is inside image.
                        if low >= high:
                            low = data.shape[ii + 1] // 2 - (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
                            high = data.shape[ii + 1] // 2 + (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
                        sample_seg_center[ii] = np.random.randint(low=low, high=high)

                else:
                    # not guaranteed to be empty. probability of emptiness depends on the data.
                    sample_seg_center = {ii: np.random.randint(low=self.cf.pre_crop_size[ii]//2,
                                                           high=data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2) for ii in crop_dims}

                for ii in crop_dims:
                    min_crop = int(sample_seg_center[ii] - self.cf.pre_crop_size[ii] // 2)
                    max_crop = int(sample_seg_center[ii] + self.cf.pre_crop_size[ii] // 2)
                    data = np.take(data, indices=range(min_crop, max_crop), axis=ii + 1)
                    seg = np.take(seg, indices=range(min_crop, max_crop), axis=ii)

            batch_data.append(data)
            batch_segs.append(seg)

        data = np.array(batch_data)
        seg = np.array(batch_segs).astype(np.uint8)
        class_target = np.array(batch_targets)
        return {'data': data, 'seg': seg, 'pid': batch_pids, 'class_target': class_target}


class PatientBatchIterator(SlimDataLoaderBase):
    """
    creates a test generator that iterates over entire given dataset returning 1 patient per batch.
    Can be used for monitoring if cf.val_mode = 'patient_val' for a monitoring closer to actualy evaluation (done in 3D),
    if willing to accept speed-loss during training.
    :return: out_batch: dictionary containing one patient with batch_size = n_3D_patches in 3D or
    batch_size = n_2D_patches in 2D .
    """
    def __init__(self, data, cf): #threads in augmenter
        super(PatientBatchIterator, self).__init__(data, 0)
        self.cf = cf
        self.patient_ix = 0
        self.dataset_pids = [v['pid'] for (k, v) in data.items()]
        self.patch_size = cf.patch_size
        if len(self.patch_size) == 2:
            self.patch_size = self.patch_size + [1]


    def generate_train_batch(self):
        pid = self.dataset_pids[self.patient_ix]
        patient = self._data[pid]
        # data shape: from (c, z, y, x) to (c, y, x, z).
        data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(3, 1, 2, 0)).copy()
        seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(3, 1, 2, 0))[0].copy()
        batch_class_targets = np.array([patient['class_target']])

        # pad data if smaller than patch_size seen during training.
        if np.any([data.shape[dim + 1] < ps for dim, ps in enumerate(self.patch_size)]):
            new_shape = [data.shape[0]] + [np.max([data.shape[dim + 1], self.patch_size[dim]]) for dim, ps in enumerate(self.patch_size)]
            data = dutils.pad_nd_image(data, new_shape) # use 'return_slicer' to crop image back to original shape.
            seg = dutils.pad_nd_image(seg, new_shape)

        # get 3D targets for evaluation, even if network operates in 2D. 2D predictions will be merged to 3D in predictor.
        if self.cf.dim == 3 or self.cf.merge_2D_to_3D_preds:
            out_data = data[np.newaxis]
            out_seg = seg[np.newaxis, np.newaxis]
            out_targets = batch_class_targets

            batch_3D = {'data': out_data, 'seg': out_seg, 'class_target': out_targets, 'pid': pid}
            converter = ConvertSegToBoundingBoxCoordinates(dim=3, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
            batch_3D = converter(**batch_3D)
            batch_3D.update({'patient_bb_target': batch_3D['bb_target'],
                                  'patient_roi_labels': batch_3D['class_target'],
                                  'original_img_shape': out_data.shape})

        if self.cf.dim == 2:
            out_data = np.transpose(data, axes=(3, 0, 1, 2))  # (z, c, y, x )
            out_seg = np.transpose(seg, axes=(2, 0, 1))[:, np.newaxis]
            out_targets = np.array(np.repeat(batch_class_targets, out_data.shape[0], axis=0))

            # if set to not None, add neighbouring slices to each selected slice in channel dimension.
            if self.cf.n_3D_context is not None:
                slice_range = range(self.cf.n_3D_context, out_data.shape[0] + self.cf.n_3D_context)
                out_data = np.pad(out_data, ((self.cf.n_3D_context, self.cf.n_3D_context), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
                out_data = np.array(
                    [np.concatenate([out_data[ii] for ii in range(
                        slice_id - self.cf.n_3D_context, slice_id + self.cf.n_3D_context + 1)], axis=0) for slice_id in
                     slice_range])

            batch_2D = {'data': out_data, 'seg': out_seg, 'class_target': out_targets, 'pid': pid}
            converter = ConvertSegToBoundingBoxCoordinates(dim=2, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
            batch_2D = converter(**batch_2D)

            if self.cf.merge_2D_to_3D_preds:
                batch_2D.update({'patient_bb_target': batch_3D['patient_bb_target'],
                                      'patient_roi_labels': batch_3D['patient_roi_labels'],
                                      'original_img_shape': out_data.shape})
            else:
                batch_2D.update({'patient_bb_target': batch_2D['bb_target'],
                                 'patient_roi_labels': batch_2D['class_target'],
                                 'original_img_shape': out_data.shape})

        out_batch = batch_3D if self.cf.dim == 3 else batch_2D
        patient_batch = out_batch

        # crop patient-volume to patches of patch_size used during training. stack patches up in batch dimension.
        # in this case, 2D is treated as a special case of 3D with patch_size[z] = 1.
        if np.any([data.shape[dim + 1] > self.patch_size[dim] for dim in range(3)]):
            patch_crop_coords_list = dutils.get_patch_crop_coords(data[0], self.patch_size)
            new_img_batch, new_seg_batch, new_class_targets_batch = [], [], []

            for cix, c in enumerate(patch_crop_coords_list):

                seg_patch = seg[c[0]:c[1], c[2]: c[3], c[4]:c[5]]
                new_seg_batch.append(seg_patch)

                # if set to not None, add neighbouring slices to each selected slice in channel dimension.
                # correct patch_crop coordinates by added slices of 3D context.
                if self.cf.dim == 2 and self.cf.n_3D_context is not None:
                    tmp_c_5 = c[5] + (self.cf.n_3D_context * 2)
                    if cix == 0:
                        data = np.pad(data, ((0, 0), (0, 0), (0, 0), (self.cf.n_3D_context, self.cf.n_3D_context)), 'constant', constant_values=0)
                else:
                    tmp_c_5 = c[5]

                new_img_batch.append(data[:, c[0]:c[1], c[2]:c[3], c[4]:tmp_c_5])

            data = np.array(new_img_batch) # (n_patches, c, x, y, z)
            seg = np.array(new_seg_batch)[:, np.newaxis]  # (n_patches, 1, x, y, z)
            batch_class_targets = np.repeat(batch_class_targets, len(patch_crop_coords_list), axis=0)

            if self.cf.dim == 2:
                if self.cf.n_3D_context is not None:
                    data = np.transpose(data[:, 0], axes=(0, 3, 1, 2))
                else:
                    # all patches have z dimension 1 (slices). discard dimension
                    data = data[..., 0]
                seg = seg[..., 0]

            patch_batch = {'data': data, 'seg': seg, 'class_target': batch_class_targets, 'pid': pid}
            patch_batch['patch_crop_coords'] = np.array(patch_crop_coords_list)
            patch_batch['patient_bb_target'] = patient_batch['patient_bb_target']
            patch_batch['patient_roi_labels'] = patient_batch['patient_roi_labels']
            patch_batch['original_img_shape'] = patient_batch['original_img_shape']

            converter = ConvertSegToBoundingBoxCoordinates(self.cf.dim, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
            patch_batch = converter(**patch_batch)
            out_batch = patch_batch

        self.patient_ix += 1
        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0

        out_batch['data'][:,self.cf.drop_channels_test,]= 0.
        return out_batch

def copy_and_unpack_data(logger, pids, fold_dir, source_dir, target_dir):

    start_time = time.time()
    with open(os.path.join(fold_dir, 'file_list.txt'), 'w') as handle:
        for pid in pids:
            handle.write('{}_img.npz\n'.format(pid))
            handle.write('{}_rois.npz\n'.format(pid))

    subprocess.call('rsync -av --files-from {} {} {}'.format(os.path.join(fold_dir, 'file_list.txt'),
        source_dir, target_dir), shell=True)
    n_threads = 8
    dutils.unpack_dataset(target_dir, threads=n_threads)
    copied_files = os.listdir(target_dir)
    t = utils.get_formatted_duration(time.time() - start_time)
    logger.info("\ncopying and unpacking data set finished using {} threads.\n{} files in target dir: {}. Took {}\n"
        .format(n_threads, len(copied_files), target_dir, t))


if __name__=="__main__":

    total_stime = time.time()

    cf_file = utils.import_module("cf", "configs.py")
    cf = cf_file.configs()

    cf.created_fold_id_pickle = False
    cf.exp_dir = "dev/"
    cf.plot_dir = cf.exp_dir + "plots"
    os.makedirs(cf.exp_dir, exist_ok=True)
    cf.fold = 0
    logger = utils.get_logger(cf.exp_dir)

    #batch_gen = get_train_generators(cf, logger)
    #train_batch = next(batch_gen["train"])

    test_gen = get_test_generator(cf, logger)
    test_batch = next(test_gen["test"])

    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))