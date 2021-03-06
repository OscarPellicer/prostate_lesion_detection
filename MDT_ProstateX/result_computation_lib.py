# Copyright 2021 Oscar José Pellicer Valero
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
    This file contains all functions needed to provide funtionality to `Result analysis.ipynb`
'''

#Import some libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, auc, roc_auc_score,\
    precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
import pandas as pd
from functools import partial
import copy

#Import plot_lib
from pathlib import Path
import sys, os
sys.path.append(os.path.join(Path.home(), 'plot_lib'))
from plot_lib import plot, plot_multi_mask, plot4

def get_optimal_thresholds(y, yp):
    '''
        Compute two thresholds to evaluate sensitivity/specificity:
        maximum sensititivy, balanced sensitivity/specificity, and maximum specificity
        
        Parameters
        ----------
        y: array or list
            List of Ground Truth values (0 or 1)
        yp: array or list
            List of predicted scores in range 0-1
            
        Returns
        -------
        max_spec_th: float
        optimal_th: float
        max_sens_th: float
    '''
    eps= 1e-4
    
    #Get optimal sens * spec threshold
    fp_tp_th= list(zip(*roc_curve(y, yp)))
    optimal_th= sorted(fp_tp_th, key=lambda a:a[1]*(1-a[0]), reverse=True)[0][2]
    
    #Get maximum specificity
    max_spec_th= 1.
    for fpr, _, th in fp_tp_th[1:]:
        if fpr != fp_tp_th[1][0]:
            break
        else:
            max_spec_th= th

    #Get maximum sensitivity
    max_sens_th= 0.
    rev_fp_tp_th= list(reversed(fp_tp_th))
    for _, tpr, th in rev_fp_tp_th[1:]:
        if tpr != rev_fp_tp_th[1][1]:
            break
        else:
            max_sens_th= th
    
    return max_spec_th - eps, optimal_th - eps, max_sens_th - eps

def gen_dsc(y_true, y_pred, alpha=0.5):
    '''
        Generalized DICE / Twersky loss with alpha=0.5 and beta=1-alpha
        
        Parameters
        ----------
        y_true: array or list
            List of Ground Truth values (0 or 1)
        y_pred: array or list
            List of predicted scores in range 0-1
        alpha: float, default 0.5
            Sensitivity/specificity bias
            Set to 0.25 for higher sensitivity, and to 0.75 for lower
    '''
    beta= 1 - alpha
    y = y_true.flatten();  yp = y_pred.flatten()
    I1= yp.sum(); I2= y.sum(); I3= (yp*y).sum()
    return (I3+1) / (I1*alpha + I2*beta+1)

def compact_class_detections(test_results_list, cl=-1, benign_class=1, match_iou= 0.25,
                             class_weights=np.ones(100)):
    '''
        Takes a list of Bounding Boxes (BBs) and returns only the highest-scoring 
        BB for highly overlapped detections (e.g.: `match_iou` > 0.25). 
        
        Parameters
        ----------
        test_results_list
            List of BBs returned by the model
        match_iou
            BBs above this threshold are considered highly overlapped
        cl
            Detected classes below `cl` are directly removed
        benign_class
            Classes equal or below `bening_class` are only removed in case of doubt
        class_weights
            Dictionary of weights to apply to each class' scores when comparing them
    '''
    #Get the final compacted list we will return
    test_results_list_compacted= copy.deepcopy(test_results_list)
    
    #Iterate over patients
    for pat_idx, patient_results in enumerate(test_results_list_compacted):

        #Get patient data
        pid= patient_results[1]
        b_boxes_list= patient_results[0]['boxes'][0]
        
        #Get candidate boxes (only class >= cl)
        b_cand_boxes = [box for box in b_boxes_list 
                            if (box['box_type'] == 'det' 
                                and box['box_pred_class_id'] >= cl)]
        
        #Remove all candidate boxes from the compacted list
        test_results_list_compacted[pat_idx][0]['boxes'][0]= [box for box in b_boxes_list 
                                                            if not (box['box_type'] == 'det' 
                                                                and box['box_pred_class_id'] >= cl)]

        #Sort candidate boxes by score irrespective of their class
        b_cand_boxes= sorted(b_cand_boxes, key= lambda i: i['box_score']) 
        
        #For each box, delete depending on some checks
        #Important keys: box_pred_class_id, box_score, box_coords
        if b_cand_boxes!=[]:
            b_cand_boxes_boxes= np.array([box['box_coords'] for box in b_cand_boxes])
            ious= compute_overlaps(b_cand_boxes_boxes, b_cand_boxes_boxes)
            for i, box in enumerate(b_cand_boxes):
                for j, other_box in enumerate(b_cand_boxes):
                    keep=True
                    #There it a match
                    if i!= j and ious[i,j] > match_iou:
                        #Both boxes are not benign
                        if box['box_pred_class_id'] > benign_class and other_box['box_pred_class_id'] > benign_class:
                            #Drop if the other box is better
                            if other_box['box_score']*class_weights[other_box['box_pred_class_id']] > \
                               box['box_score']*class_weights[box['box_pred_class_id']]:
                                keep=False
                        #Current box is benign and the other is not
                        elif box['box_pred_class_id'] == benign_class and other_box['box_pred_class_id'] > benign_class:
                            keep= False
                        #Both boxes are benign
                        elif box['box_pred_class_id'] == benign_class and other_box['box_pred_class_id'] == benign_class:
                            #Drop if the other box is better
                            if other_box['box_score'] > box['box_score']:
                                keep=False
                        #If current box is not benign and the other is benign, keep (do nothing)
                        else:
                            pass
                        #print('Evaluated box vs king box:\n', box, other_box, ious[i,j])
                        #If keep was set to False, break out of the loop
                        if not keep:
                            break
                if keep:
                    test_results_list_compacted[pat_idx][0]['boxes'][0].append(box)

    return test_results_list_compacted

def compute_distances(boxes1, boxes2, spacing, normalize_by_size=False):
    '''
        Computes the distance between centroids of lists of Bounding Boxes (BBs)
        
        Parameters
        ---------
        boxes1, boxes2
            BBs to comepare [N, (y1, x1, y2, x2)]. / 3D: (z1, z2))
        spacing: list or tuple
            Spacing (e.g. in mm) of the voxels (x, y, z)
        normalize_by_size: bool, default False
            If True, distances are normalized by the average side length of the detected BB
    '''
    # Areas of anchors and GT boxes
    xs, ys, zs= spacing
    if boxes1.shape[1] == 4:
        raise NotImplementedError('2D version of this function is not yet implemented')

    else:
        # Areas of anchors and GT boxes
        c1y = np.mean([boxes1[:, 2], boxes1[:, 0]], axis=0)
        c1x = np.mean([boxes1[:, 3], boxes1[:, 1]], axis=0)
        c1z = np.mean([boxes1[:, 5], boxes1[:, 4]], axis=0)
        centroid1= np.array([c1x, c1y, c1z]).T
        
        c2y = np.mean([boxes2[:, 2], boxes2[:, 0]], axis=0)
        c2x = np.mean([boxes2[:, 3], boxes2[:, 1]], axis=0)
        c2z = np.mean([boxes2[:, 5], boxes2[:, 4]], axis=0)
        centroid2= np.array([c2x, c2y, c2z]).T
        
        size1 = np.float_power( (boxes1[:, 2] - boxes1[:, 0])*ys * (boxes1[:, 3] - boxes1[:, 1])*xs * \
                                (boxes1[:, 5] - boxes1[:, 4])*zs, 1/3 )
        
        # Compute distances to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the distance value.
        distances = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(distances.shape[1]):
            distances[:, i] = np.linalg.norm((centroid1-centroid2[i][None,:])*np.array([xs, ys, zs]), axis=1)
            if normalize_by_size:
                distances[:,i]/= size1
        return distances

def match_lesions(test_results_list, patients_test, drop_nans=True, cl=3, 
                  match_iou=1e-5, score_col='score', class_col='class',
                  use_distance=True, distance_threshold=15, 
                  spacing=(0.5, 0.5, 3), normalize_by_size=False,
                  class_weights=np.ones(100)):
    '''
        Matches all Ground Truth (GT) Bounding Boxes (BBs) to a single predicted BB
        
        Parameters
        ---------
        test_results_list
            list of BBs to process
        patients_test: DataFrame
            Dataframe with a double index (patient_ID, lesion_ID) and at least a column to save the
            score to, defined by `score_col`, and antoher to save the class, defined by `class_col`
        drop_nans: bool, default True
            If True, removes patients from `patients_test` whose lesion_IDs are not found within the BBs
        cl: int, default 3
            Detected classes below `cl` are ignored
        match_iou: int, default 1e-5
            If `use_distance=False`, minimum IoU required to consider a detection for matching a GT
        score_col: str, default 'score'
            Name of the column of `patients_test` to store final detection score
        class_col: str, default 'class'
            Name of the column of `patients_test` to store final detection class
        use_distance: bool, default True
            Use distance for matching, instead of IoU
        distance_threshold: int, default 15
            Maximum distance required to consider a detection for matching a GT
        spacing: tuple or list, default (0.5, 0.5, 3)
            Voxel spacing
        normalize_by_size: bool, default False
            If True, distances are normalized by the average side length of the detected BB
        class_weights: dict or list, default [1,1,1,...]
            Dictionary of weights to apply to each class' scores when comparing them
    '''
    for patient_results in test_results_list:

        #Get patient data
        pid= patient_results[1]
        b_boxes_list= patient_results[0]['boxes'][0]

        #Target boxes (any class)
        b_tar_boxes = np.array([box['box_coords'] for box in b_boxes_list if
                                box['box_type'] == 'gt'])

        #Target boxes's class
        b_tar_labels = np.array([box['box_label'] for box in b_boxes_list if
                                box['box_type'] == 'gt'])
        #Set labels to the df
        for i,l in enumerate(b_tar_labels):
            patients_test.loc[(pid, i), 'ClinSig2']= bool(l - 1)

        #Candidate boxes (only class >= cl)
        b_cand_boxes = np.array([box['box_coords'] for box in b_boxes_list if
                                 (box['box_type'] == 'det' and
                                  box['box_pred_class_id'] >= cl)])

        #Candidate boxes' scores (only class >= cl)
        b_cand_scores = np.array([box['box_score']*class_weights[box['box_pred_class_id']]
                                  for box in b_boxes_list if
                                  (box['box_type'] == 'det' and
                                   box['box_pred_class_id'] >= cl)])
        #Candidate boxe's class (only class >= cl)
        b_cand_class = np.array([box['box_pred_class_id'] for box in b_boxes_list if
                                  (box['box_type'] == 'det' and
                                   box['box_pred_class_id'] >= cl)])

        #Match boxes according to match_iou, if there is at least one gt and one det
        if not 0 in b_cand_boxes.shape and not 0 in b_tar_boxes.shape:
            if not use_distance:
                overlaps = compute_overlaps(b_cand_boxes, b_tar_boxes)
                matches = overlaps
            else:
                distances= compute_distances(b_cand_boxes, b_tar_boxes, 
                                             spacing, normalize_by_size=normalize_by_size)
                matches = 1 - distances / distance_threshold
                matches[matches < 0]= 0
                
            #For every column (GT lesion)
            for i_gt, gt_col in enumerate(matches.T):
                cand_idx= np.argwhere(gt_col)[:,0]
                if len(cand_idx)!= 0:
                    i_cand= cand_idx[np.argmax(b_cand_scores[cand_idx] + gt_col[cand_idx] * 2)]
#                     if pid == 'ProstateX-0139': 
#                         print(gt_col, cand_idx, i_cand, b_cand_scores[i_cand], b_cand_class[i_cand])

                    #Update dataframe
                    patients_test.loc[(pid, i_gt), score_col]= b_cand_scores[i_cand]
                    if class_col is not None:
                        patients_test.loc[(pid, i_gt), class_col]= b_cand_class[i_cand]

    #Drop lesions which do not exist in the actual images
    if drop_nans:
        patients_test= patients_test[~patients_test.ClinSig2.isna()]
        
    return patients_test


def composite_score(y, yp, t=0.5):
    '''
        Returns AUC, accuracy, sensitivity, and specificity at some threshold `t`
        
        Parameters
        ----------
        y: array or list
            List of Ground Truth values (0 or 1)
        yp: array or list
            List of predicted scores in range 0-1
        t: float, default 0.5
            Threshold to evaluate predictions
            
        Returns
        -------
        AUC: float
        accuracy: float
        sensitivity: float
        specificity: float
    '''
    y= np.array(y); yp= np.array(yp)
    tn, fp, fn, tp = confusion_matrix(y, yp > t).ravel()
    return roc_auc_score(y, yp), (tn + tp) / (tn + tp + fn + fp), tp / (tp+fn), tn / (tn+fp)

def plot_auc(y, yp, new_fig=True, legend='ROC-', annotate='auto', is_point=False, 
             step_like=True, i=0, offset=0, save_name='', 
             markers=['s', 'D', '^', '<', 'v', '>', 'o', 'P'],
             colors= list(mcolors.TABLEAU_COLORS)[1:],
             lines=['--', '-']):
    '''
        Plots the ROC curve or a single ROC point
        
        Parameters
        ----------
        y: array or list
            List of Ground Truth values (0 or 1)
        yp: array or list
            List of predicted scores in range 0-1
        new_fig: bool, default True
            Use `new_fig=True` for the first plot, and `False` for the following for htem to appear
            in the same figure
        legend: str, default 'ROC-'
            Name for the plotted line, to be added to the legend
        is_point: bool, default True 
            Assume inputs to be boolean (rather than a score) and plot them as a single point each
        step_like: bool, default True
            Use a step plot if True, otherwise points will be joined by diagonal lines
        save_name: str, default '' 
            If different to '', save the figure using the provided `save_name`
        annotate: bool or 'auto', default True
            If true, add the thresolds to the ROC line
            
        The rest of the inputs are mostly cosmetical, and depend on the item number provided (`i`)
    '''    
    fpr, tpr, thresh = roc_curve(y, yp)
    roc_auc = auc(fpr, tpr)

    if new_fig:
        plt.figure(figsize=(8,5))
        #plt.grid(True)

    if not is_point:
        if step_like:
            plt.step(fpr, tpr, where='mid', color= colors[i], ls=lines[i],
                     lw=1, label='%s (AUC = %0.3f)' % (legend, roc_auc))
        else:
            plt.plot(fpr, tpr, lw=1, ls=lines[i], color= colors[i],
                     label='%sAUC = %0.4f' % (legend, roc_auc))
            
        if (annotate == 'auto' and new_fig) or annotate == True:
            for t,x,y in zip(thresh[offset::2], fpr[offset::2], tpr[offset::2]) :
                plt.annotate('%.2f'%t, (x+.005,y-0.04))
    else:
        tn, fp, fn, tp = confusion_matrix(y, yp > 0.5).ravel()
        sens, spec=  tp / (tp+fn), tn / (tn+fp)
        plt.plot(1-spec, sens, color= colors[i], marker= markers[i], label='%s'%legend, ms=10)
        
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    
    if save_name != '':
        plt.gcf().savefig(save_name, dpi=300)
    
def plot_patient(patient_results, data_source, N_classes= 2, min_threshold= 0.025,
                 ignore_classes= [], save_as=None, class_labels={1:'GGG0', 2:'GGG1', 3:'GGG2', 4:'GGG3+'},
                 plot_max_per_class=100, class_colors= {1:'tab:green', 2:'tab:red', 3:'tab:blue', 4:'tab:orange'},
                 spacing=(0.5, 0.5, 3), seg_threshold=0.5, **kwargs):
    '''
        Plots the GT and the predictions for a patient side by side using `plot_lib`
        
        Parameters
        ----------
        patient_results
            List of results provided by the model for a given patient
        data_source
            Directory where files PID_rois.npy and PID_img.npy are stored
        N_classes: int, default 2
            Number of classes expected
        min_threshold: int, default 0.025
            Detections with a confidence score below this threshold are not plotted
        ignore_classes: list, default []
            Classes to not plot
        save_as: str or None, default None
            Path to save the image (or None to not save anything)
        class_labels: dict, default {1:'GGG0', 2:'GGG1', 3:'GGG2', 4:'GGG3+'}
            Dictionary of the labels associated with each class
        plot_max_per_class: int, default 100
            Limit the number of lesions per class to the top N highest-scoring
        class_colors: dict, default {1:'tab:green', 2:'tab:red', 3:'tab:blue', 4:'tab:orange'}
            Dictionary of the colors associated with each class
        spacing: list or tuple, default (0.5, 0.5, 3)
            Voxel spacing of the images
        seg_threshold: boolean, default 0.5
            Threshold at which to plot the segmentation masks
        **kwargs
            Dictionary of arguments to pass to `plot` from `plot_lib`
    '''
    #Extract useful data from patient_results
    pid, boxes, seg_preds= patient_results[1], patient_results[0]['boxes'][0], patient_results[0]['seg_preds']
    seg_p= np.transpose(seg_preds[0,], axes=(2,0,1))

    #Get first N det boxes sorted by score
    boxes_det= [sorted( [b for b in boxes if 
                             b['box_type'] == 'det' and b['box_pred_class_id'] == cl and b['box_score'] > min_threshold
                          ], key= lambda b: b['box_score'], reverse=True)[:plot_max_per_class] 
                            for cl in [cl for cl in range(1,N_classes + 1) if cl not in ignore_classes] ]
    #Get all gt boxes unsorted
    boxes_gt= [[b for b in boxes if b['box_type'] == 'gt' and b['box_label'] == cl] for cl in range (1,N_classes + 1)]

    def prepare_boxes_for_plotting(boxes):
        #See batchgenerators > augmentations > utils > convert_seg_to_bounding_box_coordinates
        #Original: [y1, x1, y2, x2, (z1), (z2)].
        #Out: [xmin, xmax, ymin, ymax, zmin, zmax] + [color, text, zorder]
        boxes_list= []
        for box in boxes:
            c= box['box_coords']
            box_label= box['box_label' if box['box_type'] == 'gt' else 'box_pred_class_id']
            boxes_list.append([
                    c[1], c[3], c[0], c[2], c[4], c[5],
                    class_colors[box_label], 
                    class_labels[box_label] + \
                    ('\n%.3f'%box['box_score'] if box['box_type'] == 'det' else ''),
                     2 if box['box_type'] == 'gt' else 1])
#             for c, i in zip(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] + \
#                             ['color', 'text', 'zorder'], boxes_list[-1]): 
#                 print(c,i,'| ')
        return boxes_list

    boxes_gt_prepared= prepare_boxes_for_plotting(sum(boxes_gt, []))
    boxes_det_prepared= prepare_boxes_for_plotting(sum(boxes_det, []))

    #Load image
    img = np.load(os.path.join(data_source, '{}_img.npy'.format(pid)))
    seg = np.load(os.path.join(data_source,'{}_rois.npy'.format(pid)))

    #Plot
    plot4(img, masks=[ [seg[...,0]>seg_threshold, [1], ['tab:orange']], 
                       [img[...,-3], [1], ['tab:blue']], [img[...,-2], [1], ['tab:red']] ], 
                       boxes=boxes_gt_prepared, spacing=spacing, title='GT', **kwargs, 
                       save_as= (save_as + '_gt.png') if save_as is not None else None)
    plot4(img, masks=[ [seg_p > seg_threshold, [1], ['tab:orange']], 
                       [img[...,-3], [1], ['tab:blue']], [img[...,-2], [1], ['tab:red']] ], 
                       boxes=boxes_det_prepared, spacing=spacing, title='Predicted', **kwargs, 
                       save_as= (save_as + '_pred.png') if save_as is not None else None)
                       
                   

# The funtions below have been taken from model_utils.py, with Apache license
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
              
def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)]. / 3D: (z1, z2))
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    if boxes1.shape[1] == 4:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i] #this is the gt box
            overlaps[:, i] = compute_iou_2D(box2, boxes1, area2[i], area1)
        return overlaps

    else:
        # Areas of anchors and GT boxes
        volume1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 4])
        volume2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 4])
        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i]  # this is the gt box
            overlaps[:, i] = compute_iou_3D(box2, boxes1, volume2[i], volume1)
        return overlaps
    
def compute_iou_3D(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2, z1, z2] (typically gt box)
    boxes: [boxes_count, (y1, x1, y2, x2, z1, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    z1 = np.maximum(box[4], boxes[:, 4])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / union

    return iou
