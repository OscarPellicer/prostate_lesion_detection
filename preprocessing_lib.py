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
    This file contains all functions needed to provide funtionality to `ProstateX preprocessing.ipynb`
'''

import sys, os, subprocess, glob, pickle
from datetime import datetime

import numpy as np
import SimpleITK as sitk
import pandas as pd
import pydicom
from scipy.ndimage.morphology import binary_dilation

def info(image, show='all'):
    '''
        Prints information about a SimpleITK or a numpy image
        
        Parameters
        ----------
        image: SimpleITK Image or array
            Image to obtain information about. SimpleITK images show much more information
        show: str, any of ['all', 'size', 'origin', 'spacing', 'direction', 
            'channels', 'type', 'range'], default 'all'
            What information to show ('all') to show all
    '''
    if not isinstance(image, (np.ndarray, np.generic) ):
        if show in ['all']: print('SITK image info:')
        if show in ['all', 'size']: print(' - Size:', np.array(image.GetSize()))
        if show in ['all', 'origin']: print(' - Origin:', np.array(image.GetOrigin()))
        if show in ['all', 'spacing']: print(' - Spacing:', np.array(image.GetSpacing()))
        if show in ['all', 'direction']: print(' - Direction:', np.array(image.GetDirection()))
        if show in ['all', 'channels']: print(' - Components per pixel:', 
                                              np.array(image.GetNumberOfComponentsPerPixel()))
        if show in ['all', 'type']: print(' - Pixel type:', image.GetPixelIDTypeAsString())
        img_arr= sitk.GetArrayViewFromImage(image)
    else:
        print('Numpy image info:')
        print(' - Shape:', image.shape)
        print(' - Pixel type:', image.dtype)
        img_arr= image
        
    unique= np.unique(img_arr)
    if show in ['all', 'range']: print(' - Min/max:', np.min(img_arr), np.max(img_arr))
    if len(unique) < 50 and show in ['all', 'unique']: print(' - Unique values:', unique)
        

class ImageList():
    '''
        Very basic image loading class. It allows adding images sequentially to the dataset.
    '''
    def __init__(self):
        self.IDs= []
        self.IMAGES= []
        
    def add_dicom_series(self, img_path, ID, N=1, names=[], reverse=False,
                         interleave=False):
        '''
            Adds an image given a DICOM path.
            
            Parameters
            ----------
            img_path: str
                Path where DICOM slices are stored
            ID: str
                Patient ID
            N: int, default 1
                Number of actual images contained within the `img_path`
            names: list of str, default []
                List of DICOM sclies in the order that they should be read. If left empty ([]), 
                there will be an attempt to read them automatically
            reverse: bool, default False
                Read the names in reverse / reverse z-axis of read images
            interleave: bool, default False
                If False and N>1, assume DICOM slices for images A,B,C are provided as AAABBBCCC
                If True and N>1, assume DICOM slices for images A,B,C are provided as ABCABCABC
        '''
        if N==1 and names==[]:
            #Read image
            reader = sitk.ImageSeriesReader()
            names= reader.GetGDCMSeriesFileNames(img_path)
            #print([os.path.split(name)[-1].split('.')[-2] for name in names])
            reader.SetFileNames(names)
            img = reader.Execute()
            self.add_image(img, ID, is_path=False)
        else:
            reader = sitk.ImageSeriesReader()
            if N>1:
                names= reader.GetGDCMSeriesFileNames(img_path) if names == [] else names
                M= len(names)
                S= M//N #Number of slices per image
                if S != M/N: 
                    raise RuntimeError('All images must have the same number of slices. '+\
                                       'Please, provide the names directly')
                for n in range(N):
                    names_n= names[n*S:(n+1)*S] if not interleave else names[n::S]
                    names_n= names_n[::-1] if reverse else names_n
                    reader.SetFileNames(names_n)
                    img = reader.Execute()
                    self.add_image(img, ID + '_%d'%n, is_path=False)
            elif names != []:
                for n, names_n in enumerate(names):
                    names_n= [os.path.join(img_path, n) for n in names_n]
                    reader.SetFileNames(names_n[::-1] if reverse else names_n)
                    img = reader.Execute()
                    self.add_image(img, ID + '_%d'%n, is_path=False)
    
    def add_image(self, img_path, ID, is_path=True): 
        '''
            Adds an image given either its path or the actual SimpleITK Image
            
            Parameters
            ----------
            img_path: str or SimpleITK Image
                Path to the image, or SimpleITK Image
            ID: str
                Patient ID
            is_path: bool, default True
                If False, interpret the `img_path` as a SimpleITK image
        '''
        if is_path:        
            img = sitk.ReadImage(img_path)
        else:
            img= img_path
        
        self.IMAGES.append(img)
        self.IDs.append(ID)
    
def resampling_operation(img, mask, spacing=(0.5, 0.5, 3), size='auto',
                         transform=[], img_interpolator=sitk.sitkBSpline,
                         label_interpolator= sitk.sitkLabelGaussian,
                         pre_mask_growth_mm=-1, pre_mask_growth_mm_channels=[],
                         per_channel_transform=sitk.Euler3DTransform(), transform_channels=[]):
    '''
        Resample an image and a mask associated with that image
        
        Parameters
        ----------
        img: SimpleITK Image or []
            Image to be resampled, or [] if no image is to be provided
        mask: SimpleITK Image or []
            Image to be resampled, or [] if no mask is to be provided
        spacing: list or tuple of 3 floats, default (0.5, 0.5, 3)
            Output spacing in mm
        size: list of ints or 'auto', default 'auto'
            Size of the final image after resizing, or 'auto' to infer
        transform: SimpleITK Transform or [], default []
            Transform to apply to the whole `image` and the `mask`. 
            It can also be [] to not transform anything
        img_interpolator: SimpleITK Interpolator, default sitk.sitkBSpline
            Interpolator to be used with `img`
        label_interpolator: SimpleITK Interpolator, default sitk.sitkLabelGaussian
            Interpolator to be used with `mask`
        pre_mask_growth_mm: int, default -1
            If pre_mask_growth_mm > 0, we will grow the mask before resampling, and
            then reduce it afterwards, so that mask lesions or other small masks are 
            better preserved
        pre_mask_growth_mm_channels: list of ints, default []
            List of channels of the `mask` to which `pre_mask_growth_mm` will be applied
        per_channel_transform: SimpleITK Transform
            Transform to apply to specific channels, provided in `transform_channels`.
         pre_mask_growth_mm_channels: list of ints, default []
            List of channels of the `image` to which `per_channel_transform` will be applied
            
        Returns
        -------
        img_r: SimpleITK Image or []
        mask_r: SimpleITK Image or []
    '''
    #Set reference
    ref= img if img!=[] else mask
    
    #(sitkBSpline, sitkNearestNeighbor, sitkLinear, sitkCosineWindowedSinc)
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetOutputSpacing(spacing)
    resample.SetInterpolator(img_interpolator)
    #resample.SetDefaultPixelValue(0.)	#It is 0 by default

    if str(size)=='auto':
        orig_size = np.array(ref.GetSize(), dtype=np.int)
        orig_spacing = ref.GetSpacing()
        new_size = orig_size*(np.array(orig_spacing)/np.array(resample.GetOutputSpacing()))
        new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
        new_size = [int(s) for s in new_size]
        resample.SetSize(new_size)
    else:
        size= [int(s) for s in size]
        resample.SetSize(size)
        
    #Resample image
    if img!=[]:
        if transform!=[]:
            resample.SetTransform(transform)
        img_r= resample.Execute(img)

        #Resample other channels differently
        if transform_channels != []:
#             channel_tranform = sitk.CompositeTransform([transform, per_channel_transform])
            channel_tranform = sitk.Transform(3, sitk.sitkComposite)
            channel_tranform.AddTransform(transform)
            channel_tranform.AddTransform(per_channel_transform)
            
            channel_resample = sitk.ResampleImageFilter()
            channel_resample.SetReferenceImage(ref)
            channel_resample.SetOutputSpacing(spacing)
            channel_resample.SetSize(resample.GetSize())
            channel_resample.SetInterpolator(img_interpolator)
            channel_resample.SetTransform(channel_tranform)
            
            imgs_c= []
            for c in range(img.GetNumberOfComponentsPerPixel()):
                if c in transform_channels:
                    imgs_c.append(channel_resample.Execute(sitk.VectorIndexSelectionCast(img, c)))
                else:
                    imgs_c.append(sitk.VectorIndexSelectionCast(img_r, c))
            img_r= join_sitk_images(imgs_c, resample=False)
    else:
        img_r= []
        
    #Resample mask
    resample.SetInterpolator(label_interpolator)
    if mask != []:
        if pre_mask_growth_mm > 0:
            print(' - Information: resampling_operation with pre_mask_growth_mm applied to channels '+  
                  '%s on x,y dimensions'%pre_mask_growth_mm_channels)
            mask_r_list= []
            orig_spacing = np.array(ref.GetSpacing())
            new_spacing= np.array(spacing)
            radius= np.array([pre_mask_growth_mm, pre_mask_growth_mm, 0])
            for c in range(mask.GetNumberOfComponentsPerPixel()):
                m= sitk.VectorIndexSelectionCast(mask, c)
                
                from plot_lib import plot
                if c in pre_mask_growth_mm_channels:
                    #plot(m)
                    m = sitk.BinaryDilate(m, np.round(radius/orig_spacing).astype(np.uint32).tolist())
                    #plot(m)
                m = resample.Execute(m)
                if c in pre_mask_growth_mm_channels: 
                    #plot(m)
                    m = sitk.BinaryErode(m, np.round(radius/new_spacing).astype(np.uint32).tolist())
                    #plot(m)
                
                mask_r_list.append(m)
            #mask_r= sitk.ComposeImageFilter().Execute(*mask_r) #Only up to five images
            mask_r= sitk.GetImageFromArray(np.stack([sitk.GetArrayFromImage(m) for m in mask_r_list], 
                                                    axis=-1), isVector=True)
            mask_r.CopyInformation(mask_r_list[0])
        else:
            mask_r = resample.Execute(mask)
    else:
        mask_r= []
    
    return img_r, mask_r

def center_image(img, mask, size=(160,160,32), spacing=(1,1,3), center_around_roi=True, **kwargs):
    
    '''
        Centers an image around either its ceter, or a ROI, and cuts it into `size`
        
        Parameters
        ----------
        center_around_roi: bool, default True
            If True, center the `img` around the centroid of the `mask`
            If False, center the `img` around its central voxel
            
        **See `resampling_operation` for further information**
                    
        Returns
        -------
        img: SimpleITK Image or []
        mask: SimpleITK Image or []
    '''
    
    #Reset image properties
    img.SetOrigin((0,)*3)
    mask.SetOrigin((0,)*3)
    img.SetDirection(np.eye(3).flatten())
    mask.SetDirection(np.eye(3).flatten())
     
    spacing_orig= img.GetSpacing()
    size_orig= img.GetSize()
        
    #Set up a shift to center the downscaled image
    if not center_around_roi:
        offset= [ int((SZ*SP-sz*sp)/2) for SP, SZ, sp, sz in zip(
                    spacing_orig, size_orig, spacing, size)]
    else:
         #Get centroid
        if mask.GetNumberOfComponentsPerPixel() > 1:
            ma_centroid= sitk.VectorIndexSelectionCast(mask, 0) > 0.5
        else:
            ma_centroid= mask > 0.5
        label_analysis_filer= sitk.LabelShapeStatisticsImageFilter()
        label_analysis_filer.Execute(ma_centroid)
        centroid= label_analysis_filer.GetCentroid(1)
        offset_correction= np.array(size)*np.array(spacing)/2
        offset= np.array(centroid)-np.array(offset_correction)
                
    translation = sitk.TranslationTransform(3, offset)
    img, mask= resampling_operation(img, mask, spacing=spacing, size=size, 
                                    transform=translation, **kwargs)
    
    return img, mask

def rescale_intensity(image, thres=(1.0, 99.0), method='noclip'):
    '''
        Rescale the image intensity using several possible ways
        
        Parameters
        ----------
        image: array
            Image to rescale
        thresh: list of two floats between 0. and 1., default (1.0, 99.0)
            Percentiles to use for thresholding (depends on the `method`)
        method: str, one of ['clip', 'mean', 'median', 'noclip']
            'clip': clip intensities between the thresh[0]th and the thresh[1]th
            percentiles, and then scale between 0 and 1
            'mean': divide by mean intensity
            'meadin': divide by meadian intensity
            'noclip': Just like 'clip', but wihtout clipping the extremes
            
        Returns
        -------
        image: array
    '''
    eps= 0.000001
    def rescale_single_channel_image(image):
        #Deal with negative values first
        min_value= np.min(image)
        if min_value < 0:
            image-= min_value
        if method == 'clip':
            val_l, val_h = np.percentile(image, thres)
            image2 = image
            image2[image < val_l] = val_l
            image2[image > val_h] = val_h
            image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l + eps)
        elif method == 'mean':
            image2= image / max(np.mean(image),1)
        elif method == 'median':
            image2= image / max(np.median(image),1)
        elif method == 'noclip':
            val_l, val_h = np.percentile(image, thres)
            image2 = image
            image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l + eps)
        else:
            image2= image
        return image2
    
    #Process each channel independently
    if len(image.shape) == 4:
        for i in range(image.shape[-1]):
            image[...,i]= rescale_single_channel_image(image[...,i])
    else:
        image= rescale_single_channel_image(image)
        
    return image

def join_sitk_images(images, resample=True, verbose=True, 
                     resampler=sitk.sitkCosineWindowedSinc,
                     cast_type=None, transform_channels=[], transform=sitk.Euler3DTransform()):
    '''
        Joins images from different modalities into a single multichanel image, assuming that
        the first image on the list is the reference.
        
        Parameters
        ----------
        images: list of SimpleITK Images
            Images to join, the first one being the reference
        resample: bool, default True
            Resample the images? Shoul be True unless they are already resampled
        vebose: bool, default True
            Show some information
        resampler: SimpleITK Interpolator, default sitk.sitkCosineWindowedSinc
            Interoplator for resampling
        cast_type: SimpleITK Type or None, default None
            if not None, type to cast the images to before resampling
        transform_channels: list ints, default []
            List of image indices to apply `transform` to
        transform: SimpleITK Transform, default sitk.Euler3DTransform()
            Transform to apply to the image indices given by `transform_channels`
            
        Returns
        -------
        image_final_sitk: SimpleITK Image
    '''
    #First, set them in the frame of reference of the first image
    image_ref= images[0]
    
    #Cast images if required
    if cast_type is not None:
        images= [sitk.Cast(im, cast_type) for im in images]
    
    if resample:
        #Resample all images into the reference
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(image_ref)
        resample.SetOutputSpacing(image_ref.GetSpacing())
        resample.SetSize(image_ref.GetSize())
        resample.SetInterpolator(resampler) #sitkCosineWindowedSinc

        images_res= []
        if verbose: 
            print('Combining %d images: '%len(images), end='')
            import time
            start_time = time.time()
            
        for i,img in enumerate(images[1:]):
            if verbose: print('#', end='')
            resample.SetTransform(transform if i+1 in transform_channels else sitk.Euler3DTransform())
            img_res= resample.Execute(img)
            images_res.append(img_res)
            
        if verbose: print(' -> Elapsed: %.2fs'%(time.time() - start_time))
    else:
        images_res= images[1:]
    
    #All to numpy
    images_arr= [sitk.GetArrayFromImage(img) for img in ([image_ref] + images_res) ]
    
    #Stack
    image_final= np.stack(images_arr, axis=-1)
    
    #Get sitk image
    image_final_sitk= sitk.GetImageFromArray(image_final, isVector=True)
    image_final_sitk.CopyInformation(image_ref)
    
    return image_final_sitk

def get_blank_image(image):
    '''
        Obtains a blank (all zeros) image with same properties as `image`
        
        Parameters
        ----------
        image: SimpleITK Image
            Reference image
        
        Returns
        -------
        blank_image: SimpleITK Image
    '''
    if image.GetNumberOfComponentsPerPixel() > 1:
        ref_blank_image= sitk.VectorIndexSelectionCast(image, 0)
    else:
        ref_blank_image= image
    blank_image= np.zeros_like(sitk.GetArrayFromImage(ref_blank_image))
    blank_image= sitk.GetImageFromArray(blank_image)
    blank_image.CopyInformation(ref_blank_image)
    return blank_image

def join_masks(prostate_mask, lesions_mask, mode='append', 
               max_lesions=1000, reassign_lesion_IDs=False):
    '''
        Join masks from mask images `prostate_mask` & `lesions_mask`. Note that despite 
        the name of the variables, it can be used for other kinds of masks. 
        
        Parameters
        ----------
        prostate_mask: SimpleITK Image
            First mask to join
        lesions_mask: SimpleITK Image
            Second mask to join
        mode: str, one of ['append', 'combine'], default 'append'
            'combine': The resulting mask has a single channel, and all the masks occupy 
            the same number ID
            'append':  The resulting mask has one more channel, where the that last channel
            contains the `lesions_mask` with their respective IDs
        max_lesions: int, default 1000
            Lesions above this number will be removed
        reassign_lesion_IDs: bool, default False
            Reassign the IDs of the lesions to consecutive numbers
            
        Returns
        -------
        prostate_mask_new: SimpleITK Image
    '''
    #To numpy
    prostate= sitk.GetArrayFromImage(prostate_mask)
    lesions= sitk.GetArrayFromImage(lesions_mask)
    
    #Convert randomly assigned lesion IDs to IDs such as [1,2,..]
    if reassign_lesion_IDs:
        ids= np.unique(lesions)[1:] #Ignore BG
        for i, id in enumerate(ids): lesions[lesions==id]= i+1
    
    if mode=='combine':
        prostate[(lesions > 0) & (prostate > 0) ]= lesions + 1
    elif mode=='append':
        lesions[lesions > max_lesions] = 0 #Limit the number of saved lesions to 3
        
        #Some checks so that it does not throw an error if an image has multiple channels
        prostate = prostate if len(prostate.shape)==4 else prostate[...,np.newaxis]
        lesions = lesions if len(lesions.shape)==4 else lesions[...,np.newaxis]
        
        #Join
        prostate= np.concatenate([prostate, lesions], axis=-1)
    else:
        raise Exception('Unknown mode: %s'%mode)

    prostate_mask_new= sitk.GetImageFromArray(prostate, isVector= mode=='append')
    prostate_mask_new.CopyInformation(prostate_mask)

    return prostate_mask_new
    
def grow_regions_sitk(image, seg, clean=True, factor= 2.5, 
                      iters_threshold=150, error_threshold=0.015):
    '''
        Grow masks from a seed segmentation mask by following the intensities of
        the provided `image`, by using sitk.ThresholdSegmentationLevelSetImageFilter()
        
        Parameters
        ----------
        image: SimpleITK Image
            Reference image to guide the growing process
        seg: SimpleITK Image
            Segmentation mask with the seeds to grow from
        clean: bool, default True
            If True, apply morphological binary closing operation to the resulting image
        factor: float, default 2.5
            Parameter that controls how big of an area the algorithm originally considers
        iters_threshold: int, default 150
            Maximum number of iteretions to grow the image for
        error_threshold: float, default 0.015
            If the RMS error of the growing goes below this value, stop
            
        Returns
        -------
        mask: SimpleITK Image
    '''
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(image, seg)

    lower_threshold = stats.GetMean(1)-factor*stats.GetSigma(1)
    upper_threshold = stats.GetMean(1)+factor*stats.GetSigma(1)

    init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)

    lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
    lsFilter.SetLowerThreshold(lower_threshold)
    lsFilter.SetUpperThreshold(upper_threshold)
    lsFilter.SetMaximumRMSError(error_threshold)
    lsFilter.SetNumberOfIterations(iters_threshold)
    lsFilter.SetCurvatureScaling(0.0)
    lsFilter.SetPropagationScaling(1)
    lsFilter.ReverseExpansionDirectionOn()
    ls = lsFilter.Execute(init_ls, sitk.Cast(image, sitk.sitkFloat32))

    mask= ls>0
        
    #Clean up the resulting segmentation
    if clean:
        mask = sitk.BinaryMorphologicalClosing(mask, [1]*3, sitk.sitkBall)
        
    return mask


def read_prostatex_all_modalities(images_path, ktrans_path, ID,
                                  sequence_reduced, sequence_equivalences, 
                                  verbose=True, testing=False):
    '''
        Reads all sequences from a ProstateX patient given its `images_path`, its `ktrans_path
        and its `ID`. This is highly specialized function, which deals with all the specifics
        related to this dataset in particular, although it could be adopted to other datasets
        if required. It is meant to be called from `read_prostatex_patient` function
    '''
    
    if testing: 
        from plot_lib import plot
    
    #Read first image from first series to identify series' name
    first_series_path= os.path.join(images_path, os.listdir(images_path)[0])
    dicom_files= glob.glob(os.path.join(first_series_path,'*.dcm'))
    ds_gen= pydicom.dcmread(dicom_files[0])
    date= str(pd.to_datetime(ds_gen.StudyDate))
    if verbose: print('Patient: %s (date: %s)'%(ID, date))
    
    #Create dictionary to store images
    images={k:[] for k in sequence_reduced}
    fix_b_data= False
                
    #Read all DICOM images
    for img_path in os.listdir(images_path):
        try:
            #We use a new dataset each time only to leverage its image-reading functionality
            ds_ns= ImageList()

            #Load image info
            if verbose: print('\t- Reading: %s '%img_path, end='')        
            full_path= os.path.join(images_path, img_path)
            dicom_files= glob.glob(os.path.join(full_path,'*.dcm'))
            ds = pydicom.dcmread(dicom_files[0])
            
            #Print some info
            name= sequence_equivalences[ds.SeriesDescription] \
                if ds.SeriesDescription in sequence_equivalences else 'UNKNOWN'
            print(' (%s) (%s)'%(ds.SeriesDescription, name))
            
            #Keep only interesting series
            if not testing:
                if ds.SeriesDescription not in sequence_equivalences or \
                   sequence_equivalences[ds.SeriesDescription] not in sequence_reduced:
                    if verbose: print('\t\t- Skipping!')
                    continue
            elif 'dynamisch' in ds.SeriesDescription or 'Perfusie' in ds.SeriesDescription:
                continue
                        
            #Read image
            #b-values
            if ds.SeriesDescription in ['ep2d_diff_tra_DYNDIST', 'ep2d_diff_tra_DYNDIST_MIX', 
                                        'diffusie-3Scan-4bval_fs', 'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen',
                                        'diff tra b 50 500 800 WIP511b alle spoelen']:
                #We must handle several cases
                if ds.SeriesDescription == 'diffusie-3Scan-4bval_fs':
                    b_values= {50:[], 500:[], 800:[], }
                    fix_b_data= True
                elif ds.SeriesDescription == 'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen':
                    b_values={50:[], 500:[], 800:[], 1400:[]}
                    fix_b_data= True
                elif ds.SeriesDescription == 'diff tra b 50 500 800 WIP511b alle spoelen':
                    b_values={50:[], 500:[], 800:[]}
                    fix_b_data= True
                else:
                    b_values={50:[], 400:[], 800:[]}
                
                #Fortunately, all dicoms have a similar internal structure
                for dcm in dicom_files: 
                    dcm_ds= pydicom.dcmread(dcm)
                    b= int(dcm_ds[(0x19, 0x100c)].value) #b value
                    b_values[b].append(os.path.split(dcm)[-1])
                names= [v for k,v in b_values.items()]
                ds_ns.add_dicom_series(full_path, img_path, names=names, reverse=True)

            #General case
            else:
                ds_ns.add_dicom_series(full_path, img_path)

            #If testing is on, plot images
            if testing:
                for img, idd in zip(ds_ns.IMAGES, ds_ns.IDs):
                    info(img)
                    plot(img, title=idd)

            #Save images in dictionary
            images[sequence_equivalences[ds.SeriesDescription]]+= ds_ns.IMAGES
            
            if verbose: print('\t\t- Done!')

        except Exception as e:
            print(e)
    
    #Read ktrans image
    try:
        mhd_path= glob.glob(os.path.join(ktrans_path,'*.mhd'))[0]
        print('\t- Reading: %s (%s) (%s)'%(os.path.split(mhd_path)[-1], 'Ktrans', 'ktrans'))

        ds_ns= ImageList()
        ds_ns.add_image(mhd_path, ID)
        images['ktrans']= ds_ns.IMAGES
    except Exception as e:
        print(e)
    
    if testing: 
        info(images['ktrans'][0])
        plot(images['ktrans'][0], title='Ktrans')
    if verbose: print('\t\t- Done!')
        
    return images, fix_b_data

def read_prostatex_patient(ID, dicom_path, ktrans_path, verbose=True):
    '''
        Reads all sequences from a ProstateX patient given its `dicom_path`, its `ktrans_path
        and its `ID`. This is highly specialized function, which deals with all the specifics
        related to this dataset in particular, although it could be adopted to other datasets
        if required.
                
        Parameters
        ----------
        ID: str
            ID of the patient
        dicom_path: str
            Path to the directory where all the DICOM directories are stored for a given patient
        ktrans_path: str
            Path to the directory where all ktrans images are sotred for all patients
        verbose: bool, default True
            Print some information about the reading process of the images
            
        Returns
        -------
        images_list: list of SimpleITK Images
    '''
    
    #Define the sequences to be read, and the DICOM name-sequence equivalences
    sequence= ['T2', 'b400', 'b800', 'ADC', 'ktrans']
    sequence_reduced= ['T2', 'b', 'ADC', 'ktrans']
    sequence_equivalences= {
        't2_tse_tra': 'T2', 'ep2d_diff_tra_DYNDIST_ADC': 'ADC', 'ep2d_diff_tra_DYNDIST': 'b',
        'tfl_3d PD ref_tra_1.5x1.5_t3': 'unk', 'ep2d_diff_tra_DYNDISTCALC_BVAL': 'diff',
        'ep2d_diff_tra_DYNDIST_MIX': 'b', 'ep2d_diff_tra_DYNDIST_MIX_ADC': 'ADC',
        'ep2d_diff_tra_DYNDIST_MIXCALC_BVAL': 'diff', 'diffusie-3Scan-4bval_fs': 'b', 
        'diffusie-3Scan-4bval_fs_ADC': 'ADC', 'ep2d-advdiff-MDDW-12dir_spair_511b_ADC': 'ADC',
        'ep2d-advdiff-3Scan-4bval_spair_511b_ADC': 'ADC', 
        'ep2d-advdiff-3Scan-high bvalue 100': 'b',
        'ep2d-advdiff-3Scan-high bvalue 500': 'b',
        'ep2d-advdiff-3Scan-high bvalue 1400': 'b',
        'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen': 'b',
        'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen_ADC': 'ADC',
        'diff tra b 50 500 800 WIP511b alle spoelen': 'b',
        'diff tra b 50 500 800 WIP511b alle spoelen_ADC': 'ADC',
        't2_tse_tra_320_p2': 'T2',
        'ADC_S3_1': 'ADC',
        'ep2d_diff_tra2x2_Noise0_FS_DYNDISTCALC_BVAL': 'b',
        't2_tse_tra_Grappa3': 'T2',
    }
    
    images, fix_b_data= read_prostatex_all_modalities(dicom_path, os.path.join(ktrans_path, ID), 
                                      ID, sequence_reduced, sequence_equivalences, 
                                      verbose=verbose, testing=False)
    if verbose: print('\n\t\t> ', end='')
    images_list= []
    for s, v in images.items():
        if s in ['ADC', 'unk']: 
            img_arr= sitk.GetArrayFromImage(v[0])
            img_arr[img_arr < 0]= 0
            img= sitk.GetImageFromArray(img_arr)
            img.CopyInformation(v[0]) 

            #Blur if image has intensity problem
            if np.percentile(img_arr, 40) == 0:
                if verbose: print('(Blurrying ADC) ', end='')
                gaussian= sitk.DiscreteGaussianImageFilter()
                gaussian.SetVariance((2,)*3)
                gaussian.SetMaximumKernelWidth(16)
                gaussian.SetUseImageSpacing(True)
                img= gaussian.Execute(img)
            images_list.append(img)

        elif s in ['b']:
            if len(v) > 2:
                v=v[1:3]
            elif ID == 'ProstateX-0191':  #Exception
                v= [get_blank_image(images_list[0]), v[0]]
            images_list+= v

        else:
            #By default, append latest value (in the other sequences, the first values are kept instead)
            if len(v): 
                if ID == 'ProstateX-0148': #Exception
                    images_list.append(v[0])
                else:
                    images_list.append(v[-1])
            else:
                images_list.append(get_blank_image(images_list[0]))
                if verbose: print('\n - Error: Sequence %s could not be read!'%s)

    if fix_b_data: #Exception
        images_list[sequence.index('b400')].CopyInformation(images_list[sequence.index('ADC')])
        images_list[sequence.index('b800')].CopyInformation(images_list[sequence.index('ADC')])

    if len(images_list) != len(sequence):
        raise ValueError('Expected %d image modalities, found %d'%(len(sequence), len(images_list)))
        
    return images_list

def get_lesion_mask_id_seed(positions_img, mask):
    '''
        Create a mask of seeds from some positions
        
        Parameters
        ----------
        positions_img: list of lists of 3 floats
            List of positions (in voxel coordinates: [z,y,x]) to place the seeds at
        mask: SimpleITK Image
            Reference image to copy properties from for the output mask
        
        Returns
        -------
        lesion_mask_id_sitk: SimpleITK Image
    '''
    lesion_mask_id_seed= np.zeros_like(sitk.GetArrayViewFromImage(mask), dtype=np.uint8) 

    for i, seed in enumerate(positions_img):
        try:
            curr_mask= np.zeros_like(lesion_mask_id_seed)
            coords= np.round(seed[::-1]).astype(np.int)
            curr_mask[coords[0], coords[1], coords[2]]= 1
            curr_mask= binary_dilation(curr_mask, np.ones((3,5,5)))

            lesion_mask_id_seed[curr_mask]= i+1

        except Exception as e:
            print(' - Error growing lesions for seed %s: %s'%(seed, e))

    lesion_mask_id_sitk= sitk.GetImageFromArray(lesion_mask_id_seed)
    lesion_mask_id_sitk.CopyInformation(mask)
    return lesion_mask_id_sitk

def grow_lesions(prostate_mask_intermediate, img_final, significances, transform, 
                 iters_max=120, factors= [2.5,2.5,3.5,4]):
    '''
        Highly specialized funtion for growing seeds from a seed mask in the context
        of mpMRIs of the prostate.
        
        Parameters
        ----------
        prostate_mask_intermediate: SimpleITK Image
           Multichannel mask image, with the channel 0 being a mask of the prostate, and channel 2
           being a seed mask resulting from `get_lesion_mask_id_seed` function
        img_final: SimpleITK Image
            Image to use for growing seeds, must be a mpMRI with ['T2', 'b800', 'ADC', 'ktrans'] in
            channels [0, 2, 3, 4]
        significances: list of ints
            Significances of the lesions
        transform: SimpleITK transform
            The seed mask will be grown form the original lesion positions, but also from the
            positions after applying transform. This had to be done after registering the 
            modalities of the mpMRI, since some lesions were refered to the T2, but others
            to the ADC (which has been registered), and there is no way of knowing to which 
            sequence they were originally assigned of the both
        iters_max: int, default 120
            Maximum number of iterations for the lesion growing algorithm
        factors: list of four floats, default [2.5,2.5,3.5,4]
            Parameter that controls how big of an area the algorithm originally considers for
            growing each of the channels of `img_final`
            
        Returns
        -------
        lesion_mask_id: SimpleITK Image
        lesion_mask_sig: SimpleITK Image
    '''

    #Set lesions far away from the prostate to 0, and save this engrossed prostate mask
    prostate_mask_dilated= sitk.BinaryDilate(
        sitk.VectorIndexSelectionCast(prostate_mask_intermediate, 0), [8, 8, 1])
    prostate_mask_dilated_array= sitk.GetArrayFromImage(prostate_mask_dilated)

    #Perform actual lesion growing
    lesion_mask_id= np.zeros_like(sitk.GetArrayViewFromImage(prostate_mask_intermediate), dtype=np.uint8)[...,0]
    lesion_mask_sig= np.zeros_like(lesion_mask_id)

    for i, sig in enumerate(significances):
        lesion_masks_arr_list= []
        mask= sitk.VectorIndexSelectionCast(prostate_mask_intermediate, 2) == i+1 #2: Lesion ID
        mask_t= sitk.Resample(mask, mask, transform, sitk.sitkLabelGaussian, 0, mask.GetPixelID())
        for mask_to_grow in [mask, mask_t]:
            for c, s, f, e in zip([0, 2, 3, 4], ['T2', 'b800', 'ADC', 'ktrans'], factors, [0.]*4):
                img_c= sitk.VectorIndexSelectionCast(img_final, c)
                mask_c= grow_regions_sitk(img_c, mask_to_grow, factor=f, 
                                          iters_threshold=iters_max, error_threshold=e)
                lesion_masks_arr_list.append(sitk.GetArrayFromImage(mask_c))

        lesion_mask= np.mean([m for m in lesion_masks_arr_list if np.sum(m) > 64], axis=0) > 2.5/8 #At least 3
        if lesion_mask.size == 1: 
            print(' - Error: Lesion mask array is empty: no segmentations were performed')
            lesion_mask= np.zeros_like(lesion_masks_arr_list[0])
        lesion_mask[prostate_mask_dilated_array!=1]= 0
        lesion_mask_sig[lesion_mask]= sig
        lesion_mask_id[lesion_mask]= i+1

    lesion_mask_sig= sitk.GetImageFromArray(lesion_mask_sig.astype(np.uint8))
    lesion_mask_sig.CopyInformation(prostate_mask_intermediate)

    lesion_mask_id= sitk.GetImageFromArray(lesion_mask_id.astype(np.uint8))
    lesion_mask_id.CopyInformation(prostate_mask_intermediate)

    return lesion_mask_id, lesion_mask_sig

class ProgressBar():
    '''
        Very simple progress bar implementation.
        The class must be intialized outside the loop.
        
        Parameters
        ----------
        number_of_iterations: int
            The number of iterations that the loop is going to perform
        MAX_N: int, default 80
            Maximum number of charecters the progress bar takes up in the terminal
            `number_of_iterations` > MAX_N, otherwise the bar will not be completed
    '''
    def __init__(self, number_of_iterations, MAX_N= 80):
        self.M= number_of_iterations
        self.MAX_N= MAX_N
        
        self.prev_value= -1
        print('Progress:\n'+('_'* self.MAX_N))
        
    def go(self, curr_iter):
        '''
            Call this method at each iteration to -possibly- update the progress bar.
            
            Parameters
            ----------
            curr_iter: int
                Current loop iteration
        '''
        if int(curr_iter/self.M * self.MAX_N) != self.prev_value:
            self.prev_value+= 1
            print('#', end='')
                  
            if self.prev_value==self.MAX_N-1:
                print('\n')
                
class EasyTimer():
    '''
        Very simple timer class
    '''
    def __init__(self):
        self.reset()
        
    def time(self, title='Time elapsed'):
        '''
            Prints the elapsed time since the instance was created, or since the 
            method reset() was last called
            
            Parameters
            ----------
            title: str, default 'Time elapsed'
                Message to show alongside the ellpased time
        '''
        self.last= self.current
        self.current= datetime.now()
        print('%s: %s'%(title, self.current - self.last))
        
    def reset(self):
        '''
            Reset the timer
        '''
        self.current= datetime.now()
        self.last= None