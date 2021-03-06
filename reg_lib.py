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
# 
# Much of the code here was taken from https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks
# which is under the Apache 2.0 license, by Insight Software Consortium

'''
    This file contains functions needed to provide funtionality to `Registration example.ipynb`
'''

import os, sys

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from IPython.display import clear_output

class RegistrationTracker():
    '''
        Plots the evolution of the registration loss/metric over time
    '''
    def __init__(self):
        self.start_plot()
    
    def start_plot(self):
        '''
            Callback invoked when the StartEvent happens, sets up our new data.
        '''
        self.metric_values = []
        self.multires_iterations = []

    def end_plot(self):
        '''
            Callback invoked when the EndEvent happens, do cleanup of data and figure.
        '''
        # Close figure, we don't want to get a duplicate of the plot latter on.
        plt.close()

    def plot_values(self, registration_method):
        '''
             Callback invoked when the IterationEvent happens, update our data and display new figure.
             
             Paramters
             ---------
             registration_method: Object
                 Any objetc with a method GetMetricValue() that reurtuns a float
        '''
        self.metric_values.append(registration_method.GetMetricValue())                                       
        # Clear the output area (wait=True, to reduce flickering), and plot current data
        clear_output(wait=True)
        # Plot the similarity metric values
        plt.plot(self.metric_values, 'r')
        plt.plot(self.multires_iterations, [self.metric_values[i] for i in self.multires_iterations], 'b*')
        plt.xlabel('Iteration Number',fontsize=12)
        plt.ylabel('Metric Value',fontsize=12)
        plt.show()

    def update_multires_iterations(self):
        '''
            Callback invoked when the sitkMultiResolutionIterationEvent happens, update
            the index into the  metric_values list. 
        '''
        self.multires_iterations.append(len(self.metric_values))
    
def register_spline(fixed_image, moving_image, fixed_image_mask=None, lr=200,
                    show_progress=False, verbose=True):
    '''
        Register `moving_image` to `fixed_image` using a spline transformation
        
        Parameters
        ----------
        fixed_image: SimpleITK Image
            Reference image
        moving_image: SimpleITK Image
            Image that will be transformed to match `fixed_image`
        fixed_image_mask: SimpleITK Image or None, default None
            Mask (of same size as the rest of the images) with voxels to consider for registration
            set to 1, and the rest set to 0, or None to not apply any mask
        lr: float, default 200
            Learning rate of the optimization algorithm
        show_progress: bool, default False
            Show a convergence plot during training
        verbose: bool, default True
            Show some information after training
        
        Returns
        -------
        initial_transform: SimpleITK Transform
        metric_value: float
    '''

    fixed_image= sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image= sitk.Cast(moving_image, sitk.sitkFloat32)
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_tracker= RegistrationTracker()
    
    # Determine the number of BSpline control points using the physical spacing we 
    # want for the finest resolution control grid. 
    grid_physical_spacing = [20.]*3 # A control point every 50mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    # The starting mesh size will be 1/4 of the original, it will be refined by 
    # the multi-resolution framework.
    mesh_size = [int(sz/4 + 0.5) for sz in mesh_size]
    
    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with 
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                     inPlace=True,
                                                     scaleFactors=[1,scale])
    #registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
    
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Use the LBFGS2 instead of LBFGS. The latter cannot adapt to the changing control grid resolution.
    #registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-5, numberOfIterations=100, deltaConvergenceTolerance=0.01)
    registration_method.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=50,
                                                      estimateLearningRate=registration_method.Never)
    
    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    #Add event handlers
    if show_progress:
        registration_method.AddCommand(sitk.sitkStartEvent, registration_tracker.start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, registration_tracker.end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, 
                                       registration_tracker.update_multires_iterations) 
        registration_method.AddCommand(sitk.sitkIterationEvent, 
                                       lambda: registration_tracker.plot_values(registration_method))
    
    registration_method.Execute(fixed_image, moving_image)
    
    #Resample
#     moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, 
#                                      sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    
    if verbose:
        print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
        print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    
    return initial_transform, registration_method.GetMetricValue()


def register_rigid(fixed_image, moving_image, lr=4, fixed_image_mask=None, 
                   show_progress=False, verbose=True):
    '''
        Register `moving_image` to `fixed_image` using a rigid transformation
        
        Parameters
        ----------
        fixed_image: SimpleITK Image
            Reference image
        moving_image: SimpleITK Image
            Image that will be transformed to match `fixed_image`
        fixed_image_mask: SimpleITK Image or None, default None
            Mask (of same size as the rest of the images) with voxels to consider for registration
            set to 1, and the rest set to 0, or None to not apply any mask
        lr: float, default 200
            Learning rate of the optimization algorithm
        show_progress: bool, default False
            Show a convergence plot during training
        verbose: bool, default True
            Show some information after training
        
        Returns
        -------
        initial_transform: SimpleITK Transform
        metric_value: float
    '''
    
    #A reasonable guesstimate for the initial translational alignment can be obtained by using 
    #the CenteredTransformInitializer (functional interface to the CenteredTransformInitializerFilter).
    
    #The resulting transformation is centered with respect to the fixed image and the translation 
    #aligns the centers of the two images. There are two options for defining the centers of the images, 
    #either the physical centers of the two data sets (GEOMETRY), or the centers defined by the intensity moments (MOMENTS).
    
    #Two things to note about this filter, it requires the fixed and moving image have the same type 
    #even though it is not algorithmically required, and its return type is the generic SimpleITK.Transform.
    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_tracker= RegistrationTracker()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings
    #registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-5, numberOfIterations=100, deltaConvergenceTolerance=0.01)
    registration_method.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=150, 
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=20,
                                                      estimateLearningRate=registration_method.Never)
    registration_method.SetOptimizerScalesFromPhysicalShift() 

    #Set the initial moving and optimized transforms.
    final_transform = sitk.Euler3DTransform(initial_transform)
    registration_method.SetInitialTransform(final_transform)
    
    # Setup for the multi-resolution framework. 
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1])
#     registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    #Add event handlers
    if show_progress:
        registration_method.AddCommand(sitk.sitkStartEvent, registration_tracker.start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, registration_tracker.end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, 
                                       registration_tracker.update_multires_iterations) 
        registration_method.AddCommand(sitk.sitkIterationEvent, 
                                       lambda: registration_tracker.plot_values(registration_method))

    registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                sitk.Cast(moving_image, sitk.sitkFloat32))
    
#     moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, 
#                                      sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    
    if verbose:
        print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
        print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    
    return final_transform, registration_method.GetMetricValue()

def get_gradient_features(image):
    '''
        Return the average gradient in x and y for a given `image`
        
        Parameters
        ----------
        image: SimpleITK Image
        
        Returns
        -------
        image: SimpleITK Image
        
    '''
    image_grad= sitk.GradientImageFilter().Execute(image)
    image_grad= 0.5* (sitk.VectorIndexSelectionCast(image_grad, 0) + \
                      sitk.VectorIndexSelectionCast(image_grad, 1) )
    return image_grad

def evaluate_registration(fixed_sitk, moving_sitk, registered_sitk, mask_list_sitk, 
                          factors, metrics= [np.mean]):
    '''
        Evaluate the registration using a custom metric that computes the weighted 
        average correlation of both input images at several points of the prostate,
        given by the list of masks
        
        Parameters
        ----------
        fixed_sitk: SimpleITK Image
            Reference image
        moving_sitk: SimpleITK Image
            Image to be transformed to match `fixed_image`. This input is unused as of now,
            but could be used to provide a metric of relative improvement
        registered_sitk: SimpleITK Image
            Image that has already been transformed to `fixed_sitk`
        nask_list_sitk: List of SimpleITK Image
            List of masks upon which to evaluate the correlation
        factors: array
            Weightings / ponderations for the correlations. Must be of the same length as
            the number of provided masks
        metrics: list of metrics, default [np.mean]
            List of metrics used to combine the results evaluated over the different masks
        
    '''
    #Convert to numpy
    (fixed, moving, registered)= [sitk.GetArrayFromImage(i) for i in 
                                  (fixed_sitk, moving_sitk, registered_sitk)]
    masks= [sitk.GetArrayFromImage(i) for i in mask_list_sitk]

#     orig_results= []
#     for m in masks:
#         for metric in metrics:
#             orig_results.append( np.corrcoef(fixed[m > 0.5], moving[m > 0.5])[0,1] )

    reg_results= []
    for m in masks:
        for metric in metrics:
            reg_results.append( np.corrcoef(fixed[m > 0.5], registered[m > 0.5])[0,1] )

    factors= np.array(factors)
    final_values= np.array(reg_results) * factors
    custom_metric= np.mean(final_values)
    #print(custom_metric, final_values * factors)

    return custom_metric, final_values

def save_transform_auto(pid, transform, transform_dir):
    '''
        Function to save a transform automatically (called by the code)
                
        Parameters
        ----------
        pid: str
            Patient ID
        transform: SimpleITK Transform
            The transform to save
        transform_dir: str
            Pase path to save the transforms
    '''
    sitk.WriteTransform(transform, os.path.join(transform_dir, pid + '.tfm'))
    print('Saved: ', os.path.join(transform_dir, pid + '.tfm'))

def save_transform(b, pid, transform, transform_dir):
    '''
        Callback function to save a transform (called by user interaction on a button b)
        
        Parameters
        ----------
        b: Jupyter Widget Button
            The calling element to this callback
        pid: str
            Patient ID
        transform: SimpleITK Transform
            The transform to save
        transform_dir: str
            Pase path to save the transforms
    '''
    b.description = 'Transform saved!'
    sitk.WriteTransform(transform, os.path.join(transform_dir, pid + '.tfm'))
    print('Saved: ', os.path.join(transform_dir, pid + '.tfm'))