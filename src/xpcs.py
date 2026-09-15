# xpcs.py

### Libraries ###
import sys
import numpy as np
import h5py
import hdf5plugin
import time
import os
from scipy.optimize import curve_fit
from scipy.special import erfinv
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.signal import detrend
from scipy import constants as sc
import scipy.io
import ipywidgets as widgets
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json



import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

from PIL import Image, ImageSequence


### For Data Analysis ###

#g2_exp = g2(q,dt)-1 = A*np.exp(-(t/tt)**b)
def kev_to_angstroms(E):
    return 12.398/E

def angstroms_to_kev(l):
    return 12.398/l

def scherrer(sigma,theta,wavelength,K):
    beta = 2*sigma*np.sqrt(2*np.log(2))
    return K*wavelength/(beta*np.cos(np.radians(theta)))

def gaussian(x, A, mu, sigma, c):
    return A*np.exp(-0.5 * ((x - mu)/sigma)**2) + c

def gaussian_no_bl(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0)**2 / (2 * sigma**2))

def skew_gaussian(x,A,mu,sigma,alpha,c):
    return A*np.exp(-0.5 * ((x - mu)/sigma)**2)*(1+erf(alpha*(x-mu)/np.sqrt(2))) + c

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, background):
    x, y = xy
    #x0 = float(x0)
    #y0 = float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = background + amplitude*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    
    return g.ravel()

def pseudo_voigt(x, amplitude, center, fwhm, eta,c):
    """
    Pseudo-Voigt profile: linear combination of Gaussian and Lorentzian.
    
    Parameters
    ----------
    x : array-like
        Q or 2θ values.
    amplitude : float
        Integrated intensity of the peak.
    center : float
        Peak center (same units as x).
    fwhm : float
        Full width at half maximum.
    eta : float
        Mixing parameter between Lorentzian (eta=1) and Gaussian (eta=0).
    c : float
        Background.
    
    Returns
    -------
    y : array-like
        Pseudo-Voigt lineshape values.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Gaussian sigma
    gamma = fwhm / 2                             # Lorentzian HWHM

    # Gaussian component (area-normalized)
    gaussian = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    gaussian /= sigma * np.sqrt(2 * np.pi)

    # Lorentzian component (area-normalized)
    lorentzian = gamma / (np.pi * ((x - center) ** 2 + gamma ** 2))

    # Weighted sum
    profile = eta * lorentzian + (1 - eta) * gaussian

    # Scale to requested amplitude
    return amplitude * profile +c

def fit_gaussian_2d(data,init):
    """
    data: Data to be fit. 2-Dimensional
    init: Initial conditions obtained from initial fitting and centering 
    
    """
    x = np.linspace(0, data.shape[1] - 1, data.shape[1])
    y = np.linspace(0, data.shape[0] - 1, data.shape[0])
    x, y = np.meshgrid(x, y)
    #initial_guess = (data.max, data.shape[1]/2, data.shape[0]/2, 20, 20, 0, 10)
    
    popt, pcov = curve_fit(gaussian_2d, (x, y), data.ravel(), p0=init)
    return popt, pcov

def skew_gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, alpha_x, alpha_y, theta, background):
    x, y = xy
    #x0 = float(x0)
    #y0 = float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = background + amplitude*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    
    #A*np.exp(-0.5 * ((x - mu)/sigma)**2)*(1+erf(alpha*(x-mu)/np.sqrt(2))) + c
    
    return g.ravel()


def integrated_gaussian(det,stdy=200,stdx=10):
    
        #Central Slice
    #central_slice = det.shape[0]//2

    ### Sum data along both axes for fitting ###
    y1 = np.sum(det[:,:],axis=0)
    x1 = np.arange(y1.shape[0])

    y2 = np.sum(det[:,:],axis=1)
    x2 = np.arange(y2.shape[0])

    popt1, pcov1 = curve_fit(gaussian, x1, y1, p0=[np.max(y1), np.argmax(y1) , stdy, np.min(y1[np.nonzero(y1)])])
    popt2, pcov2 = curve_fit(gaussian, x2, y2, p0=[np.max(y2), np.argmax(y2) , stdx, np.min(y2[np.nonzero(y2)])])

    
    return popt1, popt2, pcov1, pcov2

def com(det):
    
    x = np.arange(det.shape[1])
    y = np.arange(det.shape[0])
    X, Y = np.meshgrid(x, y)

    # Calculate total mass (sum of all intensities)
    total_mass = det.sum()

    # Calculate center of mass
    row_cm = (X * det).sum() / total_mass
    col_cm = (Y * det).sum() / total_mass
    
    return row_cm, col_cm


def g2_exp(t,A,tt,b):
    return 1+A*np.exp(-(t/tt)**b)**2

def g2_exp_pow(t,A,tt1,tt2,b,a):
    return 1+A*(np.exp(-(t/tt1)**b)+t/tt2**a)**2

def g2_exp_c(t,A,tt,b,c):
    return c+A*np.exp(-(t/tt)**b)**2

def norm_g2(t,tt,b):
    return np.exp(-(t/tt)**b)

def g2_two_tau(t,A,a,tt1,tt2,b1,b2):
    return 1+A*np.abs(a*np.exp(-(t/tt1)**b1)+(1-a)*np.exp(-(t/tt2)**b2))**2

#def g2_two_tau(t,A1,A2,tt1,tt2,b1,b2):
#    return 1+A1*np.exp(-(t/tt1)**b1)+A2*np.exp(-(t/tt2)**b2)

def tau(q,A,c,a):
    return A*np.exp(-(q/c))

def raleigh(lam,fl,dl):
    return 1.22*fl*lam/dl

def newtth(Ein,Ebl,Th1):
    
    #Commonly used XRD energies in keV
    Cu = 8.0478
    Fe = 6.3998
    Co = 6.9257
    Mo = 17.45
    
    if isinstance(Ein, str):
        if Ein == 'Mo':
            Ein = None 
            Ein = 17.45
        if Ein == 'Cu':
            Ein = None 
            Ein = 8.0478
        if Ein == 'Co':
            Ein = None 
            Ein = 6.9257
        if Ein == 'Fe':
            Ein = None 
            Ein = 6.3998
 
    #Energies are in keV, 2Thetas in degrees
    ThNew = np.arcsin((Ein/Ebl)*np.sin((Th1/2)*np.pi/180))*180/np.pi
    #print(Th2*2)
    return 2*ThNew

def th2q(En,tth):
    wl = 1.2398*10/En
    return (4*np.pi/wl)*np.sin(tth*np.pi/(180*2))

def beta_Michelson(arr):
    I_min = np.min(arr)
    I_max = np.max(arr)
    return (I_max-I_min)/(I_max+I_min)

def beta(arr):
    return np.std(arr)/(np.mean(arr))

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def binning(det,binSize,pixSize=0.075,mode='mean'):
    # To bin the 800x800 to 400x400, we first reshape the array
    # Now, we take the mean across the newly introduced dimensions (2 and 4, the binning dimensions)
    # This averages every 2x2 bin into a single value, effectively reducing the resolution
    # Also returns the binned pixel sizes 
    
    
    
    if len(det.shape)==2:
        
        # Takew the modulus of the columns and rows by the bin size
        modRow = det.shape[0]%binSize
        modCol = det.shape[1]%binSize
        
        #Check if the detector dimensions are dividable by the bin size. Remove a few rows/columns if not
        if modRow != 0:
            det = det[modRow//2:(det.shape[0]-(modRow-modRow//2)),:]
        if modCol != 0:
            det = det[:,modCol//2:(det.shape[1]-(modCol-modCol//2))]
        
        if mode == 'mean':
            binnedDet = (det.reshape(int(det.shape[0]//binSize), binSize, int(det.shape[1]//binSize),binSize)).mean(axis=(1, 3))

        if mode == 'sum':
            binnedDet = (det.reshape(int(det.shape[0]//binSize), binSize, int(det.shape[1]//binSize), binSize)).sum(axis=(1, 3))
            
    if len(det.shape)==3:
        
        # Take the modulus of the columns and rows by the bin size
        modRow = det.shape[1]%binSize
        modCol = det.shape[2]%binSize
        
        #Check if the detector dimensions are dividable by the bin size. Remove a few rows/columns if not
        if modRow != 0:
            det = det[:,modRow//2:(det.shape[1]-(modRow-modRow//2)),:]
        if modCol != 0:
            det = det[:,:,modCol//2:(det.shape[2]-(modCol-modCol//2))]
            
        
        if mode == 'mean':
            binnedDet = (det.reshape(det.shape[0], int(det.shape[1]//binSize), binSize, int(det.shape[2]/binSize),binSize)).mean(axis=(2, 4))

        if mode == 'sum':
            binnedDet = (det.reshape(det.shape[0], int(det.shape[1]//binSize), binSize, int(det.shape[2]//binSize), binSize)).sum(axis=(2, 4))
    
    return binnedDet, pixSize*binSize

def time_bin(arr, bin_size, method="sum"):
    """
    Bins a 3D array along the first axis, time.

    Parameters:
    - arr: np.ndarray of shape (N, H, W)
    - bin_size: int, number of slices to bin together
    - method: str, either "sum" or "mean" for binning strategy

    Returns:
    - Binned array of shape (N // bin_size, H, W)
    """
    N, H, W = arr.shape
    assert N % bin_size == 0, "N must be divisible by bin_size for clean binning."

    # Reshape to group `bin_size` slices together
    arr_binned = arr.reshape(N // bin_size, bin_size, H, W)

    # Sum or average over the second axis
    if method == "sum":
        return arr_binned.sum(axis=1)
    elif method == "mean":
        return arr_binned.mean(axis=1)
    else:
        raise ValueError("Method must be either 'sum' or 'mean'.")


### Creates Elliptical Masks ###

def const_int_mask(arr,sigmax,sigmay,x0=None,y0=None,sigAll = 3,num_rings=5,num_slices=10,tilt=0,tol=0.20,res=0.1):
    """
    Creates ROI masks of ellipses of equal probability rings. Past attempts have used erfinv but this one will integrate the    intensity up to the 3rd std and calculate the percentage for each ring interatively. First it will find the appropriate ring "widths" by this procedure. 
    
    Inputs:
    arr: Cropped detector array (nframes, height, width)
    sigmax: Standard deviation in the x-direction in pixels from Gaussian fit
    sigmay: Standard deviation in the y-direction in pixels from Gaussian fit
    
    
    """
    # Parameters
    _, height, width = arr.shape  # Image dimensions
    
    if x0 is None or y0 is None:
        center = (height // 2, width // 2)
    else:
        center = (float(y0), float(x0))  # (row, col) in the cropped frame
    
    '''''
    #Addresses strange issue of 1 being the last number or not
    #Outer boundary for rings of equal probabilities in fractions of standard deviations
    probs = np.arange(1/num_rings,1,1/num_rings)
    if probs[-1]>0.99 and probs[-1]<=1:
        probs[-1] = 0.998  
    else:
        probs = np.append(np.arange(1/num_rings,1,1/num_rings),0.998)
        
    
    #ring_widths = 1/np.sqrt(probs)
    #ring_widths = np.sqrt(2)*erfinv(probs)
    #print(ring_widths)
    '''''
        
    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]

    ### First we must calculate the ring widths of constant intensity ###
    # Calculate the elliptical distance from the center
    #elliptical_distance = np.sqrt((x - center[1])**2/(sigmax)**2 + (y - center[0])**2/(sigmay)**2)
    
    # Apply rotation (tilt) to the coordinate system. COUNTER-CLOCKWISE!!!
    x_shifted = x - center[1]
    y_shifted = y - center[0]

    cos_t = np.cos(tilt)
    sin_t = np.sin(tilt)

    x_rot = x_shifted * cos_t + y_shifted * sin_t
    y_rot = -x_shifted * sin_t + y_shifted * cos_t

    elliptical_distance = np.sqrt((x_rot)**2/(sigmax)**2 + (y_rot)**2/(sigmay)**2)
    
    total_peak_mask = (elliptical_distance >= 0) & (elliptical_distance <= sigAll) #Mask for integrating the whole peak intensity, 3std
    
    avg_img = np.mean(arr[0:-1,:,:],axis=0) #Take the mean of the first 10 images 
    
    total_int = np.sum(avg_img[total_peak_mask==1]) #Get the total intensity of the whole peak
    
    ring_widths = np.zeros(num_rings)
    
    for i in range(num_rings):
        
        integ_int = 0
        
        while not((integ_int/total_int) <= (1/num_rings)*(1+tol) and (integ_int/total_int) >= (1/num_rings)*(1-tol)):
            
            ring_widths[i] += res

            # Choose ellipse rings
            if i==0:
                inner_radius = 0
                outer_radius = ring_widths[i]
            else:
                inner_radius = ring_widths[i-1]
                outer_radius = ring_widths[i]
             
            
            temp_mask = (elliptical_distance >= inner_radius) & (elliptical_distance < outer_radius) #Create maske
            
            integ_int = np.sum(avg_img[temp_mask==1]) #Get the total intensity of the whole peak
    
        print(integ_int/total_int)
        
    print(ring_widths)
    

    # Calculate the angle of each pixel relative to the center
    angle_from_center = np.arctan2(y - center[0], x - center[1])  # Angle in radians
    angle_from_center = (angle_from_center + 2 * np.pi) % (2 * np.pi) # Normalize to [0, 2*pi)
    #angle_from_center_shifted = (angle_from_center + np.pi) % (2 * np.pi) - np.pi  # From [-pi, pi)
    
    angle_width = 2 * np.pi / num_slices  # Width of each slice

    # Define the boundaries for the elliptical rings
    #ring_width = 1.0 / num_rings  # Each ring will occupy a fraction of the elliptical distance range
    roi_mask = np.zeros((num_rings*num_slices,height, width))  # Create a blank mask for all ROIs
    #roi_mask[:,:,:] = -1 #This is for processing later. -1 is an unphysical value, unlike 0
    
    # Assign each ROI a unique index for visualization
    roi_index = 0
    
    
    
    for i in range(num_rings):
        # Choose ellipse rings
        if i==0:
            inner_radius = 0
            outer_radius = ring_widths[i]
        else:
            inner_radius = ring_widths[i-1]
            outer_radius = ring_widths[i]
        
        # Create a mask for each elliptical ring
        ring_mask = (elliptical_distance >= inner_radius) & (elliptical_distance < outer_radius)

        for j in range(num_slices):
            
            start_angle = -angle_width / 2 + j * angle_width + tilt
            end_angle = start_angle + angle_width

            if j==0:
                slice_mask = (angle_from_center >= 2*np.pi+start_angle) | (angle_from_center < end_angle) 
            else:
                slice_mask = (angle_from_center >= start_angle) & (angle_from_center < end_angle)
    
    
            # Define angular boundaries for each segment
            #start_angle = j * (2 * np.pi / num_slices) - (2 * np.pi / num_slices) /2
            #end_angle = (j + 1) * (2 * np.pi / num_slices) - (2 * np.pi / num_slices) /2
            # Create a mask for each angular segment

            #slice_mask = (angle_from_center < end_angle) & (angle_from_center >= start_angle) 
                
            # Combine the elliptical ring mask and slice mask
            combined_mask = ring_mask & slice_mask
            # Assign a unique value to each region for visualization
            roi_mask[roi_index,combined_mask] = 1
            roi_index += 1
            
    return roi_mask,ring_widths 

def create_elliptical_mask(arr,sigmax,sigmay,num_rings=5,num_slices=10):
    """
    Creates ROI masks of ellipses of equal probability rings
    
    Inputs:
    arr: Cropped detector array (nframes, height, width)
    sigmax: Standard deviation in the x-direction in pixels from Gaussian fit
    sigmay: Standard deviation in the y-direction in pixels from Gaussian fit
    
    
    """
    # Parameters
    _, height, width = arr.shape  # Image dimensions
    
    center = (height // 2, width // 2)  # Center of the image
    
    #Addresses strange issue of 1 being the last number or not
    #Outer boundary for rings of equal probabilities in fractions of standard deviations
    probs = np.arange(1/num_rings,1,1/num_rings)
    if probs[-1]>0.99 and probs[-1]<=1:
        probs[-1] = 0.998  
    else:
        probs = np.append(np.arange(1/num_rings,1,1/num_rings),0.998)
        
    
    #ring_widths = 1/np.sqrt(probs)
    ring_widths = np.sqrt(2)*erfinv(probs)
    print(ring_widths)
    
        
    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]

    # Calculate the elliptical distance from the center
    elliptical_distance = np.sqrt((x - center[1])**2/(sigmax)**2 + (y - center[0])**2/(sigmay)**2)
    
    #print(np.max(elliptical_distance))

    # Calculate the angle of each pixel relative to the center
    angle_from_center = np.arctan2(y - center[0], x - center[1])  # Angle in radians
    angle_from_center = (angle_from_center + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2*pi)

    # Define the boundaries for the elliptical rings
    #ring_width = 1.0 / num_rings  # Each ring will occupy a fraction of the elliptical distance range
    roi_mask = np.zeros((num_rings*num_slices,height, width))  # Create a blank mask for all ROIs
    #roi_mask[:,:,:] = -1 #This is for processing later. -1 is an unphysical value, unlike 0
    
    # Assign each ROI a unique index for visualization
    roi_index = 0
    
    for i in range(num_rings):
        # Choose ellipse rings
        if i==0:
            inner_radius = 0
            outer_radius = ring_widths[i]
        else:
            inner_radius = ring_widths[i-1]
            outer_radius = ring_widths[i]
        
        # Create a mask for each elliptical ring
        ring_mask = (elliptical_distance >= inner_radius) & (elliptical_distance < outer_radius)

        for j in range(num_slices):
            # Define angular boundaries for each segment
            start_angle = j * (2 * np.pi / num_slices)
            end_angle = (j + 1) * (2 * np.pi / num_slices)
            # Create a mask for each angular segment
            slice_mask = (angle_from_center >= start_angle) & (angle_from_center < end_angle)
            # Combine the elliptical ring mask and slice mask
            combined_mask = ring_mask & slice_mask
            # Assign a unique value to each region for visualization
            roi_mask[roi_index,combined_mask] = 1
            roi_index += 1
            
    return roi_mask 

    

### Creates square grid masks ###

def create_square_mask(arr, block_shape, mask_shape):
    l, m, n = arr.shape
    dy, dx = block_shape
    ny, nx = mask_shape

    yc, xc = m // 2, n // 2 #Image center
    y0, x0 = yc-(ny//2)*dy, xc-(nx//2)*dx #Loop starting point
    y, x = y0, x0 #Position variables
    
    blocks = []
    coords = []
    dPixels = [] #Number of Pixel distance in [x, y]
    
    for i in range(0, ny, 1):
        x = x0
        for j in range(0, nx, 1):
            blocks.append(arr[:,(y-dy//2):(y+dy//2), (x-dx//2):(x+dx//2)])
            coords.append((x, y)) 
            dPixels
            x += dx
        y += dy
    return blocks, coords

### For Converting from Real Space to Reciprical Space ###

def q_to_tth(q,lambDuh):
    return 2*np.arcsin(q*lambDuh/(4*np.pi))*(180/np.pi)

def reciprocal_space_map(lambDuh,tth,hor0,ver0,pix,sam2det,detShape,geometry):
    # Written by Vanya-GPT
    #lambDuh: beam wavelength [Å]
    #tth: detector angle posistion
    #hor0: horizontal beam position
    #ver0: vertical beam position
    #pix: pixel size [mm]
    #sam2det: sample to detector distance [mm]
    #detShape: 2x2 horizontal, vertical
    
    if geometry == 'Horizontal':
        HorScattAngle=tth-np.arctan((np.arange(detShape[0,0],detShape[0,1]+1,1)-hor0)*pix/sam2det)*(180/np.pi); #In degrees
        VerScattAngle=-1*np.arctan((np.arange(detShape[1,0],detShape[1,1]+1,1)-ver0)*pix/sam2det)*(180/np.pi); #In degrees
    if geometry == 'Vertical':
        VerScattAngle=tth-np.arctan((np.arange(detShape[1,0],detShape[1,1]+1,1)-ver0)*pix/sam2det)*(180/np.pi); #In degrees
        HorScattAngle=-1*np.arctan((np.arange(detShape[0,0],detShape[0,1]+1,1)-hor0)*pix/sam2det)*(180/np.pi); #In degrees
    if geometry != 'Horizontal' and geometry != 'Vertical':
        sys.exit(f"Geometry must be either Vertical or Horizontal.")
        
    qx=(4*np.pi/lambDuh)*np.sin((HorScattAngle/2)*np.pi/180); # [Å^-1]
    qy=(4*np.pi/lambDuh)*np.sin((VerScattAngle/2)*np.pi/180); # [Å^-1]
    
    return qx,qy

def reciprocal_space_mapping(detector_width, detector_height, sample_to_detector_distance, pixel_size, two_theta, wavelength, initial_beam_position):
    # Convert two theta to radians
    two_theta_rad = np.deg2rad(two_theta)

    # Create a meshgrid for the detector pixels
    x = np.arange(detector_width)
    y = np.arange(detector_height)
    X, Y = np.meshgrid(x, y)

    # Calculate the pixel positions with respect to the initial beam position
    X = (X - initial_beam_position[0]) * pixel_size
    Y = (Y - initial_beam_position[1]) * pixel_size

    # Calculate the distance from each pixel to the sample
    Z = np.sqrt(sample_to_detector_distance**2 + X**2 + Y**2)

    # Calculate the angles of each pixel
    theta_x = np.arctan(X / sample_to_detector_distance)
    theta_y = np.arctan(Y / sample_to_detector_distance)

    # Calculate the reciprocal space coordinates
    k = 2 * np.pi / wavelength
    q_x = k * (np.cos(theta_x) - np.cos(two_theta_rad))
    q_y = k * np.sin(theta_y)
    q_z = k * (np.sin(theta_x) + np.sin(two_theta_rad))

    return q_x, q_y, q_z

### For Plotting ####

def make_detector_movie(filename, imgs, scan_var,period, fig, ax, fps,scale='log',clims = [0,10]):
    # Makes an animation of the detector images stored in `imgs`.
    # `fig` and `ax` are the Figure and Axes objects used to plot each movie frame
    # `filename` is the name of the output .gif file
    # `clims` is the colormap range
    # `fps` is the frame rate (frames per second) of the movie
    Nt = imgs.shape[0]
    
    if scale=='log':
        im = ax.imshow(imgs[0,:,:]+1, cmap='nipy_spectral',norm=LogNorm(clims[0], clims[1]))
    if scale=='gray':
        im = ax.imshow(imgs[0,:,:],cmap = 'gray',clim = (clims[0],clims[1]))
    else:
        im = ax.imshow(imgs[0,:,:], cmap='nipy_spectral',clim = (clims[0],clims[1]))
    
    
    def func(ii):
        im.set_data(imgs[ii,:,:])
        ax.set_title('Time = ' + str(np.around(period*scan_var[ii-1],2)) + ' seconds')
        return im
    
    anim = FuncAnimation(fig, func, frames=range(1,Nt))
    anim.save(filename + '.gif', writer=PillowWriter(fps=fps))

def create_color_list(length):
    # Get the default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Cycle through the colors and create a list of the specified length
    color_list = [color_cycle[i % len(color_cycle)] for i in range(length)]
    return color_list


### For Opening h5 Files ###

def print_h5_item(item, indent=''):
    """
    Recursively print the contents of an h5py group or dataset.
    
    Args:
    - item: The h5py group or dataset to print.
    - indent: A string of spaces used to indent nested items for better readability.
    """
    
    if isinstance(item, h5py.Group):  # Check if item is a group
        for key, subitem in item.items():
            print(f"{indent}/{key}")  # Print group name
            print_h5_item(subitem, indent + '    ')  # Recursively print contents of the group with additional indentation
    elif isinstance(item, h5py.Dataset):  # Check if item is a dataset
        print(f"{indent}[Dataset] Shape: {item.shape}, Type: {item.dtype}")
        # To print actual data, uncomment the line below. Be cautious with large datasets.
        # print(item[:])

        
### For Opening Batchinfo Files ###

def load_batchinfo(file_path,splitter):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by the first occurrence of the splitter
            key, value = line.split(splitter, 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert value to appropriate type (int, float, or leave as string)
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.startswith("[") and value.endswith("]"):
                        value = eval(value)
                    
            parameters[key] = value
    return parameters


    ############################################################
    # For Two-Time Correlation Analysis
    ############################################################

def semi_antidiagonal_linecut(C, extent=None, t1_start=None):
    """
    Linecut perpendicular to the main diagonal (t1 = t2), from a start point
    toward the lower-right corner (origin='lower' imshow convention).

    Parameters
    ----------
    t1_start : float or None
        Physical t1 (seconds) where the cut begins. If None, starts at the
        matrix center. Requires ``extent``.
    """
    n, m = C.shape
    cy, cx = n // 2, m // 2
    di, dj = -1, 1  # index step: down in t1, right in t2

    if t1_start is None:
        i0, j0 = cy, cx
    else:
        if extent is None:
            raise ValueError("extent is required when t1_start is set")
        ext = np.asarray(extent, dtype=float)
        t1_axis = np.linspace(ext[2], ext[3], n)
        i0 = int(np.argmin(np.abs(t1_axis - t1_start)))
        j0 = int(round(i0 + (cx - cy)))
        if j0 < 0 or j0 >= m:
            raise ValueError(
                f"Start point (i={i0}, j={j0}) out of bounds for t1_start={t1_start}"
            )

    ii, jj, vals = [], [], []
    k = 0
    while True:
        i = i0 + di * k
        j = j0 + dj * k
        if i < 0 or i >= n or j < 0 or j >= m:
            break
        ii.append(i)
        jj.append(j)
        vals.append(C[i, j])
        k += 1

    out = {
        "values": np.asarray(vals),
        "i": np.asarray(ii),
        "j": np.asarray(jj),
        "k": np.arange(len(vals)),
        "start": (i0, j0),
        "center": (cy, cx),
    }

    if extent is not None:
        ext = np.asarray(extent, dtype=float)
        t2 = np.linspace(ext[0], ext[1], m)
        t1 = np.linspace(ext[2], ext[3], n)
        out["t1"] = t1[out["i"]]
        out["t2"] = t2[out["j"]]
        out["delta_tau"] = out["t2"] - out["t1"]
        out["t1_start"] = t1[i0]

    return out

def average_linecuts(cut_list, tau_list):
    """Mean/std over cuts of different length; pad with NaN, not zero."""
    max_len = max(len(v) for v in cut_list)
    cuts = np.full((len(cut_list), max_len), np.nan)
    taus = np.full((len(cut_list), max_len), np.nan)
    for i, (v, t) in enumerate(zip(cut_list, tau_list)):
        n = len(v)
        cuts[i, :n] = v
        taus[i, :n] = t

    cut = np.nanmean(cuts, axis=0)
    cut_err = np.nanstd(cuts, axis=0)/np.sqrt(len(cuts))
    tau = np.nanmean(taus, axis=0)

    valid = ~np.isnan(cut)
    return cut[valid], cut_err[valid], tau[valid]

def fft_analysis(tau_trim, resid_trim, n_uniform=4096, n_fft=16384):
    """FFT of residuals on a uniform linear tau grid (frequencies in Hz)."""
    tau_uniform = np.linspace(tau_trim.min(), tau_trim.max(), n_uniform)
    d_tau = tau_uniform[1] - tau_uniform[0]
    resid_uniform = interp1d(
        tau_trim, resid_trim, kind='linear', bounds_error=False, fill_value=0.0
    )(tau_uniform)

    hann_window = np.hanning(len(resid_uniform))
    resid_windowed = resid_uniform * hann_window

    resid_padded = np.pad(resid_windowed, (0, n_fft - len(resid_windowed)))
    freqs = np.fft.rfftfreq(n_fft, d=d_tau)
    power = np.abs(np.fft.rfft(resid_padded))**2

    return freqs, power