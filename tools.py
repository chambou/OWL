#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for different purposes
"""
import numpy as np
from PIL import Image
import scipy.ndimage
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import time
from numpy.fft import fft2,ifft2,fftshift
from scipy.optimize import leastsq
import scipy as scp
import os
from scipy.fft import dctn, idctn
#from krtc import *
from hcipy import *
#import zmq
from astropy.io import fits

#%% ====================================================================
#   ==================== USEFUL FUNCTIONS ==============================
#   ====================================================================

def makePupil(Rpx,nPx):
    
    # Cicular pupil
    pupil = np.zeros((nPx,nPx))
    for i in range(0,nPx):
        for j in range(0,nPx):
            if np.sqrt((i-nPx/2+0.5)**2+(j-nPx/2+0.5)**2)<Rpx:
                pupil[i,j]=1
                
    return pupil
    
def transformation_dm2wfs(alpha,beta,theta,x_wfs_0,y_wfs_0,x_dm_0,y_dm_0,xx_dm,yy_dm):
        # ---- Tramsformation ------
        xx_wfs = np.cos(theta)*alpha*(xx_dm-x_dm_0)-np.sin(theta)*beta*(yy_dm-y_dm_0)+x_wfs_0
        yy_wfs = -(np.sin(theta)*alpha*(xx_dm-x_dm_0)+np.cos(theta)*beta*(yy_dm-y_dm_0))+y_wfs_0
        
        return xx_wfs,yy_wfs
        
def gauss2d(A,x0,y0,sigma,n):
    # ------------ MASK -------------
    X = np.round(np.linspace(0,n-1,n))
    [x,y] = np.meshgrid(X,X)
    g = A*np.exp(-(1/(2*sigma**2))*((x-x0)**2+(y-y0)**2))
        
    return g

def build_poke_matrix_wfs(xx_wfs,yy_wfs,A,sigma,nPx,pupil):
        poke_matrix = np.zeros((nPx**2,xx_wfs.shape[0]))
        for k in range(0,xx_wfs.shape[0]):
            poke = gauss2d(A,xx_wfs[k],yy_wfs[k],sigma,nPx)
            poke = poke*pupil
            poke_matrix[:,k] = np.ravel(poke)
    
        return poke_matrix

def binArray(arr, K):
    shape = (arr.shape[0]//K, K,
             arr.shape[1]//K, K)
    return arr.reshape(shape).mean(-1).mean(1)

def binArray255(input_array, N):
    """
    Bin array to N x N
    """
    # Normalize the input array to values between 0 and 1
    input_array_norm = (input_array - input_array.min()) / (input_array.max() - input_array.min())
    # Convert the normalized array to an image with values between 0 and 255
    input_image = Image.fromarray((input_array_norm * 255).astype(np.uint8))
    # Resize the image to N by N pixels
    output_image = input_image.resize((N, N))
    # Convert the resized image back to a 2D array with float values between 0 and 1
    output_array_norm = np.array(output_image, dtype=np.float32) / 255
    # Scale the output array back to the original range of values
    output_array = output_array_norm * (input_array.max() - input_array.min()) + input_array.min()
    return output_array

def binIM(arr,N):
    new_arr = np.zeros((N**2,arr.shape[1]))
    for k in range(0,arr.shape[1]):
        a = np.copy(arr[:,k])
        a = a.reshape((int(np.sqrt(arr.shape[0])),int(np.sqrt(arr.shape[0]))))
        a = binArray255(a,N)
        a = a.ravel()
        new_arr[:,k] = a
    return new_arr
    
def error_rms(A,B,pupil=None):
    if pupil is None:
        pupil = np.ones(A.shape)
    err = np.sqrt(np.sum(np.sum((A*pupil-B*pupil)**2)))/np.sqrt(np.sum(pupil))
    return err

def fig(mat,vmin=None,vmax=None):
   plt.figure()
   if vmin is not None and vmax is not None:
      plt.imshow(mat,vmin=vmin,vmax=vmax)
   elif vmin is not None and vmax is None:
      plt.imshow(mat,vmin=vmin)
   elif vmin is None and vmax is not None:
      plt.imshow(mat,vmax=vmax)
   else:
      plt.imshow(mat)
   plt.colorbar()
   plt.show(block=False)

def save_fits(im,full_path,header_info=dict()):
    """
    Save 2-D or 3-D array as fits file
    Option: Store information in header
    """
    hdu = fits.PrimaryHDU(im)
    for key in header_info.keys():
        hdu.header[key] = header_info[key]
    hdu.writeto(full_path)

    # Check that image saved:
    if os.path.exists(full_path):
        print("--Image saved:",full_path)
    else:
        print("!!! Something went wrong, didn't save:",full_path)

def open_fits(image_path,dtype_option=float,ignore_end_card=False):
    """
    Open 2-D or 3-D array saved as fits
    Load header information
    """
    if ignore_end_card == True:
        hdulist = fits.open(image_path,ignore_missing_end=True)
    else:
        hdulist = fits.open(image_path,mmap=True)
    im = np.array(hdulist[0].data,dtype=dtype_option)
    header = hdulist[0].header

    hdulist.close()

    return im, header

#%% ====================================================================
#   ================ TAKE off mask pupil on PWFS =======================
#   ====================================================================	

def take_image_pupils(cam,dm,amp):
    img = 0
    for k in range(0,4):
        if k == 0:
            dm.pokeZernike([amp,amp],[2,3])
        elif k == 1:
            dm.pokeZernike([-amp,amp],[2,3])
        elif k == 2:
            dm.pokeZernike([-amp,-amp],[2,3])
        elif k == 3:
            dm.pokeZernike([amp,-amp],[2,3])
        time.sleep(1)
        img = img+cam.get()
    dm.setFlatSurf()
    img = img[:,int(360/cam.binning):-int(360/cam.binning)]
    return img
    
#%% ====================================================================
#   ==================== Modal Basis: KL and Zernike  ==================
#   ====================================================================

def KLBasis(N,nPx_pup,pupil_footprint=None):
    """
    Function to define KL basis 

    Parameters: 
    - N: Number of modes
    - nPx: Resolution in pupil
    - pupil
    """
    if pupil_footprint is None:
        pupil_footprint = makePupil(nPx_pup/2,nPx_pup)
    nPx = int(N)
    pupil_f = scipy.ndimage.zoom(pupil_footprint,nPx/pupil_footprint.shape[0])
    pupil_f[pupil_f<0.99] = 0
    pupil_f[pupil_f>1.1] = 1
    fig(pupil_f)
    r0 = 0.15 # fried parameter in meter
    D = 10 # Telescope diameter in meter
    L0 = 30 # Outer scale in meter
    fx_size = 1/D
    freq = np.linspace((-nPx-1)*fx_size,nPx*fx_size,2*nPx) # Double size w.r.t to pupil
    xx,yy = np.meshgrid(freq,freq)
    f = np.sqrt(xx**2+yy**2)
    C = 0.023*r0**(-5/3)*(f**2+1/L0**2)**(-11/6) # Von Karman DSP
    B = np.real(fftshift(fft2(fftshift(C)))) # Von Karman Covariance map

    # Build Covariance matrix in pupil space
    n_pixel_pup = int(np.sum(pupil_f))
    Cp = np.zeros((n_pixel_pup,n_pixel_pup))
    compt = 0 # Iterate to fill Cp
    plt.figure()
    for i in range(0,nPx):
        for j in range(0,nPx):
            if pupil_f[i,j] == 1:
                B_offset = B[nPx-i:2*nPx-i,nPx-j:2*nPx-j]
                Cp[:,compt] = B_offset[pupil_f.astype(bool)]
                compt = compt + 1

    # Compute SVD
    U,S,V = np.linalg.svd(Cp)
    # Fill modal Basis matrix
    modalBasis = np.zeros((pupil_footprint.shape[0]**2,N))
    for k in range(N):
        pup = np.copy(pupil_f)
        u = U[:,k]
        pup[pupil_f.astype(bool)] = u
        pup = scipy.ndimage.zoom(pup,pupil_footprint.shape[0]/nPx)*pupil_footprint
        # Normalization to 1 radians rms
        pup = pup/(np.sqrt(np.sum(pup**2)/np.sum(pupil_footprint)))
        # Increase size to 
        modalBasis[:,k] = pup.ravel()
 
    return modalBasis


def zernikeBasis(N,nPx):
    """ Define a Zernike Basis using HCIpy function (faster than previous function)"""
    pupil_grid = make_pupil_grid(nPx,10) # Diameter size = 10m but doesn't matter
    a=make_zernike_basis(N,10,pupil_grid)
    zernike = a.transformation_matrix
    # # Defined on a Cicular pupil
    # Rpx = int(nPx/2)
    # nPx = int(nPx)
    # pupil = np.zeros((nPx,nPx))
    # for i in range(0,nPx):
    #     for j in range(0,nPx):
    #         if np.sqrt((i-nPx/2+0.5)**2+(j-nPx/2+0.5)**2)<=(Rpx):
    #             pupil[i,j]=1
    
    # pupil_px = sum(sum(pupil))
    # zernike = np.zeros((nPx,nPx,N))
    # X = np.linspace(-nPx/2,nPx/2,nPx)
    # Y = np.linspace(nPx/2,-nPx/2,nPx)

    # x,y = np.meshgrid(X,Y)
    
    # for n in range(0,N):
    #     for m in range(0,N):
    #         if m == 0:
    #             i1 = n*(n+1)/2+m+1
    #         else:
    #             i1 = n*(n+1)/2+m+1
    #             i2 = n*(n+1)/2+m
    
    #         if m <= n and i1 <= N and n%2 == m%2:
    #             S = np.linspace(0,int((n-m)/2),int((n-m)/2+1))
    #             R = np.zeros((nPx,nPx))
    #             for s in S:
    #                 R = R+(-1)**s*np.math.factorial(n-s)*(np.sqrt(x**2+y**2)/Rpx)**(n-2*s)/(np.math.factorial(s)*np.math.factorial((n+m)/2-s)*np.math.factorial((n-m)/2-s))
                
    #             if m == 0:
    #                 Z = np.sqrt(n+1)*R
    #                 zernike[:,:,int(i1)-1] = Z*pupil/np.sqrt((1/pupil_px)*sum(sum((Z*pupil)**2)))
    #             else:
    #                 if i1%2 == 0:
    #                     Z = np.sqrt(2*(n+1))*R*np.cos(m*np.arctan2(x,y))
    #                 else:
    #                     Z = np.sqrt(2*(n+1))*R*np.sin(m*np.arctan2(x,y))
                    
    #                 zernike[:,:,int(i1)-1] = Z*pupil/np.sqrt((1/pupil_px)*sum(sum((Z*pupil)**2)))
                        
    #                 if i2%2 == 0:
    #                     Z = np.sqrt(2*(n+1))*R*np.cos(m*np.arctan2(x,y))
    #                 else:
    #                     Z = np.sqrt(2*(n+1))*R*np.sin(m*np.arctan2(x,y))
                    
    #                 zernike[:,:,int(i2)-1] = Z*pupil/np.sqrt((1/pupil_px)*sum(sum((Z*pupil)**2)))
                
    # zernike = zernike.reshape((nPx**2,N))
        
    return zernike

#%% ====================================================================
#   ============= Custom FFT to be centered on 4 pixels ================
#   ====================================================================

############# CHANGE FFT DEFINITION TO FIX ISSUE IN PWFS CODE ###########
def fft_centre(X):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(X)))

def ifft_centre(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(X)))

''''
def fft_centre(X):
    
    # Compute a phasor in direct space + Fourier space to be centered on both.
    # Phasor computation
    nPx = X.shape[0]
    [xx,yy] = np.meshgrid(np.linspace(0,nPx-1,nPx),np.linspace(0,nPx-1,nPx))
    phasor = np.exp(-(1j*np.pi*(nPx+1)/nPx)*(xx+yy))
    # Phasor in Direct space
    Y = np.fft.fft2(X*phasor)
    # Phasor in Fourier space
    Y = phasor*Y
    # Normalisation
    Y = np.exp(-(1j*np.pi*(nPx+1)**2/nPx))*Y
    # Normalisation DFT - Divide by nPx because fft2
    Y = Y/nPx

    return Y

def ifft_centre(X):
    
    # Compute a phasor in direct space + Fourier space to be centered on both.
    # Phasor computation
    nPx = X.shape[0]
    [xx,yy] = np.meshgrid(np.linspace(0,nPx-1,nPx),np.linspace(0,nPx-1,nPx))
    phasor = np.exp((1j*np.pi*(nPx+1)/nPx)*(xx+yy))
    # Phasor in Direct space
    Y = np.fft.ifft2(X*phasor)
    # Phasor in Fourier space
    Y = phasor*Y
    # Normalisation
    Y = np.exp((1j*np.pi*(nPx+1)**2/nPx))*Y
    # Normalisation DFT - Multiply by nPx because ifft2
    Y = Y*nPx
    
    return Y
'''
#%% ====================================================================
#   ============== LOCATE PUPIL in WFS images     =====================
#   ====================================================================

def findPWFS_param(img):
    # -------------
    #  - Use img (cumulative img of PWFS camera with PSF displaced in each quasrant)
    # -------------
    
    # --------------- Track -----------------------------
    nPx_img = img.shape[0]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.15)
    #ax.axis([0, nPx_img, 0, nPx_img])
    ax.set_aspect("equal")
    plt.imshow(img)
    plt.title('Find pupils positions')
    #======================= SLIDERS ==============================
    axcolor = 'skyblue'
    # --- LEFT TOP PUPIL -----
    sl_lt_1 = plt.axes([0.05, 0.25, 0.15, 0.03], facecolor=axcolor)
    slider_lt_1 = Slider(sl_lt_1 , 'X', 0.0*nPx_img, 0.5*nPx_img, 0.2922*nPx_img)# big pupil 0.2908
    sl_lt_2 = plt.axes([0.05, 0.2, 0.15, 0.03], facecolor=axcolor)
    slider_lt_2 = Slider(sl_lt_2, 'Y', 0.5*nPx_img, nPx_img, 0.7481*nPx_img) # 0.7258
    # --- LEFT BOTTOM PUPIL -----
    sl_lb_1 = plt.axes([0.05, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_lb_1 = Slider(sl_lb_1, 'X', 0.0*nPx_img, 0.5*nPx_img, 0.2956*nPx_img) #0.2958
    sl_lb_2 = plt.axes([0.05, 0.75, 0.15, 0.03], facecolor=axcolor)
    slider_lb_2 = Slider(sl_lb_2 , 'Y', 0.0*nPx_img, 0.5*nPx_img, 0.3133*nPx_img) # 0.2958
    # --- RIGHT TOP PUPIL -----
    sl_rt_1 = plt.axes([0.75, 0.25, 0.15, 0.03], facecolor=axcolor)
    slider_rt_1 = Slider(sl_rt_1, 'X', 0.5*nPx_img, nPx_img, 0.7372*nPx_img) #0.7292
    sl_rt_2 = plt.axes([0.75, 0.2, 0.15, 0.03], facecolor=axcolor)
    slider_rt_2 = Slider(sl_rt_2 , 'Y', 0.5*nPx_img, nPx_img, 0.7498*nPx_img) #0.7292
    # --- RIGHT BOTTOM PUPIL -----
    sl_rb_1 = plt.axes([0.75, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_rb_1 = Slider(sl_rb_1, 'X', 0.5*nPx_img, nPx_img, 0.7422*nPx_img) #0.7358
    sl_rb_2 = plt.axes([0.75, 0.75, 0.15, 0.03], facecolor=axcolor)
    slider_rb_2 = Slider(sl_rb_2, 'Y', 0.0*nPx_img, 0.5*nPx_img, 0.3158*nPx_img) #0.2942
    # ---- Pupils Diameter -----
    start_radius = 0.1792*nPx_img #0.2008
    sl3 = plt.axes([0.35, 0.05, 0.3, 0.03], facecolor=axcolor)
    slider_r = Slider(sl3, 'Radius', 0.01*nPx_img, 0.25*nPx_img,start_radius)
    
    #======================= CIRCLES ==============================
    circ_lt = plt.Circle((0.2922*nPx_img,0.7481*nPx_img),start_radius,facecolor = [0,0,0,0],ec="w")
    circ_lb = plt.Circle((0.2956*nPx_img,0.3133*nPx_img),start_radius,facecolor = [0,0,0,0],ec="w")
    
    circ_rt = plt.Circle((0.7372*nPx_img,0.7498*nPx_img),start_radius,facecolor = [0,0,0,0],ec="w")
    circ_rb = plt.Circle((0.7422*nPx_img,0.315*nPx_img),start_radius,facecolor = [0,0,0,0],ec="w")
    
    ax.add_patch(circ_lt)
    ax.add_patch(circ_lb)
    ax.add_patch(circ_rt)
    ax.add_patch(circ_rb)
    
    #======================= PLOT ==============================
    def update(val):
        # ----- UPDATE X and Y position -----
        r_lt_1 = slider_lt_1.val
        r_lt_2 = slider_lt_2.val
        r_lb_1 = slider_lb_1.val
        r_lb_2 = slider_lb_2.val
        r_rt_1 = slider_rt_1.val
        r_rt_2 = slider_rt_2.val
        r_rb_1 = slider_rb_1.val
        r_rb_2 = slider_rb_2.val
        circ_lt.center = r_lt_1 , r_lt_2
        circ_lb.center = r_lb_1 , r_lb_2
        circ_rt.center = r_rt_1 , r_rt_2
        circ_rb.center = r_rb_1 , r_rb_2  
        # ----- UPDATE Radius -----
        circ_lt.set_radius(slider_r.val) 
        circ_lb.set_radius(slider_r.val) 
        circ_rt.set_radius(slider_r.val) 
        circ_rb.set_radius(slider_r.val) 
        fig.canvas.draw_idle()
        
    slider_lt_1.on_changed(update)
    slider_lt_2.on_changed(update)
    slider_lb_1.on_changed(update)
    slider_lb_2.on_changed(update)
    slider_rt_1.on_changed(update)
    slider_rt_2.on_changed(update)
    slider_rb_1.on_changed(update)
    slider_rb_2.on_changed(update)
    slider_r.on_changed(update)
    
    plt.show(block=True)
    
    r_lt_1 = int(slider_lt_1.val)
    r_lt_2 = int(slider_lt_2.val)
    r_lb_1 = int(slider_lb_1.val)
    r_lb_2 = int(slider_lb_2.val)
    r_rt_1 = int(slider_rt_1.val)
    r_rt_2 = int(slider_rt_2.val)
    r_rb_1 = int(slider_rb_1.val)
    r_rb_2 = int(slider_rb_2.val)
    r = int(slider_r.val)
    
    #  ----- BUILD PYRAMID MASK --------
    nPx_pup = 2*r
    x_center = int(nPx_img/2+1)#nPx_pup*shannon
    y_center = int(nPx_img/2+1)#nPx_pup*shannon
    
    # angle for all faces
    position_pup = np.zeros((8,1))
    # left top pupil -----
    position_pup[0] = (x_center-r_lt_1)/nPx_pup 
    position_pup[1] = (y_center-r_lt_2)/nPx_pup
    # left bottom pupil -----
    position_pup[2] = (x_center-r_lb_1)/nPx_pup 
    position_pup[3] = (y_center-r_lb_2)/nPx_pup 
    # right top pupil -----
    position_pup[4] = (x_center-r_rt_1)/nPx_pup 
    position_pup[5] = (y_center-r_rt_2)/nPx_pup 
    # right bottom pupil -----
    position_pup[6] = (x_center-r_rb_1)/nPx_pup 
    position_pup[7] = (y_center-r_rb_2)/nPx_pup 

    pup_lt = img[r_lt_2-r+1:r_lt_2+r+1,r_lt_1-r+1:r_lt_1+r+1]
    pup_lb = img[r_lb_2-r+1:r_lb_2+r+1,r_lb_1-r+1:r_lb_1+r+1]
    pup_rt = img[r_rt_2-r+1:r_rt_2+r+1,r_rt_1-r+1:r_rt_1+r+1]
    pup_rb = img[r_lb_2-r+1:r_lb_2+r+1,r_lb_1-r+1:r_lb_1+r+1]
    pupil = 1/4*(pup_lt+pup_lb+pup_rt+pup_lt+pup_rb)
    
    # Remove outside pixel
    pupil_footprint = makePupil(int(pupil.shape[0]/2),pupil.shape[0])
    pupil = pupil*pupil_footprint
    
    plt.figure()
    plt.imshow(pupil)
    plt.colorbar()
    plt.show(block=False)
    
    return pupil,position_pup


def maskFineTuning(img,PWFS):
    # -------------
    #  - Use img (cumulative img of PWFS camera with PSF displaced in each quasrant)
    # -------------
    
    # --------------- Track -----------------------------
    nPx_img = img.shape[0]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.15)
    ax.set_aspect("equal")
    imageDiff = plt.imshow(PWFS.img0-img)
    plt.title('Adjust PWFS mask angles')
    #======================= SLIDERS ==============================
    axcolor = 'skyblue'
    # --- LEFT TOP PUPIL -----
    sl_lt_1 = plt.axes([0.05, 0.25, 0.15, 0.03], facecolor=axcolor)
    slider_lt_1 = Slider(sl_lt_1 , 'X', 0.7, 1.3, 1)
    sl_lt_2 = plt.axes([0.05, 0.2, 0.15, 0.03], facecolor=axcolor)
    slider_lt_2 = Slider(sl_lt_2, 'Y', 0.7, 1.3, 1)
    # --- LEFT BOTTOM PUPIL -----
    sl_lb_1 = plt.axes([0.05, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_lb_1 = Slider(sl_lb_1, 'X', 0.7, 1.3, 1)
    sl_lb_2 = plt.axes([0.05, 0.75, 0.15, 0.03], facecolor=axcolor)
    slider_lb_2 = Slider(sl_lb_2 , 'Y', 0.7, 1.3, 1)
    # --- RIGHT TOP PUPIL -----
    sl_rt_1 = plt.axes([0.75, 0.25, 0.15, 0.03], facecolor=axcolor)
    slider_rt_1 = Slider(sl_rt_1, 'X', 0.7, 1.3, 1)
    sl_rt_2 = plt.axes([0.75, 0.2, 0.15, 0.03], facecolor=axcolor)
    slider_rt_2 = Slider(sl_rt_2 , 'Y', 0.7, 1.3, 1)
    # --- RIGHT BOTTOM PUPIL -----
    sl_rb_1 = plt.axes([0.75, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_rb_1 = Slider(sl_rb_1, 'X', 0.7, 1.3, 1)
    sl_rb_2 = plt.axes([0.75, 0.75, 0.15, 0.03], facecolor=axcolor)
    slider_rb_2 = Slider(sl_rb_2, 'Y', 0.7, 1.3, 1)

    #======================= PLOT ==============================
    def update(val):
        # ----- UPDATE X and Y position -----
        r_lt_1 = slider_lt_1.val
        r_lt_2 = slider_lt_2.val
        r_lb_1 = slider_lb_1.val
        r_lb_2 = slider_lb_2.val
        r_rt_1 = slider_rt_1.val
        r_rt_2 = slider_rt_2.val
        r_rb_1 = slider_rb_1.val
        r_rb_2 = slider_rb_2.val
        PWFS.offset_angle = np.array([r_lt_1,r_lt_2,r_lb_1,r_lb_2,r_rt_1,r_rt_2,r_rb_1,r_rb_2])
        new_mask = PWFS.build_mask(PWFS.position_pup)
        PWFS.changeMask(new_mask)
        imageDiff.set_data(PWFS.img0-img)
        #plt.imshow(PWFS.img0-img)
        fig.canvas.draw_idle()
        
    slider_lt_1.on_changed(update)
    slider_lt_2.on_changed(update)
    slider_lb_1.on_changed(update)
    slider_lb_2.on_changed(update)
    slider_rt_1.on_changed(update)
    slider_rt_2.on_changed(update)
    slider_rb_1.on_changed(update)
    slider_rb_2.on_changed(update)
    
    plt.show(block=True)
    
    
    return PWFS.offset_angle

def findZWFS_param(img,guess=[899,542,1080,546,69],dpix=50):
    # -------------
    #  - Use off dimple img
    # -------------
    
    # --------------- Track -----------------------------
    nPx_img = img.shape[0]
    figu, ax = plt.subplots()
    figu.set_size_inches(10, 6)
    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.15)
    #ax.axis([800, 1300, 450, 700])
    #ax.set_aspect("equal")
    plt.imshow(img)
    plt.title('Find pupil positions')
    #======================= SLIDERS ==============================
    axcolor = 'skyblue'
    # --- LEFT PUPIL -----
    start_left_1 = guess[0] #989#911#203
    start_left_2 = guess[1] #570#262
    sl_left_1 = plt.axes([0.05, 0.53, 0.15, 0.03], facecolor=axcolor)
    slider_left_1 = Slider(sl_left_1 , 'X', start_left_1-dpix,start_left_1+dpix, start_left_1)#100,600, start_left_1)#
    sl_left_2 = plt.axes([0.05, 0.48, 0.15, 0.03], facecolor=axcolor)
    slider_left_2 = Slider(sl_left_2, 'Y', start_left_2-dpix,start_left_2+dpix, start_left_2)#500,600, start_left_2)
    # --- RIGHT PUPIL -----
    start_right_1 = guess[2] #1157#1083#443
    start_right_2 = guess[3] #574#262
    sl_right_1 = plt.axes([0.75, 0.53, 0.15, 0.03], facecolor=axcolor)
    slider_right_1 = Slider(sl_right_1, 'X',start_right_1-dpix,start_right_1+dpix ,start_right_1)#100,600 ,start_right_1)#
    sl_right_2 = plt.axes([0.75, 0.48, 0.15, 0.03], facecolor=axcolor)
    slider_right_2 = Slider(sl_right_2 , 'Y', start_right_2-dpix, start_right_2+dpix, start_right_2)#500, 600, start_right_2)
    # ---- Pupils Diameter -----
    start_radius = guess[4]#69#108
    sl3 = plt.axes([0.35, 0.05, 0.3, 0.03], facecolor=axcolor)
    slider_r = Slider(sl3, 'Radius', start_radius-dpix,start_radius+dpix,start_radius)#65,75,start_radius)
    
    #======================= CIRCLES ==============================
    circ_left = plt.Circle((start_left_1,start_left_2),start_radius,facecolor = [0,0,0,0],ec="w")	
    circ_right = plt.Circle((start_right_1,start_right_2),start_radius,facecolor = [0,0,0,0],ec="w")
    
    ax.add_patch(circ_left)
    ax.add_patch(circ_right)

    
    #======================= PLOT ==============================
    def update(val):
        # ----- UPDATE X and Y position -----
        r_left_1 = slider_left_1.val
        r_left_2 = slider_left_2.val
        r_right_1 = slider_right_1.val
        r_right_2 = slider_right_2.val
        circ_left.center = r_left_1 , r_left_2
        circ_right.center = r_right_1 , r_right_2
        # ----- UPDATE Radius -----
        circ_left.set_radius(slider_r.val) 
        circ_right.set_radius(slider_r.val) 
        figu.canvas.draw_idle()
        
    slider_left_1.on_changed(update)
    slider_left_2.on_changed(update)
    slider_right_1.on_changed(update)
    slider_right_2.on_changed(update)
    slider_r.on_changed(update)
    
    plt.show(block=True)
    
    r_left_1 = int(slider_left_1.val)
    r_left_2 = int(slider_left_2.val)
    r_right_1 = int(slider_right_1.val)
    r_right_2 = int(slider_right_2.val)
    r = int(slider_r.val)

    # ----- Mask parameters -----
    nPx = 2*r
    pupil_footprint = makePupil(nPx/2,nPx)
    p = int((r_right_1-r_left_1 - nPx)/2)
    center_x = int(r_left_1+nPx/2 + p)
    center_y =int((r_right_2 +r_left_2)/2)
    nPx_img_y = nPx+2*p
    nPx_img_x = 2*(nPx+2*p)
    pupil_img_left = pupil_footprint*img[r_left_2-int(nPx/2):r_left_2+int(nPx/2),r_left_1-int(nPx/2):r_left_1+int(nPx/2)]
    fig(pupil_img_left)
    pupil_img_right = pupil_footprint*img[r_right_2-int(nPx/2):r_right_2+int(nPx/2),r_right_1-int(nPx/2):r_right_1+int(nPx/2)]
    fig(pupil_img_right)
    img_wfs = img[center_y-int(nPx_img_y/2):center_y+int(nPx_img_y/2),center_x-int(nPx_img_x/2):center_x+int(nPx_img_x/2)]
    fig(img_wfs)

    pupil_ZWFS = np.stack((pupil_img_left,pupil_img_right))
    #position_ZWFS = np.array([center_x,center_y,nPx_img_x,nPx_img_y])
    position_pups = np.array([r_left_1,r_left_2,r_right_1,r_right_2,nPx_img_x,nPx_img_y])

    return pupil_ZWFS,position_pups


def extract_pupil(img,position_pup):
    nPx_img = img.shape[0]
    r = int(0.1792*nPx_img)
    nPx_pup = 2*r
    x_center = int(nPx_img/2+1)#nPx_pup*shannon
    y_center = int(nPx_img/2+1)#nPx_pup*shannon
    # left top pupil -----
    r_lt_1 = int(x_center-nPx_pup*position_pup[0])
    r_lt_2 = int(y_center-nPx_pup*position_pup[1])
    # left bottom pupil -----
    r_lb_1 = int(x_center-nPx_pup*position_pup[2])
    r_lb_2 = int(y_center-nPx_pup*position_pup[3])
    # right top pupil -----
    r_rt_1 = int(x_center-nPx_pup*position_pup[4])
    r_rt_2 = int(y_center-nPx_pup*position_pup[5])
    # right bottom pupil -----
    r_rb_1 = int(x_center-nPx_pup*position_pup[6])
    r_rb_2 = int(y_center-nPx_pup*position_pup[7])
    # pupil crop
    pup_lt = img[r_lt_2-r+1:r_lt_2+r+1,r_lt_1-r+1:r_lt_1+r+1]
    pup_lb = img[r_lb_2-r+1:r_lb_2+r+1,r_lb_1-r+1:r_lb_1+r+1]
    pup_rt = img[r_rt_2-r+1:r_rt_2+r+1,r_rt_1-r+1:r_rt_1+r+1]
    pup_rb = img[r_lb_2-r+1:r_lb_2+r+1,r_lb_1-r+1:r_lb_1+r+1]
    pupil = 1/4*(pup_lt+pup_lb+pup_rt+pup_lt+pup_rb)
    # Remove outside pixel
    pupil_footprint = makePupil(int(pupil.shape[0]/2),pupil.shape[0])
    pupil = pupil*pupil_footprint
    return pupil

def make_SEAL_aperture(nPx,Dscaling=1.,Gscaling=1.,Sscaling=1.,normalized=True):
    """
    Make IRIS AO aperture using hcipy
    Return:
     - full pupil aperture 2-D array of Shape (2*aperture_r,2*aperture_r)
     - segment apertures: 3D array of Shape (37,2*aperture_r,2*aperture_r)
        where 37 is the number of segments
    Note: with current smaller SEAL pupil, best values (from ZWFS images) are:
        Dscaling=0.96,Gscaling=1.2,Sscaling=1.
    """
    # IRIS AO diameter:
    pupil_diameter = 3.6e-3 * Dscaling # [m] Dscaling to scale diameter (<1 to decrease and > 1 to increase)
    segment_size_s2s = 0.606e-3 * Sscaling # [m] (0.606 mm side to side) Sscaling to scale side  (<1 to decrease and > 1 to increase)
    irisao_segment_gap_size = 12e-6  * Gscaling # m (in Maaike's code: 7.4e-17 ?) Gscaling to scale gap size  (<1 to decrease and > 1 to increase)

    if normalized == True:
        irisao_segment_gap_size/=pupil_diameter
        segment_size_s2s/=pupil_diameter
        pupil_diameter/=pupil_diameter

    # Number of full rings of hexagons around central segment:
    num_rings = 3 
    
    irisao_aperture, irisao_segments = make_hexagonal_segmented_aperture(num_rings,
        segment_size_s2s, irisao_segment_gap_size, starting_ring=0, return_segments=True)

    pupil_grid = make_pupil_grid(nPx, pupil_diameter)
    
    telescope_pupil = evaluate_supersampled(irisao_aperture, pupil_grid, 8)
    telescope_pupil_circ = telescope_pupil*circular_aperture(pupil_diameter)(pupil_grid)
    telescope_pupil_circ[np.where(telescope_pupil_circ<1)] = 0
    
    segments_pupil = evaluate_supersampled(irisao_segments, pupil_grid, 8)
    segments_pupil_circ = segments_pupil*circular_aperture(pupil_diameter)(pupil_grid)
    segments_pupil_circ[np.where(segments_pupil_circ<1)] = 0

    return np.array(telescope_pupil_circ.shaped), np.array(segments_pupil_circ.shaped)

def make_keck_aperture(nPx,normalized=True, with_secondary=True,with_spiders=True, with_segment_gaps=True, gap_padding=1., segment_transmissions=1, return_segments=True):
    
    ##### CREAT FIELD OBJECT #########
    
    pupil_diameter = 10.95 #m actual circumscribed diameter
    actual_segment_flat_diameter = np.sqrt(3)/2 * 1.8 #m actual segment flat-to-flat diameter
    # iris_ao_segment = np.sqrt(3)/2 * .7 mm (~.606 mm)
    actual_segment_gap = 0.05 #m actual gap size between segments
    # (3.5 - (3 D + 4 S)/6 = iris_ao segment gap (~7.4e-17)
    spider_width = 1*5e-2#Value from Sam; Jules value: 0.02450 #m actual strut size
    if normalized: 
        actual_segment_flat_diameter/=pupil_diameter
        actual_segment_gap/=pupil_diameter
        spider_width/=pupil_diameter
        pupil_diameter/=pupil_diameter
    if with_segment_gaps == True:
        segment_gap = actual_segment_gap * gap_padding #padding out the segmentation gaps so they are visible and not sub-pixel
    else:
        segment_gap = 0.
        actual_segment_gap = 0.
    segment_transmissions = 1.

    segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
    segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter #segment circumscribed diameter

    num_rings = 3 #number of full rings of hexagons around central segment

    segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)
    segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

    segment = hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=True)

    segmentation, segments = segmented_aperture
    
    if with_spiders == True:
        spider1 = make_spider_infinite([0, 0], 0, spider_width)
        spider2 = make_spider_infinite([0, 0], 60, spider_width)
        spider3 = make_spider_infinite([0, 0], 120, spider_width)
        spider4 = make_spider_infinite([0, 0], 180, spider_width)
        spider5 = make_spider_infinite([0, 0], 240, spider_width)
        spider6 = make_spider_infinite([0, 0], 300, spider_width)

        def segment_with_spider(segment):
            return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
        
        segments = [segment_with_spider(s) for s in segments]
    
    contour = make_segmented_aperture(segment, segment_positions)
    conversion=(10.95/(2*12.05/1000))

    central_obscuration_diameter=0.25

    def func(grid):
        #co=circular_aperture(central_obscuration_diameter)
        if with_spiders == True:
            ap = contour(grid) * spider1(grid) * spider2(grid) * spider3(grid)* spider4(grid) * spider3(grid)* spider5(grid) * spider6(grid)  #* co(grid)
        else:
            ap=contour(grid)
        co=circular_aperture(central_obscuration_diameter)(grid)
        if with_secondary == True:
            ap[co==1]=0
        else:
            ap[co==1]=1
        res = (ap )
        #res*=co
        return Field(res, grid)
 
    ##### CREATE ARRAYS #########
    
    pupil_grid = make_pupil_grid(nPx)
    pupil_keck = func(pupil_grid).reshape((nPx,nPx))
    
    segments_keck = np.zeros((len(segments),nPx,nPx))
    for i in range(0,len(segments)):
        segments_keck[i,:,:] = segments[i](pupil_grid).reshape((nPx,nPx))
    
    return pupil_keck, segments_keck
    
def makeRing(Rpx_outer,Rpx_inner,nPx,set_nans=False):

    outer = makePupil(Rpx_outer,nPx)
    inner = makePupil(Rpx_inner,nPx)
    ring = np.copy(outer)
    ring[np.where(inner==1)] = 0
    if set_nans == True:
        ring[np.where(ring==0)] = np.nan

    return ring


#%% ====================================================================
#   ============  Actuators registration with Pyramid WFS===============
#   ====================================================================


def build_pokeMatrix(wfs,dm,nRecPoke = None):
    """
    This fonction allows to register DM actuators w.r.t to reconstructed phase maps
     - Takes push-pull of 2 actuators and fit gaussian to determine position
     - Take waffle image and other pokes images to check results
    """
    #------------------------------ PARAMETERS --------------------------------------------
    if nRecPoke is None:
        nRecPoke = wfs.nIterRec # NUMBER OF ITERATION to reconstruct 2 main pokes used for registration
    nRec_offset = 5 # for other pokes
    nRec_waffle = 5 # FOR WAFFLE
    threshold_act = 10 # to select actuators only in a circle in percent of diameter
    if dm.model == 'mems':
        # MEMS DM
        if wfs.model == 'PWFS':
            amp = 0.05 # AMPLITUDE POKE
            amp_waffle = 0.2 # AMPLITUDE WAFFLE
            x_dm_poke = np.array([15,10,21,9])
            y_dm_poke = np.array([15,11,16,23])
        elif wfs.model == 'ZWFS':
            amp = 0.02 # AMPLITUDE POKE
            amp_waffle = 0.05 # AMPLITUDE WAFFLE
            x_dm_poke = np.array([15,21,21,9])
            y_dm_poke = np.array([15,21,16,23])
    elif dm.model == 'alpao':
        # ALPAO DM
        amp = 0.01 # AMPLITUDE POKE
        amp_waffle = 0.05 # AMPLITUDE WAFFLE
        x_dm_poke = np.array([5,3,7,7])
        y_dm_poke = np.array([5,7,5,7])
    elif dm.model == 'slm':
        # SLM DM
        amp = 0.4 # AMPLITUDE POKE
        amp_waffle = 0.5 # AMPLITUDE WAFFLE
        x_dm_poke = np.array([int(dm.nAct/2),int(dm.nAct/4),int(2.8*dm.nAct/4),int(2.8*dm.nAct/4)])
        y_dm_poke = np.array([int(dm.nAct/2),int(2.8*dm.nAct/4),int(dm.nAct/2),int(2.8*dm.nAct/4)])
    number_valid_actuators = np.copy(dm.valid_actuators_number)
    # -------------------------------------------------------------------------------------
    
    
    #%% =========================== POKES ================================
    
    # Send poke, save signal, reconstruct, and fit with Gaussian
    poke = np.zeros((wfs.nPx,wfs.nPx,x_dm_poke.shape[0]))
    # -------  RECORD POKES ------
    for i in range(0,x_dm_poke.shape[0]):
        # ---- Send poke -------
        dm.pokeAct(amp,[x_dm_poke[i],y_dm_poke[i]])
        time.sleep(1)
        img_push = wfs.getImage()
        img_push[img_push<0] = 0
        dm.pokeAct(-amp,[x_dm_poke[i],y_dm_poke[i]])
        time.sleep(1)
        img_pull = wfs.getImage()
        img_pull[img_pull<0] = 0
        dm.setFlatSurf()
        # ---- Reconstruct poke -------
        print('Reconstruction of the poke number: ',i)
        if i<2:
            nRec = nRecPoke
        else:
            nRec = nRec_offset
            
        wfs.reconNonLinear(img_push,nRec)
        push = wfs.phase_rec
        wfs.reconNonLinear(img_pull,nRec)
        pull = wfs.phase_rec
        poke[:,:,i] = (push-pull)/(2*amp)
        
    # ------- Gaussian Fit 2 first pokes ------
    poke_rec = np.zeros((wfs.nPx,wfs.nPx,2))
    A_poke = []
    pos_act_x = []
    pos_act_y = []
    sigma_poke = []
    for i in range(0,2):
        # ----- FITTING A GAUSSIAN on first 2 pokes---------
        nPx = poke.shape[0]
        func = lambda param: np.ravel(gauss2d(param[0],param[1],param[2],param[3],nPx)-poke[:,:,i])
        if i == 0:
            post_x_guess = nPx/2
            post_y_guess = nPx/2
        elif i == 1:
            post_x_guess = nPx/4
            post_y_guess = nPx/4
        param0 = np.array([-20,post_x_guess,post_y_guess,10])
        param_poke = leastsq(func,param0)
        A_poke.append(param_poke[0][0])
        pos_act_x.append(param_poke[0][1])
        pos_act_y.append(param_poke[0][2])
        sigma_poke.append(param_poke[0][3])
        poke_rec[:,:,i] = gauss2d(A_poke[i],pos_act_x[i],pos_act_y[i],sigma_poke[i],nPx)
    
    print(A_poke)
    # --- Show Reconstructed poke and Gaussian Fit results --------
    plt.figure()
    plt.subplot(121)
    plt.imshow(poke[:,:,0]+poke[:,:,1])
    plt.subplot(122)
    plt.imshow(poke_rec[:,:,0]+poke_rec[:,:,1])
    plt.show(block=True)
    
    #%% ======================== WAFFLE MODE TO CHECK GRID ================================
    # ----- Send WAFFLE ------------
    if nPx > 140:
        dm.pokeWaffle(amp_waffle)
        time.sleep(1)
        img_push = wfs.getImage()
        dm.pokeWaffle(-amp_waffle)
        time.sleep(1)
        img_pull = wfs.getImage()
        dm.setFlatSurf()
    else:
        dm.pokeWaffleLarge(amp_waffle)
        time.sleep(1)
        img_push = wfs.getImage()
        dm.pokeWaffleLarge(-amp_waffle)
        time.sleep(1)
        img_pull = wfs.getImage()
        dm.setFlatSurf()
    #  ------ RECONSTRUCT WAFFLE -------
    wfs.reconNonLinear(img_push,nRec_waffle)
    push = wfs.phase_rec
    wfs.reconNonLinear(img_pull,nRec_waffle)
    pull = wfs.phase_rec
    phi_waffle = (push-pull)/(2*amp_waffle)
        
    #%% =========================== TRANSFORMATION DM -> WFS ================================
    # central pixel in wfs map
    y_dm_0 = x_dm_poke[0]
    x_dm_0 = y_dm_poke[0]
    x_wfs_0 = pos_act_x[0]
    y_wfs_0 = pos_act_y[0]

    # COMPUTE TRANSFORMATION PARAMETERS
    v_dm = np.array([x_dm_poke[0]-x_dm_poke[1],y_dm_poke[0]-y_dm_poke[1]])
    v_wfs = np.array([pos_act_y[0]-pos_act_y[1],pos_act_x[0]-pos_act_x[1]])
    print(pos_act_x)
    print(pos_act_y)
    # scaling along X ---------
    alpha = v_wfs[1]/v_dm[1]
    print('ALPHA',alpha)
    # scaling along Y ---------
    beta = -v_wfs[0]/v_dm[0]
    print('BETA',beta)
    # rotation ---------
    ps = np.dot(v_dm,v_wfs)/(np.linalg.norm(v_dm)*np.linalg.norm(v_wfs)) # normalized scalar product
    theta = 0*np.arccos(ps)
    print('THETA',theta)
    xx_wfs,yy_wfs = transformation_dm2wfs(alpha,beta,theta,x_wfs_0,y_wfs_0,x_dm_0,y_dm_0,dm.xx_dm,dm.yy_dm)


    #%% ======================== FINAL SELECTION ON PLOT ================================
    time.sleep(1)
    # ---- Waffle ----
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8)
    plt.subplots_adjust(left=0.1,right=0.8,bottom=0.1)
    ax.set_aspect("equal")
    axcolor = 'skyblue'
    plt.subplot(121)
    plt.imshow(phi_waffle)
    grid_wfs = plt.scatter(xx_wfs,yy_wfs,edgecolor ="orange",facecolors='none')
    #ax.scatter(xx_wfs_valid,yy_wfs_valid,edgecolor ="yellow",facecolors='none')
    # ---- POKE ---
    plt.subplot(122)
    plt.imshow(np.sum(poke,axis=2))
    grid_wfs_2 = plt.scatter(xx_wfs,yy_wfs,edgecolor ="orange",facecolors='none')
    #plt.scatter(xx_wfs_valid,yy_wfs_valid,edgecolor ="yellow",facecolors='none')
    poke_pos = plt.scatter(xx_wfs[int(number_valid_actuators[x_dm_poke[0],y_dm_poke[0]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[0],y_dm_poke[0]]-1)],edgecolor ="red",facecolors='none')
    poke_2_pos = plt.scatter(xx_wfs[int(number_valid_actuators[x_dm_poke[1],y_dm_poke[1]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[1],y_dm_poke[1]]-1)],edgecolor ="red",facecolors='none')
    poke_3_pos = plt.scatter(xx_wfs[int(number_valid_actuators[x_dm_poke[2],y_dm_poke[2]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[2],y_dm_poke[2]]-1)],edgecolor ="red",facecolors='none')
    poke_4_pos = plt.scatter(xx_wfs[int(number_valid_actuators[x_dm_poke[3],y_dm_poke[3]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[3],y_dm_poke[3]]-1)],edgecolor ="red",facecolors='none')

    # ------- SLIDERS --------
    slider_alpha_ax = plt.axes([0.2, 0.9, 0.15, 0.03], facecolor=axcolor)
    slider_alpha = Slider(slider_alpha_ax, 'Alpha', alpha-2, alpha+2, alpha)# big pupil 0.2908

    slider_beta_ax = plt.axes([0.2, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_beta = Slider(slider_beta_ax, 'Beta', beta-2, beta+2, beta)# big pupil 0.2908

    slider_theta_ax = plt.axes([0.4, 0.85, 0.15, 0.03], facecolor=axcolor)
    slider_theta = Slider(slider_theta_ax, 'Theta', -np.pi, np.pi, theta)

    slider_xwfs0_ax = plt.axes([0.6, 0.9, 0.15, 0.03], facecolor=axcolor)
    slider_xwfs0 = Slider(slider_xwfs0_ax, 'X0_WFS', 0.8*x_wfs_0, 1.2*x_wfs_0, x_wfs_0)# big pupil 0.2908
    
    slider_ywfs0_ax = plt.axes([0.6, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_ywfs0 = Slider(slider_ywfs0_ax, 'Y0_WFS',0.8*y_wfs_0, 1.2*y_wfs_0, y_wfs_0)# big pupil 0.2908
    
    def update(val):
        # ----- UPDATE X and Y position -----
        alpha = slider_alpha.val
        beta = slider_beta.val
        theta = slider_theta.val
        x_wfs_0 = slider_xwfs0.val
        y_wfs_0 = slider_ywfs0.val
        xx_wfs,yy_wfs = transformation_dm2wfs(alpha,beta,theta,x_wfs_0,y_wfs_0,x_dm_0,y_dm_0,dm.xx_dm,dm.yy_dm)
        grid_wfs.set_offsets(np.transpose(np.array([xx_wfs,yy_wfs])))
        grid_wfs_2.set_offsets(np.transpose(np.array([xx_wfs,yy_wfs])))
        poke_pos.set_offsets(np.array([xx_wfs[int(number_valid_actuators[x_dm_poke[0],y_dm_poke[0]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[0],y_dm_poke[0]]-1)]]))
        poke_2_pos.set_offsets(np.array([xx_wfs[int(number_valid_actuators[x_dm_poke[1],y_dm_poke[1]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[1],y_dm_poke[1]]-1)]]))
        poke_3_pos.set_offsets(np.array([xx_wfs[int(number_valid_actuators[x_dm_poke[2],y_dm_poke[2]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[2],y_dm_poke[2]]-1)]]))
        poke_4_pos.set_offsets(np.array([xx_wfs[int(number_valid_actuators[x_dm_poke[3],y_dm_poke[3]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[3],y_dm_poke[3]]-1)]]))
        fig.canvas.draw_idle()
    
    slider_alpha.on_changed(update)
    slider_beta.on_changed(update)
    slider_theta.on_changed(update)
    slider_xwfs0.on_changed(update)
    slider_ywfs0.on_changed(update)
    
    plt.show(block = True)
    
    #%% ======================== SELECTING WFS VALID ACTUATORS ================================
    xx_wfs_valid = []
    yy_wfs_valid = []
    validWFS_map = np.zeros((dm.nAct,dm.nAct))
    for i in range(0,dm.nAct):
        for j in range(0,dm.nAct):
            if dm.valid_actuators_map[i,j] == 1:
                if ((xx_wfs[int(dm.valid_actuators_number[i,j]-1)]-nPx/2)**2+(yy_wfs[int(dm.valid_actuators_number[i,j]-1)]-nPx/2)**2)< (threshold_act*nPx/2)**2:
                    # Add in valid list
                    xx_wfs_valid.append(xx_wfs[int(dm.valid_actuators_number[i,j]-1)])
                    yy_wfs_valid.append(yy_wfs[int(dm.valid_actuators_number[i,j]-1)])
                    # add in Valid MAP
                    validWFS_map[i,j] = 1
    xx_wfs_valid = np.array(xx_wfs_valid)	
    yy_wfs_valid = np.array(yy_wfs_valid)
    
    #%% ======================== CREATE POKE MATRIX ================================
    poke_matrix = build_poke_matrix_wfs(xx_wfs_valid,yy_wfs_valid,A_poke[0],sigma_poke[0],wfs.nPx,wfs.pupil_footprint)
    return poke_matrix,validWFS_map

#%% ====================================================================
#   ============  unwrapping PHASE (code from internet)  ===============
#   ====================================================================


def phase_unwrap_ref(psi, weight, kmax=100):    
    """
    A weighed phase unwrap algorithm implemented in pure Python
    author: Tobias A. de Jong
    Based on:
    Ghiglia, Dennis C., and Louis A. Romero. 
    "Robust two-dimensional weighted and unweighted phase unwrapping that uses 
    fast transforms and iterative methods." JOSA A 11.1 (1994): 107-117.
    URL: https://doi.org/10.1364/JOSAA.11.000107
    and an existing MATLAB implementation:
    https://nl.mathworks.com/matlabcentral/fileexchange/60345-2d-weighted-phase-unwrapping
    Should maybe use a scipy conjugate descent.
    """
    # vector b in the paper (eq 15) is dx and dy
    dx = _wrapToPi(np.diff(psi, axis=1))
    dy = _wrapToPi(np.diff(psi, axis=0))
    
    # multiply the vector b by weight square (W^T * W)
    WW = weight**2
    
    # See 3. Implementation issues: eq. 34 from Ghiglia et al.
    # Improves number of needed iterations. Different from matlab implementation
    WWx = np.minimum(WW[:,:-1], WW[:,1:])
    WWy = np.minimum(WW[:-1,:], WW[1:,:])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = np.linalg.norm(rk);

    # start the iteration
    eps = 1e-9
    k = 0
    phi = np.zeros_like(psi);
    while (~np.all(rk == 0.0)):
        zk = solvePoisson(rk);
        k += 1
        
        # equivalent to (rk*zk).sum()
        rkzksum = np.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum  / rkzkprevsum
            pk = zk + betak * pk;

        # save the current value as the previous values
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / np.tensordot(pk, Qpk)
        phi +=  alphak * pk;
        rk -=  alphak * Qpk;

        # check the stopping conditions
        if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
            break;
        #print(np.linalg.norm(rk), normR0)
    print(k, rk.shape)
    return phi

def solvePoisson(rho):
    """Solve the poisson equation "P phi = rho" using DCT
    """
    dctRho = dctn(rho);
    N, M = rho.shape;
    I, J = np.ogrid[0:N,0:M]
    with np.errstate(divide='ignore'):
        dctPhi = dctRho / 2 / (np.cos(np.pi*I/M) + np.cos(np.pi*J/N) - 2)
    dctPhi[0, 0] = 0 # handling the inf/nan value
    # now invert to get the result
    phi = idctn(dctPhi);
    return phi

def solvePoisson_precomped(rho, scale):
    """Solve the poisson equation "P phi = rho" using DCT
    Uses precomputed scaling factors `scale`
    """
    dctPhi = dctn(rho) / scale
    # now invert to get the result
    phi = idctn(dctPhi, overwrite_x=True)
    return phi

def precomp_Poissonscaling(rho):
    N, M = rho.shape;
    I, J = np.ogrid[0:N,0:M]
    scale = 2 * (np.cos(np.pi*I/M) + np.cos(np.pi*J/N) - 2)
    # Handle the inf/nan value without a divide by zero warning:
    # By Ghiglia et al.:
    # "In practice we set dctPhi[0,0] = dctn(rho)[0, 0] to leave
    #  the bias unchanged"
    scale[0, 0] = 1. 
    return scale

def applyQ(p, WWx, WWy):
    """Apply the weighted transformation (A^T)(W^T)(W)(A) to 2D matrix p"""
    # apply (A)
    dx = np.diff(p, axis=1)
    dy = np.diff(p, axis=0)

    # apply (W^T)(W)
    WWdx = WWx * dx;
    WWdy = WWy * dy;
    
    # apply (A^T)
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)
    Qp = WWdx2 + WWdy2
    return Qp


def _wrapToPi(x):
    r = (x+np.pi)  % (2*np.pi) - np.pi
    return r

def phase_unwrap(psi, weight=None, kmax=100):
    """
    Unwrap the phase of an image psi given weights weight
    This function uses an algorithm described by Ghiglia and Romero
    and can either be used with or without weight array.
    It is especially suited to recover a unwrapped phase image
    from a (noisy) complex type image, where psi would be 
    the angle of the complex values and weight the absolute values
    of the complex image.
    """

    # vector b in the paper (eq 15) is dx and dy
    dx = _wrapToPi(np.diff(psi, axis=1))
    dy = _wrapToPi(np.diff(psi, axis=0))
    
    # multiply the vector b by weight square (W^T * W)
    if weight is None:
        # Unweighed case. will terminate in 1 round
        WW = np.ones_like(psi)
    else:
        WW = weight**2
    
    # See 3. Implementation issues: eq. 34 from Ghiglia et al.
    # Improves number of needed iterations. Different from matlab implementation
    WWx = np.minimum(WW[:,:-1], WW[:,1:])
    WWy = np.minimum(WW[:-1,:], WW[1:,:])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = np.linalg.norm(rk);

    # start the iteration
    eps = 1e-9
    k = 0
    phi = np.zeros_like(psi)
    scaling = precomp_Poissonscaling(rk)
    while (~np.all(rk == 0.0)):
        zk = solvePoisson_precomped(rk, scaling);
        k += 1
        
        # equivalent to (rk*zk).sum()
        rkzksum = np.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum  / rkzkprevsum
            pk = zk + betak * pk;

        # save the current value as the previous values
        
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / np.tensordot(pk, Qpk)
        phi +=  alphak * pk;
        rk -=  alphak * Qpk;

        # check the stopping conditions
        if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
            break;
    return phi
