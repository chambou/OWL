#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to simulate segment phasing with ZWFS or nPWFS
"""

from wfSensors import *
from turbulence import *
from tools import *

plt.close('all')
#%% ################ ZWFS #########################################
pupil_im = np.load('pupil_im.npy') # Load SEAL pupil for more realistic simulation
pup = makePupil(pupil_im.shape[1]/2,pupil_im.shape[1]) # if you prefer to start with perfect full circular aperture
pup_im = np.stack((pup,pup),axis = 0)
position_pups = np.load('position_pups.npy') # to match SEAL dimensions
# ----- CREATE ZWFS object ----
shannon = 4 # to well oversample the dimple (2 could be enough tough)
wavelength = 635 # in nm
diameter = np.array([1,1]) # dimple diameter in L/D - Left pupil and right pupil
depth = np.array([-np.pi/2,np.pi/2]) # dimple phase shift in radians - Left pupil and right pupil
ZWFS = zernikeWFS(pup_im,position_pups,diameter,depth,shannon,wavelength,0) # Create object - 0 = simulation mode

#%% ################ Load turbulence ##########################
nPx = ZWFS.nPx # match ZWFS pixel resolution
telescope_diameter = 10
wavelength = 635*1e-09
fried_parameter = 2.5
velocity = 5
outer_scale = 40
timeSampling = 0.001
turbu = atm(telescope_diameter,wavelength,fried_parameter,velocity,outer_scale,timeSampling,nPx,300)

#%% ################ Generate turbulent phase screen ##########################
t = 0
turbu.setTurbu(t) # compute turbulence realization at time t
phase_turbu = turbu.phase_screen
# --- Project on first nModes zernikes modes ----
nModes = 300
zer_modes = zernikeBasis(nModes,ZWFS.nPx)
phase2modes = np.linalg.pinv(zer_modes)
phase_turbu_zer = np.dot(phase2modes,phase_turbu.ravel())
phase_turbu_zer[0] = 0 # remove piston
phase_turbu_filtered = np.dot(zer_modes,phase_turbu_zer)
phase_turbu_filtered = phase_turbu_filtered.reshape((ZWFS.nPx,ZWFS.nPx))

#%% ################ Get ZWFS image ##########################
img = ZWFS.getImageSimu(phase_turbu_filtered) #  gives the 0-padded image
img_cropped = ZWFS.cropImg(img) # crop image to match SEAL bench
img_PSF = ZWFS.getPSF(phase_turbu_filtered) # get PSF

# ---- Plot -----
plt.figure()
plt.subplot(121)
plt.imshow(img_PSF**0.5)
plt.title('PSF')
plt.subplot(122)
plt.imshow(img_cropped)
plt.title('ZWFS image')
plt.show(block=False)

'''
#%% ################## Recosntruction ###########################
ZWFS.pupilRec = 'both' # you can choose either: 'right', 'left' or 'both'
ZWFS.algo = 'JPL' # JPL = iterative arcsin reconstructor / cam also choose GS
ZWFS.nIterRec = 15 # number of iteration

ZWFS.reconNonLinear(img_cropped) # Reconstructiom

# ---- Plot -----
plt.figure(figsize=(12,3))
plt.subplot(131)
plt.imshow(phase_turbu_filtered*ZWFS.pupil_footprint)
plt.title('Input phase')
plt.colorbar()
plt.subplot(132)
plt.imshow(ZWFS.phase_rec)
plt.title('Reconstructed phase')
plt.colorbar()
plt.subplot(133)
plt.imshow(ZWFS.phase_rec_unwrap)
plt.title('Reconstructed phase - unwrap')
plt.colorbar()
plt.show(block=False)
'''
#%% Linear iterative reconstruction
ZWFS.pupilRec = 'left' # you can choose either: 'right', 'left' or 'both'
ZWFS.nIterRec = 0 # number of iteration

ZWFS.reconLinearModel(img_cropped) # Reconstructiom
'''
# ---- Plot -----
plt.figure(figsize=(12,3))
plt.subplot(131)
plt.imshow(phase_turbu_filtered*ZWFS.pupil_footprint)
plt.title('Input phase')
plt.colorbar()
plt.subplot(132)
plt.imshow(ZWFS.phase_rec)
plt.title('Reconstructed phase')
plt.colorbar()
plt.subplot(133)
plt.imshow(ZWFS.phase_rec_unwrap)
plt.title('Reconstructed phase - unwrap')
plt.colorbar()
plt.show(block=False)
'''
