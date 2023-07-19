#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------- SIMULATION linearity scatter ZERNIKE -----------------------------
Script to do linerity Scatter plot for Von Karman PSD
"""

#############################################

import sys
sys.path.append(r"/home/lab/libSEAL")
from wfAffectors import *
from wfSensors import *
from wfControllers import *
from cameras import *
from tools import *
from turbulence import *


plt.close('all')



# ----- CREATE ZWFS object ----
pupil_im = np.load('pupil_im.npy') # Load SEAL pupil for more realistic simulation
position_pups = np.load('position_pups.npy') # to match SEAL dimensions
shannon = 4 # to well oversample the dimple (2 could be enough tough)
wavelength = 635 # in nm
diameter = np.array([1,1]) # dimple diameter in L/D - Left pupil and right pupil
depth = np.array([-np.pi/2,np.pi/2]) # dimple phase shift in radians - Left pupil and right pupil
ZWFS = zernikeWFS(pupil_im,position_pups,diameter,depth,shannon,wavelength,0) # Create object - 0 = simulation mode

####### PARAMETERS ###########
nModes = 300
nIter = 20 # Iteration for reconstruction 
nPSD = 1 # Number of phase screen
errorAMP = np.linspace(0,1,10)
zer_modes = zernikeBasis(nModes,ZWFS.nPx)
phase2modes = np.linalg.pinv(zer_modes)

# ---- Calibrate ZWFS ----
ZWFS.calibrateSimu(zer_modes) # Calibration - Interaction Matrix


# ------ Define Turbulence --------
telescope_diameter = 8
fried_parameter = 0.25
velocity = 8
outer_scale = 40
timeSampling = 0.001
turbu = atm(telescope_diameter,wavelength*1e-09,fried_parameter,velocity,outer_scale,timeSampling,ZWFS.nPx)

# ----- Save Data ------
E_1 = np.zeros((nPSD,errorAMP.shape[0]))
E_2 = np.zeros((nPSD,errorAMP.shape[0]))
E_3 = np.zeros((nPSD,errorAMP.shape[0]))
E_4 = np.zeros((nPSD,errorAMP.shape[0]))
E_5 = np.zeros((nPSD,errorAMP.shape[0]))
E_6 = np.zeros((nPSD,errorAMP.shape[0]))
E_7 = np.zeros((nPSD,errorAMP.shape[0]))
E_8 = np.zeros((nPSD,errorAMP.shape[0]))
E_9 = np.zeros((nPSD,errorAMP.shape[0]))
E_10 = np.zeros((nPSD,errorAMP.shape[0]))
E_11 = np.zeros((nPSD,errorAMP.shape[0]))


i = 0
for err in errorAMP:
    print('--------  ',str(err),'  --------')	
    for k in range(0,nPSD):
        print('ERR --------  ',str(err),'  --------')	
        print('nPSD --------  ',str(k),'  --------')
        turbu.setTurbu(k)
        phaseTurbu = np.copy(turbu.phase_screen)	
        phaseTurbu_filtered = np.dot(zer_modes,np.dot(phase2modes,phaseTurbu.ravel()))
        phaseTurbu_filtered = phaseTurbu_filtered.reshape((ZWFS.nPx,ZWFS.nPx))
        phaseTurbu_filtered = phaseTurbu_filtered-np.sum(phaseTurbu_filtered)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        e_rms = error_rms(phaseTurbu_filtered,np.zeros((ZWFS.nPx,ZWFS.nPx)),ZWFS.pupil_footprint)
        phaseTurbu_filtered_nPx = phaseTurbu_filtered/e_rms*err # normalization


        #############################################
        ######### Optical propagation in ZWFS #######
        #############################################

        img = ZWFS.cropImg(ZWFS.getImageSimu(phaseTurbu_filtered_nPx))

        #############################################
        #########     Reconstruction          #######
        #############################################

        ############ LINEAR APPROACHES ############

        # 1) ----- Linear Reconstruction Interaction Matrix-----
        ZWFS.reconLinear(img)
        # Remove piston
        phase = ZWFS.phase_rec-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_1[k,i] = e_lin_rms

        # 2) ----- Linear Reconstruction Model based 1 pupil + 0 iterations -----
        ZWFS.pupilRec = 'both'
        ZWFS.nIterRec = 0
        ZWFS.reconLinearModel(img)
        # Remove piston
        phase = ZWFS.phase_rec_right-np.sum(ZWFS.phase_rec_right)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_2[k,i] = e_lin_rms

        # 3) ----- Linear Reconstruction Model based 2 pupil + 0 iterations -----
        # Remove piston
        phase = ZWFS.phase_rec-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_3[k,i] = e_lin_rms
        
        # 4) ----- Linear Reconstruction Model based 1 pupil + nIter iterations -----
        ZWFS.pupilRec = 'both'
        ZWFS.nIterRec = nIter
        ZWFS.reconLinearModel(img)
        # Remove piston
        phase = ZWFS.phase_rec_right-np.sum(ZWFS.phase_rec_right)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_4[k,i] = e_lin_rms

        # 5) ----- Linear Reconstruction Model based 2 pupils + nIter iterations -----
        # Remove piston
        phase = ZWFS.phase_rec-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_5[k,i] = e_lin_rms

        ############ ASIN APPROACHES ############

        # 6) ----- Asin Reconstruction Model based 1 pupil + 0 iterations -----
        ZWFS.pupilRec = 'both'
        ZWFS.algo = 'JPL'
        ZWFS.nIterRec = 0
        ZWFS.reconNonLinear(img)
        # Remove piston
        phase = ZWFS.phase_rec_right-np.sum(ZWFS.phase_rec_right)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_6[k,i] = e_lin_rms

        # 7) ----- Asin Reconstruction Model based 2 pupil + 0 iterations -----
        # Remove piston
        phase = ZWFS.phase_rec-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_7[k,i] = e_lin_rms
        
        # 8) ----- Asin Reconstruction Model based 1 pupil + nIter iterations -----
        ZWFS.pupilRec = 'both'
        ZWFS.algo = 'JPL'
        ZWFS.nIterRec = nIter
        ZWFS.reconNonLinear(img)
        # Remove piston
        phase = ZWFS.phase_rec_right-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_8[k,i] = e_lin_rms

        # 9) ----- Asin Reconstruction Model based 2 pupils + nIter iterations -----
        # Remove piston
        phase = ZWFS.phase_rec-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_9[k,i] = e_lin_rms

        ############ GS APPROACHES ############

        # 10) ----- GS Reconstruction Model based 1 pupil + nIter iterations -----
        ZWFS.pupilRec = 'both'
        ZWFS.algo = 'GS'
        ZWFS.nIterRec = nIter
        ZWFS.reconLinearModel(img)
        # Remove piston
        phase = ZWFS.phase_rec_right-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_10[k,i] = e_lin_rms

        # 11) ----- GS Reconstruction Model based 2 pupils + nIter iterations -----
        # Remove piston
        phase = ZWFS.phase_rec-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_11[k,i] = e_lin_rms

    i = i + 1
        
   
## ====================== SAVE DATA ================
'''
np.save('./results/linearityScatter_lin_bis.npy',E_lin_list)
np.save('./results/linearityScatter_lin_modu_bis.npy',E_lin_modu_list)
np.save('./results/linearityScatter_GS_25_bis.npy',E_GS_25_list)
np.save('./results/linearityScatter_GS_200_bis.npy',E_GS_200_list)
np.save('./results/linearityScatter_GS_1000_bis.npy',E_GS_1000_list)
np.save('./results/linearityScatter_error_list_bis.npy',errorAMP)
'''
## PLOT DATA ###########
plt.close('all')
plt.figure()

x = errorAMP
y =E_1[0,:]
plt.plot(x,y,color='red',label='IntMat')

y =E_4[0,:]
plt.plot(x,y,color='blue',label='linear model : 1 pupil + 10 iteration')

y =E_8[0,:]
plt.plot(x,y,color='purple',label='asin model : 1 pupil + 10 iteration')

#y =E_9[0,:]
#plt.plot(x,y,color='green',label='asin model : 2 pupil + 10 iteration')

plt.legend()    
plt.xlabel('Input')
plt.ylabel('Error Reconstruction')
plt.show(block=False)





