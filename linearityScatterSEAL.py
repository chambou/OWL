#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------- SEAL - On Bench linearity scatter ZERNIKE -----------------------------
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



####### PARAMETERS ###########
nModes = 300
nIter = 30 # Iteration for reconstruction 
nPSD = 1 # Number of phase screen
errorAMP = np.linspace(0,1,30)

####### SLM ########
slm = SLM()
slm.setPhase(np.zeros((slm.nPx,slm.nPx)))
pup_slm = makePupil(slm.nPx/2,slm.nPx)
zer_modes = zernikeBasis(nModes,slm.nPx)
phase2modes = np.linalg.pinv(zer_modes)

# ---- Calibrate ZWFS ----
#ZWFS.calibrate(zer_modes) # Calibration - Interaction Matrix


# ------ Define Turbulence --------
telescope_diameter = 8
fried_parameter = 0.25
velocity = 8
outer_scale = 40
timeSampling = 0.001
turbu = atm(telescope_diameter,wavelength*1e-09,fried_parameter,velocity,outer_scale,timeSampling,slm.nPx)

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
        phaseTurbu_filtered = phaseTurbu_filtered.reshape((slm.nPx,slm.nPx))
        e_rms = error_rms(phaseTurbu_filtered,np.zeros((slm.nPx,slm.nPx)),pup_slm)
        phaseTurbu_filtered = phaseTurbu_filtered/e_rms*err # normalization
        phaseTurbu_filtered_nPx = scipy.ndimage.zoom(phaseTurbu_filtered,ZWFS.nPx/slm.nPx)
        phaseTurbu_filtered_nPx = phaseTurbu_filtered_nPx-np.sum(phaseTurbu_filtered_nPx)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        e_rms = error_rms(phaseTurbu_filtered_nPx,np.zeros((ZWFS.nPx,ZWFS.nPx)),ZWFS.pupil_footprint)
        phaseTurbu_filtered_nPx = phaseTurbu_filtered_nPx/e_rms*err # normalization


        #############################################
        ######### Optical propagation in ZWFS #######
        #############################################
        
        # ---- Send phase SLM ---
        slm.setPhase(-np.flipud(np.fliplr(phaseTurbu_filtered)))
        time.sleep(0.5)
        # ----- Get image on ZWFS ----
        img = ZWFS.getImage()

        #############################################
        #########     Reconstruction          #######
        #############################################

        ############ LINEAR APPROACHES ############
        '''
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
        '''
        ############ ASIN APPROACHES ############
        
        # 6) ----- Asin Reconstruction Model based 1 pupil + 0 iterations -----
        ZWFS.pupilRec = 'both'
        ZWFS.algo = 'JPL'
        ZWFS.nIterRec = 0
        ZWFS.reconNonLinear(img)
        ZWFS.phase_rec_right = ZWFS.phase_rec_right-phase0
        # Remove piston
        phase = ZWFS.phase_rec_right-np.sum(ZWFS.phase_rec_right)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_6[k,i] = e_lin_rms

        # 7) ----- Asin Reconstruction Model based 2 pupil + 0 iterations -----
        ZWFS.phase_rec = ZWFS.phase_rec-phase0
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
        ZWFS.phase_rec_right = ZWFS.phase_rec_right-phase0
        # Remove piston
        phase = ZWFS.phase_rec_right-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_8[k,i] = e_lin_rms

        # 9) ----- Asin Reconstruction Model based 2 pupils + nIter iterations -----
        ZWFS.phase_rec = ZWFS.phase_rec-phase0
        # Remove piston
        phase = ZWFS.phase_rec-np.sum(ZWFS.phase_rec)*ZWFS.pupil_footprint/np.sum(ZWFS.pupil_footprint)
        # Compute RMS error
        e_lin_rms = error_rms(phase,phaseTurbu_filtered_nPx,ZWFS.pupil_footprint)
        # Save error in array
        E_9[k,i] = e_lin_rms
        '''
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
        '''
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
y =E_6[0,:]
plt.plot(x,y,color='black',label='asin + one pupil + 0 iterations')

x = errorAMP
y =E_8[0,:]
plt.plot(x,y,color='red',label='asin + one pupil + 30 iterations')

x = errorAMP
y =E_7[0,:]
plt.plot(x,y,color='green',label='asin + both pupil + 0 iterations')

y =E_9[0,:]
plt.plot(x,y,color='blue',label='asin + both pupil + 30 iterations')

#y =E_8[0,:]
#plt.plot(x,y,color='purple',label='asin model : 1 pupil + 10 iteration')

#y =E_9[0,:]
#plt.plot(x,y,color='green',label='asin model : 2 pupil + 10 iteration')

plt.legend()    
plt.xlabel('Input')
plt.ylabel('Error Reconstruction')
plt.xlim([0,1])
plt.ylim([0,1])
plt.show(block=False)





