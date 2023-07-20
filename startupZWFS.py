#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to setup ZWFS
"""
import wfSensorsOWL
import datetime
import pickle
import os, glob
import sys
sys.path.append('/home/lab/scetre/')
from aptXYClient import *
sys.path.append('/home/lab/msalama/code/')
from ZWFS_tools import *

if __name__ == "__main__":
    
    cam.shutdown()

    #%% ################ ZWFS Calibration and Data Paths ##############
    ZWFS_calib_path = '/home/lab/libSEAL/calibration_files/ZWFS_calib/'
    date = datetime.datetime.now().strftime("%Y%m%d")
    savedir = '/home/lab/msalama/ZWFS_data/'+date+'/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        
    #%% ################ ZWFS CAMERA ##############
    binning = 1
    cam = blackFly_camera('ZWFS',binning)
    time.sleep(0.5)
    fps=cam.get_fps()
    expt=cam.get_exp()
    if expt == 0.0:
        expt = 1/fps*1000
    time.sleep(0.5)

    print("\n")
    print("Are the camera counts in the 30000 range?")
    ok = input("[y/n]: ")
    while ok == 'n':
        print("Try adjusting the exposure time.")
        expt = input("Exp time to try: ")
        expt = int(expt)
        cam.set_exp(expt)
        cam.get_exp()
        print("Are the camera counts in the 30000 range?")
        ok = input("[y/n]: ")

    take_new_dark = True
    ## Load existing dark:
    if take_new_dark == False:
        #cam.dark = np.load(ZWFS_calib_path+'Dark_'+str(int(expt))+'ms.npy')
        try:
            #cam.dark, dark_header = open_fits(ZWFS_calib_path+'Dark_'+str(int(expt))+'ms.fits')
            cam.dark = np.load(ZWFS_calib_path+'Dark_'+str(int(expt))+'ms.npy')
        except OSError as error:
            take_new_dark = True

    ## Take new dark:
    if take_new_dark == True:
        lightoff = input('Turn light source off then type [yes] ')
        if lightoff == 'yes':
            cam.getDark()
        save_dark = input('Save Dark? [yes or no] ')
        ## Save new dark:
        if save_dark == 'yes':
            np.save(ZWFS_calib_path+'Dark_'+str(int(expt))+'ms.npy',cam.dark)
        lighton = input('Turn light source back on then type [go]')
        if lighton == 'go':
            pass

    #%% ################ XY-motor stage ##############
    onmask_x, onmask_y, foc = np.load(ZWFS_calib_path+'XYpositions.npy') #10.045,5.05
    print("Current XY-motor positions:",round(getPosX(),3),round(getPosY(),3))

    # move off-mask:
    offmask_x, offmask_y = onmask_x - 0.5, onmask_y - 0.5 #9.5, 5.5
    moveXYstage(offmask_x,offmask_y)

    #%% ################ ZWFS ##############
    img_offMask = cam.get()
    # Position of pupil and values
    position_pups = np.load(ZWFS_calib_path+'position_pups.npy')
    left_x = position_pups[0]
    left_y = position_pups[1]
    right_x = position_pups[2]
    right_y = position_pups[3]
    nPx_img = 136
    P = makePupil(1.02*nPx_img/2,nPx_img)
    img_L = P*img_offMask[left_y-int(nPx_img/2):left_y+int(nPx_img/2),left_x-int(nPx_img/2):left_x+int(nPx_img/2)]
    img_R = P*img_offMask[right_y-int(nPx_img/2):right_y+int(nPx_img/2),right_x-int(nPx_img/2):right_x+int(nPx_img/2)]
    pupil_im = np.stack((img_L,img_R),axis=0)
    
    # ----- CREATE ZWFS object ----
    shannon = 4
    wavelength = 635 # in nm
    diameter = np.array([1,1])
    depth = np.array([np.pi/2,-np.pi/2])
    ZWFS = zernikeWFS_owl(pupil_im,position_pups,diameter,depth,shannon,wavelength,cam)

    # move on-mask:
    moveXYstage(onmask_x-0.01, onmask_y-0.01)
    
    # Focus off-set with PWFS branch
    if foc != 0:
        dm_mems.newFlat(dm_mems.pokeZernike(foc,4))


    #%% ############### Closed-Loop ################
    img = ZWFS.getImage()
    ZWFS.nIterRec = 5
    ZWFS.pupilRec = 'both'
    ZWFS.algo = 'JPL'
    ZWFS.doUnwrap = 0
    ZWFS.reconNonLinear(img)

    #%% ################ DM / WFS registration ##############
    dm = dm_mems
    #nIterRec = 1
    pokeMatrix = np.load(ZWFS_calib_path+'pokeMatrixMEMS_ZWFS_both.npy')
    validActuators = np.load(ZWFS_calib_path+'validActuatorsMEMS_ZWFS_both.npy')

    #%% ################ POKE MATRIX ##############
    thres = 1/50 # cutoff for linear reconstructor
    ZWFS.load_pokeMatrix(pokeMatrix)

    #%% ############### Controller ################
    controller = integrator(ZWFS,dm)
    controller.loopgain = 0.6
    controller.display = 1
    controller.closeLoop(5)
    bestflat = dm.flat_surf+controller.C

    dm.newFlat(bestflat)
    np.save(ZWFS_calib_path+'best_flat_'+dm.model+'_slm_ZWFS.npy',bestflat)
    np.save(ZWFS_calib_path+'best_flat_'+dm.model+'_slm_ZWFS_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.npy',bestflat)


