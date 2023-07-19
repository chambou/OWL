#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from zwfs import *
from hcipy import *
import numpy as np
from matplotlib import pyplot as plt

from zwfs import *
from tools import *

def zwfs_multistep_reconstruct(Imeas, delta_phases, zwfs, aperture, wavelength, nphot, mask, phase_start=None, apply_unwrapping=False, lobasis=None):
	'''A multi-timestep reconstruction algorithm.

	Parameters
	----------
	Imeas : array like
		
	grid : Grid or None
		

	Returns
	-------

	'''
	if phase_start is None:
		phase_start = aperture.grid.zeros()

	grid = aperture.grid

	reduced_images = grid.zeros((Imeas.shape[0],))
	cosine_phases = grid.zeros((Imeas.shape[0],))
	sine_phases = grid.zeros((Imeas.shape[0],))

	for i, delta_phase in enumerate(delta_phases):
		# Create the model input electric field
		wf_est = Wavefront(aperture * np.exp(1j * (phase_start + delta_phase)), wavelength)
		wf_est.total_power = nphot
		I0 = wf_est.power
		
		# The zwfs output of the model electric field
		wf_zwfs = zwfs.forward(wf_est)
		Imodel = wf_zwfs.power
		mask2 = mask * (Imodel > 0)

		Eref = wf_zwfs.electric_field - wf_est.electric_field
		phase_ref = np.angle(Eref)
		wf_ref = Wavefront(Eref, wavelength)
		Iref = wf_ref.power
		
		Ncorr = np.sum(Imeas[i]) / np.sum(Imodel)

		# Reconstruct the individual images
		reduced_images[i] = (Imeas[i]/Ncorr - I0 - Iref)
		reduced_images[i, mask2] = reduced_images[i, mask2] / (2 * np.sqrt(I0 * Iref))[mask2]

		cosine_phases[i] = np.cos(phase_ref - delta_phases[i])
		sine_phases[i] = np.sin(phase_ref - delta_phases[i])

	reduced_images = np.clip(reduced_images, -1, 1)

	##
	R = Field(np.array(([cosine_phases, sine_phases])), aperture.grid)
	R = field_transpose(R)
	Rinv = field_inverse_tikhonov(R, 1e-15)

	tangents = field_dot(Rinv, reduced_images)
	phase = np.arctan2(tangents[1, :], tangents[0, :]) * mask
	phase[mask] -= np.median(phase[mask])
	
	#if apply_unwrapping:
	#	phase = unwrap_field(phase)
	
	if lobasis is not None:
		phase -= lobasis.linear_combination( lobasis.coefficients_for(phase) )
	
	return phase

def make_seal_pupil(index=0):
	def func(grid):
		pupil_im = np.load('pupil_im.npy') # Load SEAL pupil for more realistic simulation
		pupil_im = np.pad(pupil_im, ((0,0), (22,22), (22,22)))
		return Field(pupil_im[index].ravel(), grid)
	return func

def make_seal_interface():
	''' Code to make the hardware interface to the seal testbed.
	'''
	pupil_im = np.load('pupil_im.npy') # Load SEAL pupil for more realistic simulation
	pup = makePupil(pupil_im.shape[1]/2,pupil_im.shape[1]) # if you prefer to start with perfect full circular aperture
	position_pups = np.load('position_pups.npy') # to match SEAL dimensions

	# ----- CREATE ZWFS object ----
	shannon = 4 # to well oversample the dimple (2 could be enough tough)
	wavelength = 635 # in nm
	diameter = np.array([1,1]) # dimple diameter in L/D - Left pupil and right pupil
	depth = np.array([-np.pi/2,np.pi/2]) # dimple phase shift in radians - Left pupil and right pupil
	sealZWFS = zernikeWFS(pupil_im,position_pups,diameter,depth,shannon,wavelength,0) # Create object - 0 = simulation mode
	return sealZWFS


def process_seal_vzwfs(image, grid, index=0):
	if index == 0:
		return Field(image[:, 0:180].ravel(), grid)
	else:
		return Field(image[:, 180:].ravel(), grid)
	

if __name__ == "__main__":
	use_simulation = True

	# The hardware interface
	sealZWFS = make_seal_interface()

	# The model
	Dtel = 1.0
	Dgrid = 180 / 136 * Dtel
	grid = make_pupil_grid(180, Dgrid)
	
	aperture = make_seal_pupil()(grid)
	mask = aperture > 0

	#
	wf = Wavefront(aperture)
	wf.total_power = 1.0

	# ZWFS parameters
	wavelength = 1 # in nm
	nphot = 1
	diameter = np.array([1,1]) # dimple diameter in L/D - Left pupil and right pupil
	depth = np.array([-np.pi/2,np.pi/2]) # dimple phase shift in radians - Left pupil and right pupil
	local_zwfs = ZernikeWavefrontSensorOptics(grid, phase_step = -np.pi/2, phase_dot_diameter=1.0, num_pix=9, pupil_diameter=Dtel)
	Iref = local_zwfs(wf).power

	# Make the zernike basis
	num_modes = 20
	zmodes = make_zernike_basis(num_modes, Dtel, grid, 5)

	sa = SurfaceAberration(grid, 0.05, Dtel)
	sa.opd[mask] -= np.mean(sa.opd[mask])

	# 
	nmeas = 5
	delta_rms = 0.3 / np.sqrt(num_modes)
	zernike_coefficients = delta_rms * np.random.randn(nmeas, num_modes)	
	
	measurements = grid.zeros((nmeas,))
	delta_phases = grid.zeros((nmeas,))
	for i, dp in enumerate(delta_phases):
		delta_phases[i] = zmodes.linear_combination(zernike_coefficients[i])
		
		apod = PhaseApodizer(delta_phases[i])
		wfout = local_zwfs(sa(apod(wf)))
		measurements[i] = wfout.power
		
		plt.subplot(1, nmeas, i+1)
		imshow_field(wfout.power)
	plt.show()

	phase_est = grid.zeros()
	for k in range(5):
		phase_est = zwfs_multistep_reconstruct(measurements, delta_phases, local_zwfs, aperture, wavelength, nphot, mask, phase_est, apply_unwrapping=False, lobasis=None)

	plt.subplot(1,2,1)
	imshow_field(2 * np.pi * sa.opd)
	plt.colorbar()
	
	plt.subplot(1,2,2)
	imshow_field(phase_est)
	plt.colorbar()
	plt.show()