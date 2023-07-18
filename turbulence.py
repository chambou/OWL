#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbulence generator class.
"""

from tools import *

class atm:
	"""
	This is a class to generate turublence.

	Attributes
	----------
	model: string
		model of the DM, can be either 'mems' or 'alpao'
	"""
	def __init__(self,telescope_diameter = 10,wavelength = 635*1e-09,fried_parameter = 0.15,velocity = 5,outer_scale = 40,timeSampling = 0.001,wfAffector=0,nModes=None):
		""" CONSTRUCTOR """
		#----  Define turbulence using HCIpy package -----
		# Fried parameter define for lambda = 500nm
		# velocity in m/s
		print('D   = {0:.1f}m'.format(telescope_diameter))
		print('Lambda   = {0:.1f}nm'.format(wavelength * 1e09))
		print('r0   = {0:.1f}cm'.format(fried_parameter * 100))
		print('L0   = {0:.1f}m'.format(outer_scale))
		print('v    = {0:.1f}m/s'.format(velocity))
		print('time sampling    = {0:.1f}ms'.format(timeSampling*1e03))
		self.D = telescope_diameter
		self.wavelength = wavelength
		self.fried_parameter = fried_parameter
		self.velocity = velocity
		self.outer_scale = outer_scale
		self.timeSampling = timeSampling
		self.resTurb = False
		if isinstance(wfAffector,int):
			self.wfAffector = 0 #for now only SLM
			self.nPx = wfAffector
		else:
			self.wfAffector = wfAffector #for now only SLM
			self.nPx = self.wfAffector.nPx
		pupil_grid = make_pupil_grid(self.nPx,telescope_diameter)
		Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
		self.layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
		self.wavelength = wavelength
		if nModes is not None:
			## Define Zernike basis set
			self.nModes = nModes
			mode_basis = make_zernike_basis(num_modes=self.nModes, D=self.D, grid=pupil_grid)
			self.residual_layer = ModalAdaptiveOpticsLayer(layer=self.layer, controlled_modes=mode_basis, lag=0) 

	def setTurbu(self,t=0):
		if self.resTurb == False:
			# Turbulence at time t on SLM
			self.layer.t = t
			phase_screen = self.layer.phase_for(self.wavelength) # in radians
			self.phase_screen = phase_screen.reshape((self.nPx,self.nPx))
			# -------- Case of SLM -------------------
			if self.wfAffector !=0:
				self.wfAffector.setPhase(self.phase_screen)
			time.sleep(0.01)
		elif self.resTurb == True and self.nModes is not None:
			# Residual Turbulence at time t on SLM
			self.residual_layer.evolve_until(t)
			residual_phase_screen = self.residual_layer.phase_for(self.wavelength) # in radians
			self.residual_phase_screen = residual_phase_screen.reshape((self.nPx,self.nPx))
			# -------- Case of SLM -------------------
			if self.wfAffector !=0:
				self.wfAffector.setPhase(self.residual_phase_screen)
			time.sleep(0.01)
	
	# def createTurbuCube(self,T=1,dT =0.01):
	# 	# Run turbulence over :
	# 	#- T seconds
	# 	#- dT step
	# 	time_turbu = np.linspace(0,T,int(T/dT))
	# 	k = 0
	# 	self.phase_TURBU = np.zeros((self.nPx,self.nPx,time_turbu.shape[0]))
	# 	for t in time_turbu:
	# 		self.layer.t = t
	# 		phase_screen_phase = self.layer.phase_for(self.wavelength) # in radians
	# 		phase_screen_phase = phase_screen_phase.reshape((self.nPx,self.nPx))
	# 		self.phase_TURBU[:,:,k] = phase_screen_phase
	# 		k = k + 1
	# 		print(k,'/',time_turbu.shape[0])
	
	
	# def runTurbu(self,speed = 0.1):
	# 	for k in range(0,time_turbu.shape[0]):
	# 		self.setPhase(self.phase_TURBU[:,:,k])
	# 		time.sleep(speed)
