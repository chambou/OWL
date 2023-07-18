# OWL

This a reduced version of libSEAL with only comments needed to run simulation.
These 2 files are array that contains informations to match SEAL testbed configuration in our simulations:
 - pupil_im.npy
 - position_pups.npy

ZWFS_simu.py is the main script to run.


## Notes from 2023/07/17

### Make a standard dataset.

- [ ] 100 phase screens with reduced dimensionality
- [ ] Assert NCPA = 0
- [ ] (Expand dataset with fitting errors)

### Reconstruction techniques

- [/] Linear, model-based
  - [x] Synthetic nteraction matrix
  - [ ] Analytical
  - [ ] Iterative
- [/] Linear, empirical
- [x] Gerchberg-Saxton
- [x] Non-linear inverse model
  - [x] Single-step
  - [x] Iterative
- [ ] Both pupils "modulated ZWFS"
- [ ] (Neural net)
- [ ] Gradient descent
- [ ] Phase-shifting interferometry