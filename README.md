# pyAPCWT
Python implementation of the Affine Poincare Continuous Wavelet Transform and related staff. It is designed more for research and play-around than for performant computations. 

To use and install
`conda env create -n <YOUR_NEW_ENVIRONMENT_NAME> -f environment.yml`

## Roadmap:
- [ ] Implement a set of 2D transforms and combinations in the most precise manner: rotation, translation, and scale and the Lorentz rotations
  - [X] Transforms and unit test for it
  - [ ] Plot demo for transform: point & function  
- [ ] Implement FFT-related helpers, both for 2d and t-x domains
  - [ ] transforms, inverse, frequencies, unit tests 
  - [ ] Demo for transforms in the FFT  
- [ ] Implement basic mother wavelets(including Fourier domain): Mexican(Ricker), vanilla Morlet, exact Morlet, and Morlet with correction factor, Gaussian wave packet
  - [ ]  
- [ ] Implement usual CWT and APCWT with the inversion
- [ ] Build and implement a tight frame for the APCWT
