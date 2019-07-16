# Find the Optimal Block Size based on Circular-Block-Bootstrap for AR(p) Financial Time Series 
using Circular Block Bootstrap method to resample time series for several block sizes, from there to find a optimal block size that make the resampled time series best inherited characteristics from original one

Objective:
  to find the optimal block size for every given p of AR(p) time series 

The main logic of the code is:
  1. AR(p) generator: generating the AR time series
  2. Model Fit Test: model fitting for bootstrapped time series; Residual Analysis 
  3. Model Same Test: built up confidence interval for different parameters; check whether the parameters are significantly different from the statistics point
  4. Circular Block Bootstrap 
  5. Multiprocessing with 3 loops: p, parameters, block sizes

Judgements:
  1. Model Right Test Rate
  2. Model Same Test Rate


  
