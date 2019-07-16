# Find the Optimal Block Size based on Circular-Block-Bootstrap for AR(p) Financial Time Series 

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

Part 1: AR(p) generator
  1 get parameter: np.random
  2.get AR(p): arma_generate_sample(ar, ma=0, size)


Part 2: Model Right Test
2.1 Model fit for the bootstrapped time series: ARMA(endog=ts, order=(p, 0))
2.2 Residual Analysis: check whether residual is white noise or not
        2.2.1 mean vs 0: 1 sample t test
        2.2.2 stationary test: AD Fuller, KPSS
        2.2.3 no significant autocorrelation among residuals: acorr_ljungbox(time series)
  
  Part 3: Model Same Test
    3.1 parameter comparison
        3.1.1 distribution test for the generated parameters
        3.1.2 build up the confidence interval for parameters
  
  Part 4: Circular Block Bootstrap 
    4.1 CircularBlockBootstrap()
  
  Part 5: Multiprocessing and 3 Loops
  
