## Preliminary Problem Understanding Analysis

### VIRGO EXPERIMENT 

The VIRGO Experiment is part of SOHO Mission and consists of measurements of 3 physical quantities:

* total solar irradiance (TSI),
* spectral solar irradiance and
* spectral radiance. 

This project is concerned by the first quantity, total solar irradiance.  Total solar irradiance measurement is in other 
words equivalent to determination of the solar "constant", which is the solar radiance per unit of area (Watts/meter^2). 

There are several experiments conducting the same measurements: ACRIM_1 and HF (before VIRGO), ACRIM2/3, VIRGO, TIM and 
PREMOS (simultaneously as VIRGO).

The purpose of the experiment is to get reliable TSI data, which can be used to study solar atmosphere, noise, etc. 
Moreover, TSI data can be used to model the climate, (potentially helping us to better understand the climate changes and/or 
global warming).

### TSI MEASUREMENT INSTRUMENTATION 

In VIRGO experiment TSI is measured with two types of absolute radiometers, DIARAD and PMO6V. Both instruments have
two instances: the main sensor and the backup sensor.

* PMO6V (mode of operation after 23 February 1996): 
    * PMO6V-A (main): sampling continuosly at ```1 measurement/minute```,
    * PMO6V-B (backup): reduced sampling at ```1 measurement/week```.
* DIARAD (mode of operation after 18 January 1996)
    * DIARAD-L (main): sampling continuosly at ```1 measurement/ (3  * minutes)```,
    * DIARAD-R (backup): reduced sampling at ```1 measurement/ (60 * days)```.
    

### DATA

Our understanding of the data and its adjustment and correction flow is described below.

* Level-0: This is the raw data format stored in daily files.
    * Algorithm: ```????```
    * Level-0 source: ```????```

* Level-1: Level-0 data is converted to physical units, this process includes all a-priori
known information about the instrument: mapping raw values to physical units, temperature variation, etc. Additionally,
measurements are adjusted to 1 AU and to zero radial velocity. **This process standardizes TSI to the same distance at all
times (Solar irradiance is proportional to 1/r^2 with r being the distance between the Sun and the satellite) and eliminates
special relativity effects.** ```????```
    * The convertion algorithm was developed by the developer.
    * Level-1 source: ftp://ftp.pmodwrc.ch/pub/data/irradiance/virgo/1-minute_Data/VIRGO_1min_0083-7404.fits/.idl 
    ```(PMO6V ????)```

* Level-1.8: Exposure dependent changes are determined for each radiometer individually.
    * Algorithm: ```????```
    * Data source: ```????```

![From Level-1 to Level-2](Level1ToLevel2.jpg)

* Level-1 to Level-2:
    * Step 1:
        * DIARAD-L time series are corrected using backup DIARAD-R with splines
    * Step 2:
        * corrected DIARAD are compared with PMO6V-B by fitting an exponential correction for the non-exposure dependent
          change of DIARAD and a dose and temperature dependent correction for the early increase of PMO6V-B simultaneously
    * Step 3:
        * fully corrected DIARAD-L corresponds to the final level-2 data set
        * corrected PMO6V-B is now used to correct PMO6V-A to level-2 data set

    * Algorithm: ```????``` 
    * Level-2.0 DIARAD data: ```????```,
    * Level-2.0 PMO6V data: ```????```.

* Level-2.0 (VIRGO TSI): Inputs are the DIARAD level-2.0, corrected data for degradation, 
and PMO6V level-2.0, corrected data for degradation. The final VIRGO TSI is obtained by a weighted average of PMO6V
and DIARAD values as: 
        
        function instrument_variance
        input(S)                    // S[t] is the instrument's measured value at time step t (days)
        for each time step t:       // determine weight at every step
            W <- S[t-40:t+40]       // get measurement window of length 81 with the center at timestep t
            var <- variance(W)      // compute variance for window W
            result[t] <- var
        output result               // variance for every time step
        
        weight_PMO6V <- instrument_variance(PMO6V) - instrument_variance(DIARAD) // ????
        weight_DIARAD <- 1 - weight_PMO6V // ????
        
    * Final TSI is 131-day boxcar smoothed.
    * VIRGO TSI is available as daily and hourly data.
        * Daily data: ftp://ftp.pmodwrc.ch/pub/data/irradiance/virgo/TSI/virgo_tsi_d_v6_005_1805.dat
        * Hourly data: ftp://ftp.pmodwrc.ch/pub/data/irradiance/virgo/TSI/virgo_tsi_h_v6_005_1805.dat
    

### PROJECT GOAL

We understand that the project final goal is to automatize the computation of the final VIRGO TSI level-2.0 signal.

In the slides:

* Extract common solar signal. (VIRGO TSI)
* Determine time-dependent sensor degradation and noise spectrum.
    * Time dependent degradation is the process between data levels 1.8-2.0
    * Noise spectrum. FFT of the final VIRGO TSI? What is considered as noise? Sun emits certain electromagnetic wavelengths,
    analysis of the spectrum?
* Algorithm to automatically perform above tasks. 
    * Automatic pipeline with user interface, which includes parameter selection (e.g. moving average window size, include
    degradation correction or not, file names, version etc.) and input data selection (e.g. import new data and input data level).
    
* Building a web app (GUI). Automatic result output.


### QUESTIONS

Questions for the upcoming session. Also, we would need further clarification on the topics with associated ```????```.

#### MAIN QUESTIONS

- Which data will be available to us? Data at which level? Will the algorithms will be available to us, e.g. to obtain
level 1.0, 1.8 and 2.0 data? We will examine IDL routines and rewrite them in Python if necessary, this will
give us a reliable baseline. These routines would be implemented in the pipeline and improved.
    * Examination of data fusion methods. (Kalman Filters, NN?)
    * If lower level data is our input, then we would need your results and algorithms for all consecutive levels for 
    the purpose of reproduction of your results.

- How to access SOHO public/private archives? Licence for IDL?

#### AUXILIARY QUESTIONS

- What are SOHO vacations? Period, when the SOHO experiments, including VIRGO, did not produce data? Instead ACRIM2 data
was used?

- Could you explain again, what is the cause of the early increase and how are the measurements corrected for it?

- We would like to know a little bit more about the VIRGO satellite. What is its orbit and the orbit's center?
What is its revolution period? Are the VIRGO instruments at all times directed directly towards the Sun?

- Difference between exposure and non-exposure changes. What counts as degradation, exposure or non-exposure? We assume
that exposure, article Frohling 2014. 

- Level-0 to level-1 convertion requires instrument's calibration and temperature variation information and temperature
measurements during the VIRGO experiment. Moreover, it requires information on the VIRGO distance to Sun and its radial velocity.
 Where can we obtain this data? Will the convertion algorithm for both instruments be available to us? Would it be sensible
 to have it as a reliable baseline?

### Meeting 9. 10. 2019

* Results.
* Drop nan, 0.17 % 
    * Outlier correction
    * Smoothness guarantee
    * Monotonically d
* Early increase
* Ensemble of models and default



