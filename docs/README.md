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
* Level-1 to Level-2:
    * Step 1:
        * DIARAD-L time series are corrected using backup DIARAD-R with splines
    * Step 2:
        * corrected DIARAD are compared with PMO6V-B by fitting an exponential correction for the non-exposure dependent
          change of DIARAD and a dose and temperature dependent correction for the early increase of PMO6V-B simultaneously
    * Step 3:
        * fully corrected DIARAD-L corresponds to the final level-2 data set
        * corrected PMO6V-B is now used to correct PMO6V-A to level-2 data set

* Level-2.0 (VIRGO TSI): Inputs are the DIARAD level-2.0, corrected data for degradation, 
and PMO6V level-2.0, corrected data for degradation. The final VIRGO TSI is obtained by a weighted average of PMO6V
and DIARAD values as: 
        
        function instrument_variance
        input(S)                                    // S[t] is the instrument's measured value at time step t (days)
        for each time step t:                       // determine weight at every step
            W <- S[t-40:t+40]                       // get measurement window of length 81 with the center at timestep t
            var <- variance(W)                      // compute variance for window W
            result[t] <- var
        output result                               // variance for every time step
        
        weight_PMO6V <- instrument_variance(PMO6V) - instrument_variance(DIARAD) // ????
        weight_DIARAD <- 1 - weight_PMO6V           // ????
        
    * Final TSI is 131-day boxcar smoothed.
    * VIRGO TSI is available as daily and hourly data.
        * Daily data: ftp://ftp.pmodwrc.ch/pub/data/irradiance/virgo/TSI/virgo_tsi_d_v6_005_1805.dat
        * Hourly data: ftp://ftp.pmodwrc.ch/pub/data/irradiance/virgo/TSI/virgo_tsi_h_v6_005_1805.dat
    

 
![From Level-1 to Level-2](Level1ToLevel2.jpg)

Current understanding of problem:

Main goal is to infer VIRGO TSI, which is at level 2. Level 2 means that exposure and non-exposure dependent changes have been accounted for. Level 1.8 on the other hand means that only exposure dependent changes have been taken into account. There are 2 signals, DIARAD and PMO6V, from which VIRGO TSI is inferred. Both of them have 2 channels, one main and one backup, where backup channel samples at much lower frequency in order to account for degradation. Degradation has lower influence on backup sensor, since degradation is proportional to exposure of sensor to sun. In latest version of algorithm, level 1.8 is never constructed. Instead, Level 2 is constructed directly from level 1 data.We are not sure what exactly is goal of our project:
- Infer corrected DIARAD signal from 2 DIARAD channels
- Same as previous bullet point for PMVO6
- Infer VIRGO TSI from DIARAD and PMVO6 (basically try to improve method 6.4 from level 1 data)Inferrence of VIRGO TSI:
DIARAD-L and PMO6V-A sample way more frequently than DIARAD-R and PMO6V-BDIARAD-R (Level 1) -> DIARAD-L (Level 1) -> Corrected DIARAD-L (Level 2)
                                            
                                            +                                            Corrected PMO6V-B -> Corrected PMO6V-A (Level 2)- Not sure what switch-offs are exactly
- Not sure what SOHO vacations are
- Do not understand rapid increase behaviourLENART:- what are SOHO vacations?

- Could you explain again, what is the cause of the early increase and how are the measurements corrected for it?

- We would like to know a little bit more about the VIRGO satellite. What is its orbit and the orbit's center?
What is its revolution period? Are the VIRGO instruments at all times

- Difference between exposure and non-exposure changes.

- Level-0 to level-1 convertion requires instrument's calibration and temperature variation information and temperature
measurements during the VIRGO experiment. Moreover, it requires information on the VIRGO distance to Sun and its radial velocity.
 Where can we obtain this data? Will the convertion algorithm for both instruments be available to us? Would it be sensible
 to have it as a reliable baseline?


