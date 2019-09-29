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
* Level-1: Level-0 data is converted to physical units.  





Current understanding of problem:

Main goal is to infer VIRGO TSI, which is at level 2. Level 2 means that exposure and non-exposure dependent changes have been accounted for. Level 1.8 on the other hand means that only exposure dependent changes have been taken into account. There are 2 signals, DIARAD and PMO6V, from which VIRGO TSI is inferred. Both of them have 2 channels, one main and one backup, where backup channel samples at much lower frequency in order to account for degradation. Degradation has lower influence on backup sensor, since degradation is proportional to exposure of sensor to sun. In latest version of algorithm, level 1.8 is never constructed. Instead, Level 2 is constructed directly from level 1 data.We are not sure what exactly is goal of our project:
- Infer corrected DIARAD signal from 2 DIARAD channels
- Same as previous bullet point for PMVO6
- Infer VIRGO TSI from DIARAD and PMVO6 (basically try to improve method 6.4 from level 1 data)Inferrence of VIRGO TSI:
DIARAD-L and PMO6V-A sample way more frequently than DIARAD-R and PMO6V-BDIARAD-R (Level 1) -> DIARAD-L (Level 1) -> Corrected DIARAD-L (Level 2)
                                            
                                            +                                            Corrected PMO6V-B -> Corrected PMO6V-A (Level 2)- Not sure what switch-offs are exactly
- Not sure what SOHO vacations are
- Do not understand rapid increase behaviourLENART:- what are SOHO vacations?