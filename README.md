# Data Science With Total Solar Irradiance from SOHO Spacecraft

Project by **Luka Kolar, Rok Šikonja and Lenart Treven**.

This project is part of Data Science Lab at ETH Zurich. It was supervised by Prof. Dr. Andreas Krause and Prof. Dr. Ce Zhang at ETH Zurich. Dataset was provided by 
Dr. Wolfgang Finsterle from PMOD/WRC Institute. Dataset description can be found at 
https://www.pmodwrc.ch/en/research-development/space/virgo-soho/.

Further explanation of results and methods are summarized in the project paper **Iterative Correction of Sensor 
Degradation using Constrained Smooth Monotonic Functions and a Bayesian Multi-sensor Data Fusion Method applied to 
TSI data from PMO6-V Radiometers** available at ```DSLab_project_paper.pdf```. Additionally, project poster can be found
at ```DSLab_project_poster```.

## Installation
* Install Miniconda (Python 3.x) from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
   * When asked `Do you wish the installer to initialize Miniconda3 by running conda init? [yes|no]` choose `yes`
* Run following commands within Miniconda terminal (**DO NOT** use `sudo` or `Run as administrator`):
    * *Optionally*:
    ```
    conda create --name virgo
    conda activate virgo
    ```
    * In any case
    ```
    conda install numpy pandas matplotlib scipy scikit-learn numba pytables flask flask-wtf flask-sqlalchemy tensorflow=2.0.0
    conda install -c conda-forge cvxpy
    pip install gpflow==2.0.0rc1
    ```
    
## Before Running or Modeling
* Download this repository via button *Clone or download* and extract it
* If *optional* step in installation was performed, make sure that you are using right environment by running:
```
conda activate virgo
```
* Copy files `VIRGO_Level1.txt` and `virgo_tsi_h_v6_005_1805.dat` to folder `dslab_virgo_tsi/data`.
* **Note**: When using graphical user interface, results are stored to `dslab_virgo_tsi/flask_backend/static/results`. In all other cases results are stored to `dslab_virgo_tsi/results`.
* **Note**: Modeling results can take a lot of space, because results of each analysis may be stored.

## Parameter Values
The first element of the set is the default, other model parameters are treated as constants and are specified in `model_constants.py`.
```
mode ∈ {"virgo", "gen"}

virgo_days_end ∈ {-1, 0, 1, ...}                   

outlier_fraction ∈ [0, 1]

exposure_method ∈ {"measurements", "cumsum"}
correction_method ∈ {"one", "both"}

model_type ∈ {"smooth_monotonic", "exp", "exp_lin", "spline", "isotonic", "ensemble"}
output_method ∈ {"svgp", "gp", "localgp"}
````

## Running and Modeling
### Graphical User Interface
```
python run.py
```
After running python script, in browser (preferably Google Chrome/Firefox) go to `localhost:5000`.
<br/>**Note**: Accepted are `.csv` files with 3 (or more) columns separated by tabs, where first column is time, second values of signal A, and third values of signal B.

### Data Analysis
```
python data_analysis.py
   --data_file = "virgo"                       // Select file {"virgo", "virgo_tsi"}

   --sampling                                  // Sampling analysis.
   --sampling_window=55                        // Minimum sampling window.

   --fft                                       // FFT analysis.

   --t_early_increase=100                      // Early increase timespan.
   --outlier_fraction = 0.0                    // Outlier fraction
```


### Modeling on VIRGO Dataset
```
python run_modeling.py
   --virgo_days_end = -1                       // Use data up to this day

   --outlier_fraction = 0.0                    // Outlier fraction
   --exposure_method = "measurements"          // Method for computing exposure

   --model_type = "smooth_monotonic"           // Degradation correction model
   --correction_method = "one"                 // Iterative correction method

   --output_method = "svgp"                    // Output model

   --save_plots = False or save_signals        // Save plots
   --save_signals = False                      // Save signals to pickle
```

### Modeling on Synthetic Dataset
```
python run_generator.py
   --random_seed = 0                           // Random seed for synthetic data generation

   --outlier_fraction = 0.0                    // Outlier fraction
   --exposure_method = "measurements"          // Method for computing exposure

   --model_type="smooth_monotonic"             // Degradation correction model
   --correction_method = "one"                 // Iterative correction method

   --output_method = "svgp"                    // Output model

   --save_plots = False or save_signals        // Save plots
   --save_signals = False                      // Save signals to pickle
```

### Model Comparison
```
python run_comparison.py
   --mode="virgo"                              // Select dataset
   --random_seed = 0                           // Random seed for synthetic data generation

   --outlier_fraction = 0.0                    // Outlier fraction
   --exposure_method = "measurements"          // Method for computing exposure

   --correction_method = "one"                 // Iterative correction method

   --save_plots = False or save_signals        // Save plots
   --save_signals = False                      // Save signals to pickle
```
