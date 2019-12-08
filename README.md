## Data Science With Total Solar Irradiance from SOHO Spacecraft

Data Science Lab Project - VIRGO Total Solar Irradiance

Project by **Luka Kolar, Rok Šikonja and Lenart Treven**.

### Installation
* Install Miniconda from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
* Run following commands within Miniconda terminal (**DO NOT** use `sudo` or `Run as administrator`):
    * *Optionally*:
    ```
    conda create --name virgo
    conda activate virgo
    ```
    * In any case
    ```
    conda install numpy pandas matplotlib ipykernel jupyter scipy scikit-learn numba pytables flask flask-wtf tensorflow=2.0.0 pykalman cvxpy
    pip install gpflow==2.0.0rc1
    ```

### Dataset

Dataset was downloaded from publicly available server: ftp://ftp.pmodwrc.ch/pub/virgo/dsl2019/.
Dataset description can be found: https://www.pmodwrc.ch/en/research-development/space/virgo-soho/.

### Run Data analysis

    python data_analysis.py
        --visualize                                 // Visualize.
    
        --sampling                                  // Sampling analysis.
        --sampling_window=55                        // Minimum sampling window.

        --fft                                       // FFT analysis.
        
        --t_early_increase=100                      // Early increase timespan.

## Modeling

Parameter values. The first element of the set is the default.

    mode ∈ {"virgo", "gen"}

    virgo_days_end ∈ {-1, 0, 1, ...}                   

    outlier_fraction ∈ [0, 1]
    
    exposure_method ∈ {"measurements", "cumsum"}
    correction_method ∈ {"one", "both"}
    
    model_type ∈ {"smooth_monotonic", "exp", "exp_lin", "spline", "isotonic", "ensemble"}
    output_method ∈ {"svgp", "gp", "localgp"}

Other model parameters are treated as constants and are specified in ```model_constants.py```.

### Run Modeling on VIRGO dataset

    python run_modeling.py
        --virgo_days = -1                           // Use data up to this day
        
        --outlier_fraction = 0.0                    // Outlier fraction
        --exposure_method = "measurements"          // Method for computing exposure
        
        --model_type = "smooth_monotonic"           // Degradation correction model
        --correction_method = "one"                 // Iterative correction method
        
        --output_method = "svgp                     // Output model
        
        --save_plots = False or save_signals        // Save plots
        --save_signals = False                      // Save signals to pickle

### Run Modeling on Synthetic dataset

    python run_generator.py
        --random_seed = 0                           // Random seed for synthetic data generation
    
        --outlier_fraction = 0.0                    // Outlier fraction
        --exposure_method = "measurements"          // Method for computing exposure
        
        --model_type="smooth_monotonic"             // Degradation correction model
        --correction_method = "one"                 // Iterative correction method
        
        --output_method = "svgp                     // Output model
        
        --save_plots = False or save_signals        // Save plots
        --save_signals = False                      // Save signals to pickle

### Run Model Comparison

    python run_comparison.py
        --mode="virgo"                              // Select dataset
        --random_seed = 0                           // Random seed for synthetic data generation

        --outlier_fraction = 0.0                    // Outlier fraction
        --exposure_method = "measurements"          // Method for computing exposure

        --correction_method = "one"                 // Iterative correction method

        --save_plots = False or save_signals        // Save plots
        --save_signals = False                      // Save signals to pickle
