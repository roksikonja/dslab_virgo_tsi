## Data Science With Total Solar Irradiance from SOHO Spacecraft

Project by **Luka Kolar, Rok Šikonja and Lenart Treven**.

This project is part of Data Science Lab at ETH Zurich. It was supervised by Prof. Dr. Andreas Krause and Prof. Dr. Ce Zhang at ETH Zurich. Dataset was provided by 
Dr. Wolfgang Finsterle from PMOD/WRC Institute. Dataset description can be found at 
https://www.pmodwrc.ch/en/research-development/space/virgo-soho/.

Further explanation of results and methods are summarized in the project paper **Iterative Correction of Sensor 
Degradation using Constrained Smooth Monotonic Functions and a Bayesian Multi-sensor Data Fusion Method applied to 
TSI data from PMO6-V Radiometers** available at ```DSLab_project_paper.pdf```. Additionally, project poster can be found
at ```DSLab_project_poster```.

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
    conda install numpy pandas matplotlib ipykernel jupyter scipy scikit-learn numba pytables flask flask-wtf tensorflow=2.0.0 cvxpy
    pip install gpflow==2.0.0rc1
    ```

### Run Data analysis

    python data_analysis.py
        --data_file = "virgo"                       // Select file {"virgo", "virgo_tsi"}
    
        --sampling                                  // Sampling analysis.
        --sampling_window=55                        // Minimum sampling window.

        --fft                                       // FFT analysis.
        
        --t_early_increase=100                      // Early increase timespan.
        --outlier_fraction = 0.0                    // Outlier fraction

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
        --virgo_days_end = -1                       // Use data up to this day
        
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
