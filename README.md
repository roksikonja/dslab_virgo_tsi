## Data Science With Total Solar Irradiance from SOHO Spacecraft

Data Science Lab Project - VIRGO Total Solar Irradiance

Project by **Luka Kolar, Rok Šikonja and Lenart Treven**.

### Dataset

Dataset was downloaded from publicly available server: ```ftp://ftp.pmodwrc.ch/pub/virgo/dsl2019/```.
Dataset description can be found: ```https://www.pmodwrc.ch/en/research-development/space/virgo-soho/```.

### Modeling

    modeling_1.py
    python modeling_1.py --model_type="exp" --window=20  // exp. model
    python modeling_1.py --model_type="exp_lin" --window=20  // exp. lin. model
    
    // TODO: Lenart
    python modeling_1.py --model_type="svr" --window=20 --param1 ... --paramK // SVM regressor
    python modeling_1.py --model_type="cubic_spline" --window=20 --param1 ... --paramK // Spline model
    python modeling_1.py --model_type="gpr" --window=20 --param1 ... --paramK // GP model


### Data analysis

    python data_analysis.py
        --visualizer                                // Visualize.
        --sampling                                  // Sampling analysis.
        --fft                                       // FFT analysis.
        --t_early_increase=100                      // Early increase timespan.
        --sampling_window=55                        // Minimum sampling window.


### Generator

Generating synthetic dataset.

    python generator.py --degradation_model="exp" --degradation_rate=1.0
    
    model ∈ {exp, exp_lin}
    rate ∈ (0.0, inf]
    
#### Meeting

* Mentor not responding
* GP local approximation
* Questions:
    * Scalable GP
    * Dual kernel question? White kernel different variance for both?
* Proof: Noisy model