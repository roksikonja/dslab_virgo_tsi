class Constants(object):
    # Directories
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"

    # Data files
    VIRGO_FILE = "VIRGO_Level1.txt"
    PREMOS_FILE = "PREMOS.dat"
    VIRGO_TSI_FILE = "virgo_tsi_h_v6_005_1805.dat"

    # PMO6V
    A = "PMO6V-A"
    B = "PMO6V-B"
    T = "TIME"
    TEMP = "TEMPERATURE"
    EA = "EXPOSURE-A"
    EB = "EXPOSURE-B"

    # VIRGO TSI
    VIRGO_TSI_NEW = "VIRGO-NEW"
    VIRGO_TSI_OLD = "VIRGO-OLD"
    DIARAD_OLD = "DIARAD-OLD"
    PMO6V_OLD = "PMO6V-OLD"
    VIRGO_TSI_NAN_VALUE = -99.0

    # RANDOMNESS
    RANDOM_SEED = 0

    # MODELS
    SPLINE = "SPLINE"
    GP = "GP"
    EXP = "EXP"
    EXP_LIN = "EXP_LIN"
    ISOTONIC = "ISOTONIC"
    SMOOTH_MONOTONIC = "SMOOTH_MONOTONIC"
    ENSEMBLE = "ENSEMBLE"

    # UNITS
    DAY_UNIT = "TIME [mission days]"
    YEAR_UNIT = "TIME [year]"
    FREQ_DAY_UNIT = "FREQ [1/day]"

    RATIO_UNIT = "RATIO [1]"
    ITERATION_UNIT = "ITERATIONS"
    LOG_LIKELIHOOD_UNIT = "LOG_LIKELIHOOD [1]"
    TSI_UNIT = "TSI [W/m^2]"
    TEMP_UNIT = "TEMPERATURE [C]"
    SPECTRUM_UNIT = "POWER SPECTRUM [1]"

    # VISUALIZATION
    FIG_SIZE = (16, 8)
    TITLE_FONT_SIZE = 22
    AXES_FONT_SIZE = 15
    LEGEND_FONT_SIZE = 12
    XTICK_SIZE = 12
    YTICK_SIZE = 12
    MATPLOTLIB_STYLE = "seaborn"  # "default"
    MATPLOTLIB_STYLE_LINEWIDTH = 0.5
    OUT_FORMAT = "pdf"
    OUT_BBOX = "tight"
    OUT_DPI = 400
    MATPLOTLIB_STYLE_MARKER = "x"
    MATPLOTLIB_STYLE_MARKERSIZE = 3
