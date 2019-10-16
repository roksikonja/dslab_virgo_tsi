class Constants(object):
    # Directories
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"

    # Data files
    VIRGO_FILE = "VIRGO_Level1.txt"

    # PMO6V
    A = "PMO6V-A"
    B = "PMO6V-B"
    T = "TIME"
    TEMP = "TEMPERATURE"

    # UNITS
    DAY_UNIT = "TIME [mission days]"
    YEAR_UNIT = "TIME [year]"
    FREQ_DAY_UNIT = "FREQ [1/day]"

    RATIO_UNIT = "RATIO [1]"
    TSI_UNIT = "TSI [W/m^2]"
    TEMP_UNIT = "TEMPERATURE [C]"
    SPECTRUM_UNIT = "POWER SPECTRUM [1]"

    # Visualization
    FIG_SIZE = (16, 8)
    MATPLOTLIB_STYLE = "seaborn"  # "default"
    MATPLOTLIB_STYLE_LINEWIDTH = 0.5
    OUT_FORMAT = "pdf"
    OUT_BBOX = "tight"
    OUT_DPI = 400
