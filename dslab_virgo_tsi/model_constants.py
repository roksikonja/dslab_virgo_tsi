import cvxpy as cp


class ModelConstants(object):

    @staticmethod
    def return_config(model_class, consts_type="MODEL"):
        d = dict(model_class.__dict__)
        out_d = dict()

        for key in d:
            if not str(key)[0:2] == "__" and not str(key) == "return_config":
                out_d[f"{consts_type}_{key}"] = d[key]

        return out_d


class GeneratorConstants(ModelConstants):
    SIGNAL_LENGTH = 100000
    DEGRADATION_MODEL = "exp"
    DEGRADATION_RATE = 1.0
    DOWNSAMPLING_A = 0.1
    DOWNSAMPLING_B = 0.9
    STD_NOISE_A = 0.025
    STD_NOISE_B = 0.015


class SmoothMonotoneRegressionConstants(ModelConstants):
    INCREASING = False
    NUMBER_OF_POINTS = 999
    Y_MAX = 1
    Y_MIN = 0
    OUT_OF_BOUNDS = "clip"
    SOLVER = cp.ECOS_BB
    LAM = 1


class IsotonicConstants(ModelConstants):
    SMOOTHING = False
    INCREASING = False
    NUMBER_OF_POINTS = 201
    Y_MAX = 1
    Y_MIN = 0
    OUT_OF_BOUNDS = "clip"
    K = 3
    STEPS = 30


class SplineConstants(ModelConstants):
    K = 3
    STEPS = 30
    THINNING = 100
    S_MAX = 100


class ExpConstants(ModelConstants):
    MAX_FEVAL = 10000


class ExpLinConstants(ModelConstants):
    MAX_FEVAL = 10000


class EnsembleConstants(ModelConstants):
    MODELS = None
    WEIGHTS = [0.1, 0.3, 0.3, 0.3]


class GaussianProcessConstants(ModelConstants):
    # Normalize
    NORMALIZATION = True

    # Clip values
    CLIPPING = True

    # Dual white kernel
    DUAL_KERNEL = True
    LABEL_A = 0
    LABEL_B = 1
    LABEL_OUT = -1

    # scikit-learn parameters
    DOWNSAMPLING_FACTOR_A = 30000
    DOWNSAMPLING_FACTOR_B = 300

    MATERN_LENGTH_SCALE = 1
    MATERN_LENGTH_SCALE_BOUNDS = (1e-3, 1e4)
    # MATERN_NU = 1.5
    MATERN_NU = 0.5

    WHITE_NOISE_LEVEL = 1
    WHITE_NOISE_LEVEL_BOUNDS = (1e-5, 1e5)

    N_RESTARTS_OPTIMIZER = 5

    # gpflow parameters
    TRAIN_INDUCING_VARIABLES = False
    INITIAL_FIT = True
    NUM_INDUCING_POINTS = 1000
    MINIBATCH_SIZE = 200
    MAX_ITERATIONS = 10000
    NUM_SAMPLES = 20000
    LEARNING_RATE = 0.005
    PRIOR_POSTERIOR_SAMPLES = 10
    PRIOR_POSTERIOR_LENGTH = 2000

    # local gp
    GPR_MODEL = "sklearn"
    WINDOW = 100
    GEN_WINDOW = 0.1
    POINTS_IN_WINDOW = 100
    WINDOW_FRACTION = 5


class OutputTimeConstants(ModelConstants):
    # Generator output
    GEN_NUM_HOURS = 10001
    GEN_NUM_HOURS_PER_DAY = 10

    # Virgo output
    VIRGO_NUM_HOURS_PER_UNIT = 24
    VIRGO_NUM_HOURS_PER_DAY = 24
