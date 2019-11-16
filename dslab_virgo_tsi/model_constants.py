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


class ExpConstants(ModelConstants):
    pass


class ExpLinConstants(ModelConstants):
    pass


class EnsembleConstants(ModelConstants):
    MODELS = None
    WEIGHTS = [0.1, 0.3, 0.3, 0.3]


class GaussianProcessConstants(ModelConstants):
    DOWNSAMPLING_FACTOR_A = 10000
    DOWNSAMPLING_FACTOR_B = 100

    MATERN_LENGTH_SCALE = 1
    MATERN_LENGTH_SCALE_BOUNDS = (1e-3, 1e3)
    MATERN_NU = 1.5

    WHITE_NOISE_LEVEL = 1e4
    WHITE_NOISE_LEVEL_BOUNDS = (1e-5, 1e5)

    N_RESTARTS_OPTIMIZER = 4
