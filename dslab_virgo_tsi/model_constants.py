import cvxpy as cp


class SmoothMonotoneRegressionConstants(object):

    INCREASING = False
    NUMBER_OF_POINTS = 999
    Y_MAX = 1
    Y_MIN = 0
    OUT_OF_BOUNDS = "clip"
    SOLVER = cp.ECOS_BB
    LAM = 1

    @staticmethod
    def return_config():
        return dict(SmoothMonotoneRegressionConstants.__dict__)


class IsotonicConstants(object):

    SMOOTHING = False
    INCREASING = False
    NUMBER_OF_POINTS = 201
    Y_MAX = 1
    Y_MIN = 0
    OUT_OF_BOUNDS = "clip"
    K = 3
    STEPS = 30

    @staticmethod
    def return_config():
        return dict(IsotonicConstants.__dict__)


class SplineConstants(object):

    K = 3
    STEPS = 30
    THINNING = 100

    @staticmethod
    def return_config():
        return dict(SplineConstants.__dict__)


class ExpConstants(object):

    @staticmethod
    def return_config():
        return dict(ExpConstants.__dict__)


class ExpLinConstants(object):

    @staticmethod
    def return_config():
        return dict(ExpLinConstants.__dict__)


class EnsembleConstants(object):

    MODELS = None
    WEIGHTS = [0.1, 0.3, 0.3, 0.3]

    @staticmethod
    def return_config():
        return dict(EnsembleConstants.__dict__)
