# coding: utf-8


class Params:
    def __init__(self, lang):
        self.lang = lang
        self.task = None
        self.fpath = None
        """
        First 5 lines of original data
        """
        self.data5 = None
        self.columns = None
        self.col_types_def = None
        self.col_types = None
        self.objective = None
        self.col_types_changed = True
        """
        Note that X was transformed
        (i.e. some features may be transformed to dummy variables)
        while columns(_def) and col_types(_def) represent original ones.
        """
        self.X = None
        self.y = None
        self.algorithms = None
        self.mdl = None
        self.results = None
        self.error = None
        # Parameters for feature selection
        self.X_fs = None
        self.algorithms_fs = None
        self.mdl_fs = None
        self.results_fs = None
        self.not_deleted = False

        self.out_dir = None
        self.path_pred = None
