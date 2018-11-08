# coding: utf-8


class Params:
    def __init__(self, lang):
        self.lang = lang
        self.task = None
        self.fpath = None
        self.data = None
        self.columns = None
        self.col_types_def = None
        self.col_types = None
        self.objective = None
        self.col_types_changed = True
        self.X = None
        self.y = None
        self.algorithms = None
        self.mdl = None
        self.results = None
        self.error = None
        # Parameters for feature selection
        self.mdl_fs = None
        self.X_fs = None
        self.not_deleted = False
