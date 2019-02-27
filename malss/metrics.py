from sklearn.metrics import precision_recall_fscore_support


def f1_weighted(y_true, y_pred):
    '''
    This method is used to supress UndefinedMetricWarning
    in f1_score of scikit-learn.
    "filterwarnings" doesn't work in CV with multiprocess.
    '''
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=None,
                                                 pos_label=1,
                                                 average='weighted',
                                                 warn_for=(),
                                                 sample_weight=None)
    return f