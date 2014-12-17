# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler


class Data(object):
    def __init__(self, X, y, shuffle=True, standardize=True,
                 random_state=None):
        if isinstance(X, np.ndarray):
            self.X = pd.DataFrame(X)
            self.y = pd.Series(y)
        else:
            self.X = X.copy(deep=True)
            self.y = y.copy(deep=True)
        if not isinstance(self.X, pd.DataFrame):
            raise ValueError('%s is not supported' % type(X))
        self.shape_before = self.X.shape

        self.__imputer()

        self.__encoder()

        if shuffle:
            self.X, self.y = sk_shuffle(self.X, self.y,
                                        random_state=random_state)
        if standardize:
            self.X = StandardScaler().fit_transform(self.X)

    def __imputer(self):
        fill = pd.Series([self.X[c].value_counts().index[0]
                          if self.X[c].dtype == np.dtype('O')
                          else self.X[c].median()
                          if self.X[c].dtype == np.dtype('int')
                          else self.X[c].mean()
                          for c in self.X],
                         index=self.X.columns)
        self.col_was_null = [c for c in self.X
                             if pd.isnull(self.X[c]).sum() > 0]
        self.X = self.X.fillna(fill)

    def __encoder(self):
        self.del_columns = []
        for i in xrange(len(self.X.columns)):
            if self.X.dtypes[i] == np.dtype('O'):
                enc = LabelEncoder()
                col_enc = enc.fit_transform(self.X.icol(i))
                col_onehot = np.array(
                    OneHotEncoder().fit_transform(
                        col_enc.reshape(-1, 1)).todense())
                col_names = [str(self.X.columns[i]) + '_' + c
                             for c in enc.classes_]
                col_onehot = pd.DataFrame(col_onehot, columns=col_names,
                                          index=self.X.index)
                self.X = pd.concat([self.X, col_onehot], axis=1)
                self.del_columns.append(self.X.columns[i])
        for col in self.del_columns:
            del self.X[col]
