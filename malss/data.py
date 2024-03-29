# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from .rfpimp import oob_importances


class Data(object):
    def __init__(self, shuffle=True, standardize=True, random_state=None):
        self.shuffle = shuffle
        self.standardize = standardize
        self.random_state = random_state
        self.X = None
        self.y = None
        self.columns = None

    def fit_transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            self.X = pd.DataFrame(X, columns=list(map(str, range(X.shape[1]))))
            if y is not None:
                self.y = pd.Series(y)
        else:
            self.X = X.copy(deep=True)
            if y is not None:
                if isinstance(y, pd.Series):
                    self.y = y.copy(deep=True)
                else:
                    self.y = y.iloc[:, 0]  # Convert Dataframe to Series
        if not isinstance(self.X, pd.DataFrame):
            raise ValueError(f'{type(X)} is not supported')
        if y is not None and len(X) != len(y):
            raise ValueError(('Found input variables with inconsistent '
                             f'numbers of samples: [{len(X)}, {len(y)}]'))
        self.shape_before = self.X.shape

        self.X, self.col_was_null = self.__impute(self.X)

        self._label_encoder = None
        self._onehot_encoder = None
        self.X, self.del_columns = self.__encode(self.X)

        self._standardizer = None
        if self.standardize:
            self.X = self.__standardize(self.X)

        if self.columns is not None:
            self.X = self.X[self.columns]

        if self.shuffle:
            if self.y is not None:
                self.X, self.y = sk_shuffle(self.X, self.y,
                                            random_state=self.random_state)
            else:
                self.X = sk_shuffle(self.X, random_state=self.random_state)

    def transform(self, X):
        if isinstance(X, np.ndarray):
            Xtrans = pd.DataFrame(X, columns=list(map(str, range(X.shape[1]))))
        else:
            Xtrans = X.copy(deep=True)

        Xtrans, col_was_null = self.__impute(Xtrans)
        Xtrans, del_columns = self.__encode(Xtrans)
        if self.standardize:
            Xtrans = self.__standardize(Xtrans)
        
        if self.columns is not None:
            Xtrans = Xtrans[self.columns]

        return Xtrans

    def drop_col(self, mdl):
        print('Start feature selection.')
        X = self.X.copy(deep=True)

        while True:
            col, X = self.__drop_col_sub(X, self.y, mdl)
            if col is None:
                break
        
        if len(X.columns) == len(self.X.columns):
            print('No features dropped.')
        else:
            self.X = X
            self.columns = X.columns

    def __drop_col_sub(self, X, y, rf, k=10, thr=0.0):
        col = None
        rf.fit(X, y)
        np.random.seed(0)
        imp = oob_importances(rf, X, y, n_samples=len(X))
        for i in range(1, k):
            imp += oob_importances(rf, X, y, n_samples=len(X))
        imp /= k
        if imp['Importance'].min() < thr:
            col = imp['Importance'].idxmin()
            print(f'  Drop {col}')
            X = X.drop(col, axis=1)
        return col, X

    def __impute(self, X):
        fill = pd.Series([X[c].value_counts().index[0]
                          if X[c].dtype == np.dtype('O')
                          else X[c].median()
                          if X[c].dtype == np.dtype('int')
                          else X[c].mean()
                          for c in X],
                         index=X.columns)
        col_was_null = [c for c in X
                        if pd.isnull(X[c]).sum() > 0]
        return X.fillna(fill), col_was_null

    def __encode(self, X):
        Xenc = X.copy(deep=True)

        if self._label_encoder is None or self._onehot_encoder is None:
            self._label_encoder = [None] * len(Xenc.columns)
            self._onehot_encoder = [None] * len(Xenc.columns)

        del_columns = []
        for i in range(len(Xenc.columns)):
            if Xenc.dtypes[i] == np.dtype('O'):
                if self._label_encoder[i] is None:
                    self._label_encoder[i] = LabelEncoder().fit(Xenc.iloc[:,i])
                col_enc = self._label_encoder[i].transform(Xenc.iloc[:,i])
                if self._onehot_encoder[i] is None:
                    self._onehot_encoder[i] = OneHotEncoder(categories='auto').fit(
                        col_enc.reshape(-1, 1))
                col_onehot = np.array(self._onehot_encoder[i].transform(
                    col_enc.reshape(-1, 1)).todense())
                col_names = [str(Xenc.columns[i]) + '_' + c
                             for c in self._label_encoder[i].classes_]
                col_onehot = pd.DataFrame(col_onehot, columns=col_names,
                                          index=Xenc.index)
                Xenc = pd.concat([Xenc, col_onehot], axis=1)
                del_columns.append(Xenc.columns[i])
        for col in del_columns:
            del Xenc[col]

        return Xenc, del_columns

    def __standardize(self, X):
        X = X.astype('float64')
        if self._standardizer is None:
            self._standardizer = StandardScaler().fit(X)
        return pd.DataFrame(self._standardizer.transform(X),
                            index=X.index, columns=X.columns)
