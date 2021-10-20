# -*- coding: utf-8 -*-
"""
Created on 2021/10/02
@author: Leo Long
@title: 常用工具包
"""
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def bin_info(df, x, y, cond, fillNa=None, independ=[]):
    df = df[[x, y]].copy()
    x_is_numeric = is_numeric_dtype(df)

    if fillNa is not None:
        df[x] = df[x].fillna(fillNa)

    # todo 待修复
    if (independ is not None) and len(independ) > 0:
        df0 = df[df[x].isin(independ)]
        dti0 = pd.crosstab(df0[x], df0[y]). \
            rename({0: "negative", 1: "positive"}, axis=1). \
            reset_index()
        df1 = df[~df[x].isin(independ)]

        dti1 = pd.crosstab(df1[x], df1[y]). \
            rename({0: "negative", 1: "positive"}, axis=1). \
            reset_index()
        dti = pd.concat([dti0, dti1], axis=0)
    else:
        dti = pd.crosstab(df[x], df[y]). \
            rename({0: "negative", 1: "positive"}, axis=1). \
            reset_index()
    return dti
