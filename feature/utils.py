# -*- coding: utf-8 -*-
"""
Created on 2021/10/02
@author: Leo Long
@title: 常用工具包
"""
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def bin_info(df, x, y, cond, fill_NA=None, independ=[], lamba=0.001):
    df = df[[x, y]].copy()
    x_is_numeric = is_numeric_dtype(df)

    if fill_NA is not None:
        df[x] = df[x].fillna(fill_NA)

    dti0 = None
    bin_name = x + "_bin"
    if x_is_numeric and isinstance(cond, list):
        if (independ is not None) and len(independ) > 0:
            df0 = df[df[x].isin(independ)]
            dti0 = pd.crosstab(df0[x], df0[y]).reset_index(). \
                rename({0: "negative", 1: "positive", x: bin_name}, axis=1)
            dti0["independ"] = True
            df1 = df[~df[x].isin(independ)]
        else:
            df1 = df
        df1[bin_name] = pd.cut(df1[x], cond, right=True)
        dti1 = pd.crosstab(df1[bin_name], df1[y]).reset_index(). \
            rename({0: "negative", 1: "positive"}, axis=1)
        dti1["independ"] = False
        if dti0 is not None:
            dti = pd.concat([dti0, df1], axis=1)
        else:
            dti = dti1

    elif (not x_is_numeric) and isinstance(cond, dict):
        if (independ is not None) and len(independ) > 0:
            max_values = max(cond.values())
            for i in range(len(independ)):
                cond[independ[i]] = max_values + i + 1
        mapping = pd.DataFrame.from_dict(cond, orient="index").reset_index()
        mapping.columns = [x, bin_name]
        df = df.merge(mapping, on=x, how="left")
        dti = pd.crosstab(df[bin_name], df[y]).reset_index(). \
            rename({0: "negative", 1: "positive"}, axis=1)
    else:
        raise ValueError("数字型变量cond必须为list，类别型变量cond必须为dict")

    n_t = dti["negative"].sum()
    p_t = dti["positive"].sum()
    dti["positive_rate"] = dti["positive"] / (dti["positive"] + dti["negative"])
    dti["woe"] = np.log(((dti["positive"] / p_t) + lamba) / ((dti["negative"] / n_t) + lamba))
    dti["iv"] = ((dti["positive"] / p_t) - (dti["negative"] / n_t)) * dti["woe"]

    info = {"iv": dti["iv"].sum(), "dti": dti}

    if not x_is_numeric:
        mapping = mapping.merge(dti, on=bin_name, how='left')
        mapping = mapping[[x, bin_name, "woe"]]
        info["mapping"] = mapping

    return info
