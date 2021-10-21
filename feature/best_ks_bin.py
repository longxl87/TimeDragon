# -*- coding: utf-8 -*-
"""
Created on 2021/10/02
@author: Leo Long
@title: best_ks分箱法
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import time


def best_ks_bin(df, x, y, bins=5, init_bins=100, init_method='qcut', init_precision=3, print_process=True):
    """
    best_ks 分箱
    :param df: 数据集
    :param x: 待分箱特征名称
    :param y: 目标变量的名称
    :param bins: 最终的分箱数量
    :param init_bins:  初始化分箱的数量，若为空则钚进行初始化的分箱
    :param init_method: 初始化分箱的方法，仅支持 qcut 和cut两种方式
    :param init_precision: 初始化分箱时的precision
    :param print_process: 是否答应分箱的过程信息
    :return: 数值型变量为右边界的list，离散型变为 value:bin 这类键值对组成的dict
    """

    def _check_y(y):
        y_unique = y.unique()
        if len(y_unique) != 2:
            raise ValueError("y必须是二值型变量,且其元素必须是 0 1")
        for item in y_unique:
            if item not in [0, 1]:
                raise ValueError("y必须是二值型变量,且其元素必须是 0 1")

    if print_process: print("开始对变量{}进行best-ks分箱:".format(x))
    time0 = time.time()
    df = df[[x, y]].copy()

    if x == y:
        raise ValueError("x和y必须是不同的变量")

    _check_y(df[y])
    x_is_numeric = is_numeric_dtype(df[x])

    if x_is_numeric:
        if init_bins is not None:
            if init_method == 'qcut':
                df.loc[:, x] = pd.qcut(df[x], init_bins, duplicates="drop", precision=init_precision)
            elif init_method == 'cut':
                df.loc[:, x] = pd.cut(df[x], init_bins, duplicates="drop", precision=init_precision)
            else:
                raise ValueError("init_method参数仅有qcut和cut两个选项")
            df.loc[:, x] = df[x].map(lambda x: x.right).astype(float)
        dti = pd.crosstab(df[x], df[y], dropna=False).reset_index()
        dti.rename({0: "negative", 1: "positive"}, axis=1, inplace=True)
        dti["variable"] = dti[x]
        mapping = None
    else:
        dti = pd.crosstab(df[x], df[y]).reset_index()
        dti.rename({0: "negative", 1: "positive"}, axis=1, inplace=True)
        dti["positive_rate"] = dti["positive"] / (dti["positive"] + dti["negative"])
        dti = dti.sort_values("positive_rate").reset_index(drop=True).reset_index()
        dti.rename({"index": "variable"}, axis=1, inplace=True)
        mapping = dti[[x, "variable", "negative", "positive"]]
    dti = dti[["variable", "negative", "positive"]].copy()

    if print_process:
        time1 = time.time()
        print("==>已经完成变量初始化耗时{:.2f}s,开始处理0值".format(time1 - time0))

    while (len(dti) > bins) and (len(dti.query('(negative==0) or (positive==0)')) > 0):
        dti["count"] = dti["negative"] + dti["positive"]
        rm_bk = dti.query("(negative==0) or (positive==0)") \
            .query("count == count.min()")
        ind = rm_bk["variable"].index[0]
        if ind == dti["variable"].index.max():
            dti.loc[ind - 1, "variable"] = dti.loc[ind, "variable"]
        else:
            dti.loc[ind, "variable"] = dti.loc[ind + 1, "variable"]
        dti = dti.groupby("variable")[["negative", "positive"]].sum().reset_index()

    if print_process:
        time2 = time.time()
        print("==>已经完成0值处理耗时{:.2f}s,开始进行分箱迭代".format(time2 - time1))

    dti["count"] = dti["negative"] + dti["positive"]
    dti["tmp"] = 0
    uni_tmp = pd.unique(dti["tmp"])
    while (len(uni_tmp) < bins) and (len(dti) > bins):
        grouped_count = dti.groupby("tmp")[["count"]].sum()
        max_len_tmp_v = grouped_count.query("count == count.max() ").index[0]
        df_tmp = dti[dti["tmp"] == max_len_tmp_v].copy()
        min_variable = df_tmp["variable"].min()
        max_variable = df_tmp["variable"].max()
        if min_variable == max_variable:
            break
        df_tmp["cum_n"] = df_tmp["negative"].cumsum()
        df_tmp["cum_p"] = df_tmp["positive"].cumsum()
        n_t = df_tmp["negative"].sum()
        p_t = df_tmp["positive"].sum()
        df_tmp["ks"] = np.abs((df_tmp["cum_n"] / n_t) - (df_tmp["cum_p"] / p_t))
        besk_ks_variable = df_tmp.query(" ks == ks.max() ")["variable"].values[0]
        if besk_ks_variable == max_variable:
            dti.loc[dti["variable"] == max_variable, "tmp"] = (uni_tmp.max() + 1)  # (dti["tmp"].max() + 1)
        else:
            dti.loc[(dti["variable"] >= min_variable) & (dti["variable"] <= besk_ks_variable), "tmp"] = \
                (uni_tmp.max() + 1)  # (dti["tmp"].max() + 1)
        uni_tmp = pd.unique(dti["tmp"])

    dti["variable"] = dti["tmp"].map(lambda x: dti[dti["tmp"] == x]["variable"].max())
    dti = dti.groupby("variable")[["negative", "positive"]].sum().reset_index()

    if print_process:
        time3 = time.time()
        print("==>完成分箱迭代耗时{:.2f}s,开始标准化输出".format(time3 - time2))

    break_points = dti['variable'].copy()
    break_points[np.where(break_points == break_points.max())[0]] = np.inf
    break_points = np.concatenate(([-np.inf], break_points))

    if x_is_numeric:
        result = list(break_points)
    else:
        interval_index = pd.IntervalIndex.from_breaks(break_points, closed='right')
        mapping["bin"] = mapping["variable"].map(lambda x: np.where(interval_index.contains(x))[0][0])
        result = pd.Series(mapping["bin"].values, index=mapping[x].values).to_dict()

    if print_process:
        time4 = time.time()
        print("==>输出结果标准化完成耗时{:.2f}s,分箱完毕".format(time4 - time3))
    return result
