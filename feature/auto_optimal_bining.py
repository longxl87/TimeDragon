# -*- coding: utf-8 -*-
"""
自动化最优分箱工具
Create Date 2020.11.27
"""

import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target


def _check_target_binary(y):
    """
    check if the target variable is binary
    ------------------------------
    Param
    y:exog variable,pandas Series contains binary variable
    ------------------------------
    Return
    if y is not binary, raise a error
    """
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('目标变量必须是二元的！')


def _isNullZero(x):
    """
    check x is null or equal zero
    -----------------------------
    Params
    x: data
    -----------------------------
    Return
    bool obj
    """
    cond1 = np.isnan(x)
    cond2 = x==0
    return cond1 or cond2


def _EqualWidthBinMap(x, Acc, adjust, method):
    """
    Data bining function,
    middle procession functions for binContVar
    method: equal width or equal frequency
    Mind: Generate bining width and interval by Acc
    --------------------------------------------
    Params
    x: pandas Series, data need to bining
    Acc: float less than 1, partition ratio for equal width bining
    adjust: float or np.inf, bining adjust for limitation
    method: equal width or equal frequency
    --------------------------------------------
    Return
    bin_map: pandas dataframe, Equal width bin map
    """
    x = x.astype(float)  # 将数据做统一转换成浮点数
    x = x.dropna()  # 删除为nan的值
    # generate range by Acc
    Mbins = int(1./Acc)  # 100个数组
    # get upper_limit and loewe_limit
    ind = range(1, Mbins + 1)
    Upper = pd.Series(index=ind, name='upper')
    Lower = pd.Series(index=ind, name='lower')
    """
    根据method判断分箱方式：1）等宽 -- 0 ；2）等频 -- 1；
    当数据分布不均或者倾斜严重时，优先等频分箱；
    当数据分布均匀或者样本量非常大时，优先等宽分箱；
    """
    varMax = x.max()  # 上限值
    varMin = x.min()  # 下限值
    minMaxSize = (varMax - varMin) / Mbins  # 获取等宽的步长step

    if method == 0: # 等宽分箱
        # 对应数组的上限值
        for i in ind:
            Upper[i] = varMin + i * minMaxSize
            Lower[i] = varMin + (i - 1) * minMaxSize
    else:  # 等频分箱
        for i in ind: # 循环分组
            Upper[i] = np.percentile(x, i)  # 当前分位点的上限
            Lower[i] = np.percentile(x, i)  # 当前分位点的下限

    # adjust the min_bin's lower and max_bin's upper
    # 根据adjust调整上下限值
    Upper[Mbins] = Upper[Mbins] + adjust  # 最大上限加上调整参数值
    Lower[1] = Lower[1] - adjust  # 最小下限减去调整参数
    bin_map = pd.concat([Lower, Upper], axis=1)  # 将上下限序列合并
    bin_map.index.name = 'bin'  # 将索引命名为bin
    return bin_map  # 输出上下限表


# 将X变量转换成对应的分箱
def _applyBinMap(x, bin_map):
    """
    Generate result of bining by bin_map
    ------------------------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, map table
    ------------------------------------------------
    Return
    bin_res: pandas Series, result of bining
    """
    # 初始化数组
    bin_res = np.array([0] * x.shape[-1], dtype=int)

    for i in bin_map.index:
        upper = bin_map['upper'][i]
        lower = bin_map['lower'][i]
        # 映射对应的分段
        x1 = x[np.where((x >= lower) & (x <= upper))[0]]
        mask = np.in1d(x, x1)
        bin_res[mask] = i

    bin_res = pd.Series(bin_res, index=x.index)
    bin_res.name = x.name + "_BIN"

    return bin_res


def _combineBins(temp_cont, target):
    """
    merge all bins that either 0 or 1 or total =0
    middle procession functions for binContVar
    ---------------------------------
    Params
    temp_cont: pandas dataframe, middle results of binContVar
    target: target label
    --------------------------------
    Return
    temp_cont: pandas dataframe
    """
    for i in temp_cont.index:
        rowdata = temp_cont.loc[i, :]

        if i == temp_cont.index.max():
            ix = temp_cont[temp_cont.index < i].index.max()

        else:
            ix = temp_cont[temp_cont.index > i].index.min()
        if any(rowdata[:3] == 0):  # 如果0,1,total有一项为0，则运行
            #
            temp_cont.loc[ix, target] = temp_cont.loc[ix, target] + rowdata[target]
            temp_cont.loc[ix, 0] = temp_cont.loc[ix, 0] + rowdata[0]
            temp_cont.loc[ix, 'total'] = temp_cont.loc[ix, 'total'] + rowdata['total']
            #
            if i < temp_cont.index.max():
                temp_cont.loc[ix, 'lower'] = rowdata['lower']
            else:
                temp_cont.loc[ix, 'upper'] = rowdata['upper']
            temp_cont = temp_cont.drop(i, axis=0)

    return temp_cont.sort_values(by='pdv1')


# 根据选定的方法计算当前分箱
def _calCMerit(temp, ix, method):
    """
    Calculation of the merit function for the current table temp
    ---------------------------------------------
    Params
    temp: pandas dataframe, temp table in _bestSplit
    ix: single int obj,index of temp, from length of temp
    method: int obj, metric to split x(1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    ---------------------------------------------
    Return
    M_value: float or np.nan
    """
    # split data by ix
    temp_L = temp[temp['i'] <= ix]
    temp_U = temp[temp['i'] > ix]
    # calculate sum of 0, 1, total for each splited data
    # 计算每个bin分组0、1和total数
    n_11 = float(sum(temp_L[0]))
    n_12 = float(sum(temp_L[1]))
    n_21 = float(sum(temp_U[0]))
    n_22 = float(sum(temp_U[1]))
    n_1s = float(sum(temp_L['total']))
    n_2s = float(sum(temp_U['total']))
    # calculate sum of 0, 1 for whole data
    n_s1 = float(sum(temp[0]))
    n_s2 = float(sum(temp[1]))
    N_mat = np.array([[n_11, n_12, n_1s],
                      [n_21, n_22, n_2s]])
    N_s = [n_s1, n_s2]
    # Gini
    if method == 1:
        N = n_1s + n_2s
        G1 = 1 - ((n_11 * n_11 + n_12 * n_12) / float(n_1s * n_1s))
        G2 = 1 - ((n_21 * n_21 + n_22 * n_22) / float(n_2s * n_2s))
        G = 1 - ((n_s1 * n_s1 + n_s2 * n_s2) / float(N * N))
        M_value = 1 - ((n_1s * G1 + n_2s * G2) / float(N * G))
    # Entropy
    elif method == 2:
        N = n_1s + n_2s
        E1 = -((n_11 / n_1s) * (np.log((n_11 / n_1s))) + \
               (n_12 / n_1s) * (np.log((n_12 / n_1s)))) / (np.log(2))
        E2 = -((n_21 / n_2s) * (np.log((n_21 / n_2s))) + \
               (n_22 / n_2s) * (np.log((n_22 / n_2s)))) / (np.log(2))
        E = -(((n_s1 / N) * (np.log((n_s1 / N))) + ((n_s2 / N) * \
                                                    np.log((n_s2 / N)))) / (np.log(2)))
        M_value = 1 - (n_1s * E1 + n_2s * E2) / (N * E)
    # Pearson chisq
    elif method == 3:
        N = n_1s + n_2s
        X2 = 0
        M = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                M[i][j] = N_mat[i][2] * N_s[j] / N
                X2 = X2 + ((N_mat[i][j] - M[i][j]) * (N_mat[i][j] - M[i][j])) / M[i][j]

        M_value = X2
    # Info Value
    else:
        try:
            IV = ((n_11 / n_s1) - (n_12 / n_s2)) * np.log((n_11 * n_s2) / (n_12 * n_s1)) \
                 + ((n_21 / n_s1) - (n_22 / n_s2)) * np.log((n_21 * n_s2) / (n_22 * n_s1))
            M_value = IV
        except ZeroDivisionError:
            M_value = np.nan
    return M_value


"""
根据计算指标寻找到最优分割点
"""
def _bestSplit(binDS, method, BinNo):
    """
    find the best split for one bin dataset
    middle procession functions for _candSplit
    --------------------------------------
    Params
    binDS: pandas dataframe, middle bining table
    method: int obj, metric to split x
        (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    BinNo: int obj, bin number of binDS
    --------------------------------------
    Return
    newbinDS: pandas dataframe
    """
    binDS = binDS.sort_values(by=['bin', 'pdv1'])
    # 计算当前bin的分组数
    mb = len(binDS[binDS['bin'] == BinNo])
    bestValue = 0
    bestI = 1
    # 遍历bin，
    for i in range(1, mb):
        # split data by i
        # metric: Gini,Entropy,pearson chisq,Info value
        value = _calCMerit(binDS, i, method)
        # if value>bestValue，then make value=bestValue，and bestI = i
        if bestValue < value:
            bestValue = value
            bestI = i
    # create new var split
    binDS['split'] = np.where(binDS['i'] <= bestI, 1, 0)
    binDS = binDS.drop('i', axis=1)
    newbinDS = binDS.sort_values(by=['split', 'pdv1'])
    # rebuild var i
    newbinDS_0 = newbinDS[newbinDS['split'] == 0]
    newbinDS_1 = newbinDS[newbinDS['split'] == 1]
    newbinDS_0['i'] = range(1, len(newbinDS_0) + 1)
    newbinDS_1['i'] = range(1, len(newbinDS_1) + 1)
    newbinDS = pd.concat([newbinDS_0, newbinDS_1], axis=0)
    return newbinDS  # .sort_values(by=['split','pdv1'])


def _Gvalue(binDS, method):
    """
    Calculation of the metric of current split
    ----------------------------------------
    Params
    binDS: pandas dataframe
    method: int obj, metric to split x(1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    -----------------------------------------
    Return
    M_value: float or np.nan
    """
    R = binDS['bin'].max()
    N = binDS['total'].sum()

    N_mat = np.empty((R, 3))
    # calculate sum of 0,1
    N_s = [binDS[0].sum(), binDS[1].sum()]
    # calculate each bin's sum of 0,1,total
    # store values in R*3 ndarray
    for i in range(int(R)):
        subDS = binDS[binDS['bin'] == (i + 1)]
        N_mat[i][0] = subDS[0].sum()
        N_mat[i][1] = subDS[1].sum()
        N_mat[i][2] = subDS['total'].sum()

    # Gini
    if method == 1:
        G_list = [0] * R
        for i in range(int(R)):

            for j in range(2):
                G_list[i] = G_list[i] + N_mat[i][j] * N_mat[i][j]
            G_list[i] = 1 - G_list[i] / (N_mat[i][2] * N_mat[i][2])
        G = 0
        for j in range(2):
            G = G + N_s[j] * N_s[j]

        G = 1 - G / (N * N)
        Gr = 0
        for i in range(int(R)):
            Gr = Gr + N_mat[i][2] * (G_list[i] / N)
        M_value = 1 - Gr / G
    # Entropy
    elif method == 2:
        for i in range(int(R)):
            for j in range(2):
                if np.isnan(N_mat[i][j]) or N_mat[i][j] == 0:
                    M_value = 0

        E_list = [0] * R
        for i in range(int(R)):
            for j in range(2):
                E_list[i] = E_list[i] - ((N_mat[i][j] / float(N_mat[i][2])) \
                                         * np.log(N_mat[i][j] / N_mat[i][2]))

            E_list[i] = E_list[i] / np.log(2)  # plus
        E = 0
        for j in range(2):
            a = (N_s[j] / N)
            E = E - a * (np.log(a))

        E = E / np.log(2)
        Er = 0
        for i in range(2):
            Er = Er + N_mat[i][2] * E_list[i] / N
        M_value = 1 - (Er / E)
        return M_value
    # Pearson X2
    elif method == 3:
        N = N_s[0] + N_s[1]
        X2 = 0
        M = np.empty((R, 2))
        for i in range(int(R)):
            for j in range(2):
                M[i][j] = N_mat[i][2] * N_s[j] / N
                X2 = X2 + (N_mat[i][j] - M[i][j]) * (N_mat[i][j] - M[i][j]) / (M[i][j])

        M_value = X2
    # Info value
    else:
        if any([_isNullZero(N_mat[i][0]),
                _isNullZero(N_mat[i][1]),
                _isNullZero(N_s[0]),
                _isNullZero(N_s[1])]):
            M_value = np.NaN
        else:
            IV = 0
            for i in range(int(R)):
                IV = IV + (N_mat[i][0] / N_s[0] - N_mat[i][1] / N_s[1]) \
                     * np.log((N_mat[i][0] * N_s[1]) / (N_mat[i][1] * N_s[0]))
            M_value = IV

    return M_value


# 根据选定的最优分箱方法
def _candSplit(binDS, method):
    """
    Generate all candidate splits from current Bins
    and select the best new bins
    middle procession functions for binContVar & reduceCats
    ---------------------------------------------
    Params
    binDS: pandas dataframe, middle bining table
    method: int obj, metric to split x
        (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    --------------------------------------------
    Return
    newBins: pandas dataframe, split results
    """
    # sorted data by bin&pdv1
    binDS = binDS.sort_values(by=['bin','pdv1'])
    # get the maximum of bin
    # 当前已划分的bin数
    Bmax = max(binDS['bin'])
    # screen data and cal nrows by diffrence bin
    # and save the results in dict
    # 保存不同bin的相关信息
    temp_binC = dict()
    m = dict()
    # 循环遍历分解待分箱的数据
    for i in range(1, Bmax+1):
        temp_binC[i] = binDS[binDS['bin'] == i]
        m[i] = len(temp_binC[i])
    """
    CC
    """
    # create null dataframe to save info
    temp_trysplit = dict()
    temp_main = dict()
    bin_i_value = []

    """
    1）遍历每个分箱bin，计算最优的分箱点；
    2）计算每个分箱的指标特征值；
    3）根据指标特征值选择需要划分的最优分箱；
    """
    for i in range(1, Bmax+1):
        if m[i] > 1: # if nrows of bin > 1
            # split data by best i
            temp_trysplit[i] = _bestSplit(temp_binC[i], method, i)
            temp_trysplit[i]['bin'] = np.where(temp_trysplit[i]['split']==1,
                                               Bmax+1,
                                               temp_trysplit[i]['bin'])
            # delete bin == i
            temp_main[i] = binDS[binDS['bin']!=i]
            # vertical combine temp_main[i] & temp_trysplit[i]
            temp_main[i] = pd.concat([temp_main[i],temp_trysplit[i]], axis=0)
            # calculate metric of temp_main[i]
            value = _Gvalue(temp_main[i], method)
            newdata = [i, value]
            bin_i_value.append(newdata)
    # find maxinum of value bintoSplit
    bin_i_value.sort(key=lambda x:x[1], reverse=True)
    # binNum = temp_all_Vals['BinToSplit']
    binNum = bin_i_value[0][0]
    newBins = temp_main[binNum].drop('split', axis=1)
    return newBins.sort_values(by=['bin', 'pdv1'])


def _getNewBins(sub, i):
    """
    get new lower, upper, bin, total for sub
    middle procession functions for binContVar
    -----------------------------------------
    Params
    sub: pandas dataframe, subdataframe of temp_map
    i: int, bin number of sub
    ----------------------------------------
    Return
    df: pandas dataframe, one row
    """
    l = len(sub)
    total = sub['total'].sum()
    first = sub.iloc[0, :]
    last = sub.iloc[l - 1, :]

    lower = first['lower']
    upper = last['upper']
    df = pd.DataFrame()
    df = df.append([i, lower, upper, total], ignore_index=True).T
    df.columns = ['bin', 'lower', 'upper', 'total']
    return df


def binContVar(x, y, method, mmax=5, Acc=0.01, target=1, adjust=0.0001):
    """
    连续型变量的最优分箱：
    1）连续变量先按等宽处理，等宽距离按100等分处理，依据参数调整大小
    2）分箱方法常见指标：a、Gini值；b、Entropy熵；c、person chisq卡方值；d、iv信息值
    3）逐箱变量计算相应指标，找到区分效果最高的切分点
    4）基于上述环节找到的最优分箱点，对x变量进行分箱，并计算当前各分箱对应的指标值
    5）判断当前分箱数量，若小于设定分箱参数，则基于各分箱继续前面几步迭代分箱，直到分到最终设定箱数
    ----------------------------------------------------------
    Params
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    method: int obj, metric to split x
    mmax: int, bining number
    Acc: float less than 1, partition ratio for equal width bining
    badlabel: target label
    adjust: float or np.inf, bining adjust for limitation
    ----------------------------------------------------------
    """

    # if y is not 0-1 binary variable, then raise a error
    _check_target_binary(y)

    # data bining by Acc, method: width equal
    # 返回X变量对应的等距分箱分段
    bin_map = _EqualWidthBinMap(x, Acc, adjust=adjust, method=0)

    # mapping x to bin number and combine with x&y
    # 将分箱后的X变量作转换映射
    bin_res = _applyBinMap(x, bin_map)


    """
    1）组合x,y以及映射后的分箱；
    2）分组计算0、1的频数；
    3）针对0、1的分箱进行合并；
    """
    temp_df = pd.concat([x, y, bin_res], axis=1)
    # 使用列联表分组统计
    t1 = pd.crosstab(index=temp_df[bin_res.name], columns=y)
    # 统计每个分组总数
    t2 = temp_df.groupby(bin_res.name).count().iloc[:, 0]
    t2 = pd.DataFrame(t2)
    t2.columns = ['total']
    # 合并t1和t2统计，即统计每个分段里面0、1以及0+1的情况
    t = pd.concat([t1, t2], axis=1)
    # merge t & bin_map by t,
    # if all(0,1,total) == 1, so corresponding row will not appear in temp_cont
    temp_cont = pd.merge(t, bin_map,
                         left_index=True, right_index=True,
                         how='left')
    # 最终合并结果，分箱后各分段的统计
    temp_cont['pdv1'] = temp_cont.index

    # if any(0,1,total)==0, then combine it with per bin or next bin
    # 合并bins
    temp_cont = _combineBins(temp_cont, target)
    # calculate other temp vars
    temp_cont['bin'] = 1
    temp_cont['i'] = range(1, len(temp_cont) + 1)
    temp_cont['var'] = temp_cont.index


    """
    1）连续型变量已按照等频或等宽方式分组；
    2）按照设定的分箱数，循环遍历计算各分组对应指标；
    3）指标计算方式：基尼值、熵、卡方值、信息值；
    """
    nbins = 1
    # exe candSplit mmax times
    # 递进分箱处理
    while (nbins < mmax):
        temp_cont = _candSplit(temp_cont, method=method)
        nbins += 1

    temp_cont = temp_cont.rename(columns={'var': 'oldbin'})
    temp_Map1 = temp_cont.drop([0, target, 'pdv1', 'i'], axis=1)
    temp_Map1 = temp_Map1.sort_values(by=['bin', 'oldbin'])
    # get new lower, upper, bin, total for sub
    data = pd.DataFrame()
    s = set()
    # # 最终分箱
    for i in temp_Map1['bin']:
        if i in s:
            pass
        else:
            sub_Map = temp_Map1[temp_Map1['bin'] == i]
            rowdata = _getNewBins(sub_Map, i)
            data = data.append(rowdata, ignore_index=True)
            s.add(i)

    # resort data
    data = data.sort_values(by='lower')
    data['newbin'] = range(1, mmax + 1)
    data = data.drop('bin', axis=1)
    data.index = data['newbin']
    return data


def _groupCal(x, y, badlabel=1):
    """
    group calulate for x by y
    middle proporcessing function for reduceCats
    -------------------------------------
    Params
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    badlabel: target label
    ------------------------------------
    Return
    temp_cont: group calulate table
    m: nrows of temp_cont
    """

    temp_cont = pd.crosstab(index=x, columns=y, margins=False)
    temp_cont['total'] = temp_cont.sum(axis=1)
    temp_cont['pdv1'] = temp_cont[badlabel] / temp_cont['total']

    temp_cont['i'] = range(1, temp_cont.shape[0] + 1)
    temp_cont['bin'] = 1
    m = temp_cont.shape[0]
    return temp_cont, m


# 类别变量的降基处理
def reduceCats(x, y, method=1, mmax=5, badlabel=1):
    """
    Reduce category for x by y & method
    method is represent by number,
        1:Gini, 2:Entropy, 3:person chisq, 4:Info value
    ----------------------------------------------
    Params:
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    method: int obj, metric to split x
    mmax: number to reduce
    badlabel: target label
    ---------------------------------------------
    Return
    temp_cont: pandas dataframe, reduct category map
    """
    _check_target_binary(y)
    temp_cont, m = _groupCal(x, y, badlabel=badlabel)
    nbins = 1
    while (nbins < mmax):
        temp_cont = _candSplit(temp_cont, method=method)
        nbins += 1

    temp_cont = temp_cont.rename(columns={'var': x.name})
    temp_cont = temp_cont.drop([0, 1, 'i', 'pdv1'], axis=1)
    return temp_cont.sort_values(by='bin')


def applyMapCats(x, bin_map):
    """
    convert x to newbin by bin_map
    ------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, mapTable contain new bins
    ------------------------------
    Return
    new_x: pandas Series, convert results
    """
    d = dict()
    for i in bin_map.index:
        subData = bin_map[bin_map.index == i]
        value = subData.loc[i, 'bin']
        d[i] = value

    new_x = x.map(d)
    new_x.name = x.name + '_BIN'
    return new_x


def tableTranslate(red_map):
    """
    table tranlate for red_map
    ---------------------------
    Params
    red_map: pandas dataframe,reduceCats results
    ---------------------------
    Return
    res: pandas series
    """
    l = red_map['bin'].unique()
    res = pd.Series(index=l)
    for i in l:
        value = red_map[red_map['bin'] == i].index
        value = list(value.map(lambda x: str(x) + ';'))
        value = "".join(value)
        res[i] = value
    return res


if __name__ == '__main__':
    df = pd.read_excel("credit_review_new_all_0922.xlsx")
    tmp_df = reduceCats(df["continent"],df["def_pd10"],)
    print(tmp_df)
    # # load data
    # path = "E:/project/risk_strategy/data/UCI_Credit_Card.csv"
    # raw_data = pd.read_csv(path)
    # df = raw_data.copy()
    # df_y = df['label']
    # df_x = df.drop(['ID', 'label'], axis=1)
    # #df_x = df[['LIMIT_BAL', 'PAY_0']]
    #
    # """
    # 数据勘探：1）离散型变量；2）连续型变量；
    # 数值变量：唯一值数量>20即为连续型，否则判为离散型
    # """
    # # 离散型list
    # discrete = []
    # # 连续型list
    # continues = []
    # # 不需要分箱的变量类型
    # other = []
    # for v in df_x.columns:
    #     if len(df_x[v].unique()) > 20:
    #         continues.append(v)
    #     elif len(df_x[v].unique()) > 5:  # 此处参数阈值需与后文 mmax 最优分箱数 保持一致
    #         discrete.append(v)
    #     else:
    #         other.append(v)
    # # 离散型变量
    # df_discrete = df[discrete]
    # # 连续型变量
    # df_continues = df[continues]
    # # 不需处理的变量
    # df_other = df[other]
    #
    # """
    # 最优分箱处理：
    # 1）连续型变量按最优分箱；
    # 2）离散型变量做降基处理；
    # 3）分箱样本异常值可提前处理，使用均值+/-3倍标准差替代极端异常值；
    # """
    # # 连续型变量分箱处理
    # new_ds = pd.DataFrame()
    # # 变量列循环处理
    # for v in df_continues.columns:
    #     x = df_continues[v]
    #     # 连续型变量最优分箱
    #     bin_map = binContVar(x, df_y, method=4)
    #     # 组装
    #     new_x = _applyBinMap(x, bin_map)
    #     new_x.name = v + "_BIN"
    #     new_ds = pd.concat([new_ds, new_x], axis=1)
    # # 连续型变量分箱结果
    # # print(new_ds)
    #
    # # 离散变量降基处理
    # new_dc = pd.DataFrame()
    # bin_maps_str = dict()
    # for var in df_discrete.columns:
    #     # print("当前处理的变量x:", var)
    #     x = df_discrete[var]
    #     single_map = reduceCats(x, df_y, method=4)
    #     # 组装 根据映射map转换
    #     new_x = applyMapCats(x, single_map)
    #     new_dc = pd.concat([new_dc, new_x], axis=1)
    # # 离散型变量降基结果
    # # print(new_dc)
    #
    # # data combine
    # new_df = pd.concat([new_ds, new_dc, df_other], axis=1)
    # new_df.to_excel("E:/project/risk_strategy/data/new_df.xlsx")