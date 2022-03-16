# -*- coding: utf-8 -*-

"""
单变量分析中常用工具，主要包含以下几类工具：
1、自动分箱（降基）模块：包括卡方分箱、Best-ks分箱
2、基本分析模块，单变量分析工具，以及woe编码工具，以及所有变量的分析报告
3、单变量分析绘图工具，如AUC，KS，分布相关的图
"""
# Author: Leon Long

import numpy as np
import pandas as pd
from abc import abstractmethod
from abc import ABCMeta
from sklearn.utils.multiclass import type_of_target
from pandas.api.types import is_numeric_dtype
import warnings
import time


class Discretization(metaclass=ABCMeta):
    """
    离散化基类，包含了基本的参数定义和特征预处理的方法
    注：分箱的预处理过程中就会剔除x变量中的缺失值，若要将缺失值也纳入分箱运算过程，请先在数据中进行填充
    Parameters
    ----------
    max_bin: int 最大分箱数量
    init_thredhold: int  初始化分箱的数量，若为空则钚进行初始化的分箱
    init_method : str 初始化方法默认为'qcut' , 初始化分箱方法，目前仅支持 'qcut' 和 'cut'
    precision : 小数精度，默认为3
    print_process : bool 是否答应分箱的过程信息
    positive_label : 正样本的定义，根据target的数据类型来定
    """

    def __init__(self, max_bin=12, init_thredhold=100, init_method='qcut', precision=3, print_process=False,
                 positive_label=1, negative_label=0, max_bestDsct_bin=4):
        self.max_bin = max_bin
        self.init_thredhold = init_thredhold
        self.init_method = init_method
        self.precision = precision
        self.print_process = print_process
        self.positive_label = positive_label
        self.negative_label = negative_label

    @abstractmethod
    def dsct(self, dat: pd.DataFrame, var_name: str, target: str):
        """
        抽象接口，定义分箱方法的名称和入参，只能调用子类实现
        Parameters
        ----------
        dat: 数据集，格式必须为pd.DataFrame
        var_name:  待分箱的因变量
        target: 目标变量
        """
        raise NotImplementedError("该方法只为定义基本的函数接口，不可直接调用，请使用子类实现的方法")

    def _init_data(self, data, var_name, target):
        """
        init the input：
        1、校验自变量和因变量的类型
        2、判断x是否为数值类型
        3、初始化数据结果
        """
        # assert var_name in dat.columns, "数据中不包含变量%s，请检查数据" % (var_name)
        # assert var_name != target, "因变量和自变量必须是不同的变量"
        #
        # dat = dat[[var_name, target]].copy()
        #
        # self._y_check(dat[target])
        #
        # is_numeric = is_numeric_dtype(dat[var_name])
        #
        # # unique = self.unique_noNA(dat[var_name])
        #
        # if is_numeric:
        #     dat[var_name] = pd.qcut(dat[var_name], self.init_thredhold, duplicates="drop", precision=self.precision)
        #     dti = pd.crosstab(dat[var_name], dat[target], dropna=True)
        #     dti["positive_rate"] = dti[self.positive_label] / dti.sum(axis=1)
        #     dti["bin"] = dti.index.map(lambda x: x.right)
        #     mapping = None
        # else:
        #     dti = pd.crosstab(dat[var_name], dat[target], dropna=True)
        #     dti["positive_rate"] = dti[self.positive_label] / dti.sum(axis=1)
        #     dti = dti.sort_values(by="positive_rate").reset_index().reset_index().rename({"index": "bin"},
        #                                                                                  axis=1)
        #     mapping = dti[[var_name, "bin"]].copy()
        #
        # return dti[["bin", self.negative_label, self.positive_label]], var_name, mapping, is_numeric
        if self.print_process: print("开始对变量{}进行卡方分箱:".format(x, ))

        time0 = time.time()
        assert var_name in data.columns, "数据中不包含变量%s，请检查数据" % (var_name)
        assert var_name != target, "因变量和自变量必须是不同的变量"
        data = data[[var_name, target]].copy()

        self._y_check(data[target])

        is_numeric = is_numeric_dtype(data[var_name])

        if is_numeric:
            if self.init_thredhold is not None:
                if self.init_method == 'qcut':
                    data.loc[:, var_name] = pd.qcut(data[var_name], self.init_thredhold, duplicates="drop",
                                                    precision=self.precision)
                elif self.init_method == 'cut':
                    data.loc[:, var_name] = pd.cut(data[var_name], self.init_thredhold, duplicates="drop",
                                                   precision=self.precision)
                else:
                    raise ValueError("init_method参数仅有qcut和cut两个选项")
                data.loc[:, var_name] = data[var_name].map(lambda x: x.right).astype(float)
            dti = pd.crosstab(data[var_name], data[target], dropna=False).reset_index()
            dti.rename({self.negative_label: "negative", self.positive_label: "positive"}, axis=1, inplace=True)
            dti["variable"] = dti[var_name]
            mapping = None
        else:
            dti = pd.crosstab(data[var_name], data[target]).reset_index()
            dti.rename({self.negative_label: "negative", self.positive_label: "positive"}, axis=1, inplace=True)
            dti["positive_rate"] = dti["positive"] / (dti["positive"] + dti["negative"])
            dti = dti.sort_values("positive_rate").reset_index(drop=True).reset_index()
            dti.rename({"index": "variable"}, axis=1, inplace=True)
            mapping = dti[[var_name, "variable", "negative", "positive"]]
        dti = dti[["variable", "negative", "positive"]].copy()

        if self.print_process:
            time1 = time.time()
            print("==>已经完成变量初始化耗时{:.2f}s,开始处理0值".format(time1 - time0))

        while (len(dti) > self.max_bin) and (len(dti.query('(negative==0) or (positive==0)')) > 0):
            dti["count"] = dti["negative"] + dti["positive"]
            rm_bk = dti.query("(negative==0) or (positive==0)") \
                .query("count == count.min()")
            ind = rm_bk["variable"].index[0]
            if ind == dti["variable"].index.max():
                dti.loc[ind - 1, "variable"] = dti.loc[ind, "variable"]
            else:
                dti.loc[ind, "variable"] = dti.loc[ind + 1, "variable"]
            dti = dti.groupby("variable")[["negative", "positive"]].sum().reset_index()

        if self.print_process:
            time2 = time.time()
            print("==>已经完成0值处理耗时{:.2f}s,开始进行分箱迭代".format(time2 - time1))

        return dti, var_name, is_numeric, mapping

    def _normalize_output(self, dti, var_name, is_numeric, mapping):
        """
        根据分箱后计算出来的结果，标准化输出
        """
        break_points = dti['variable'].copy()
        break_points[np.where(break_points == break_points.max())[0]] = np.inf
        break_points = np.concatenate(([-np.inf], break_points))

        if is_numeric:
            cond = list(break_points)
        else:
            interval_index = pd.IntervalIndex.from_breaks(break_points, closed='right')
            mapping["bin"] = mapping["variable"].map(lambda x: np.where(interval_index.contains(x))[0][0])
            cond = pd.Series(mapping["bin"].values, index=mapping[var_name].values).to_dict()

        return cond

    # def unique_noNA(self, x: pd.Series):
    #     """
    #     pandas 中存在bug，许多场景的group 和 crosstab中的dropna参数的设定会失效，
    #     在分箱的过程中剔除缺失值，若要考虑缺失值的场景，请提前对数据进行fillNa操作。
    #     """
    #     return np.array(list(filter(lambda ele: ele == ele, x.unique())))

    def _x_check(self, dat, var_name):
        """
        对自变量进行校验：校验是否存在缺失值
        ------------------------------
        Return
        若自变量中存在缺失值, 报错
        """
        x_na_count = pd.isna(dat[var_name]).sum()
        assert x_na_count != 0, f"自变量'{var_name}'中存在缺失值，自动分箱前请处理自变量中的缺失值"

    def _y_check(self, y: pd.Series):
        """
        校验y值是否符合以下两个条件:
        1、y值必须是二分类变量
        2、positive_label必须为y中的结果
        ------------------------------
        Param
        y:exog variable,pandas Series contains binary variable
        ------------------------------
        """
        y_type = type_of_target(y)
        # if y_type not in ['binary']:
        #     raise ValueError('目标变量必须是二元的！')
        # if self.positive_label not in y:
        #     raise ValueError('请根据设定positive_label')
        assert y_type in ['binary'], "目标必须是二分类"
        assert not y.hasnans, "target中不能包含缺失值，请优先进行填充"
        assert self.positive_label in y, "请根据target正确设定positive_label"
        assert self.negative_label in y, "请根据target的结果正确设定negative_label"

    def _check_target_type(self, y):
        """
        判断y的类型，将y限定为 0和1 的数组。
        """
        warnings.warn("some_old_function is deprecated", DeprecationWarning)
        y_unique = y.unique()
        if len(y_unique) != 2:
            raise ValueError("y必须是二值型变量,且其元素必须是 0 1")
        for item in y_unique:
            if item not in [0, 1]:
                raise ValueError("y必须是二值型变量,且其元素必须是 0 1")

    def bestDsct(self, dat: pd.DataFrame, var_name: str, target: str):
        """
        在基础分箱的方法上增加单调性校验功能
        """
        pass


class ChiMerge(Discretization):
    """
    卡方分箱法
    """

    def dsct(self, dat: pd.DataFrame, var_name: str, target: str):
        dti, var_name, is_numeric, mapping = self._init_data(dat, var_name, target)

        time0 = time.time()

        while len(dti) > self.max_bin:
            dti["chi2"] = dti.apply(lambda row: self._calc_chi2(dti, row), axis=1)
            min_chi2_ind = dti.query("chi2 == chi2.min()").index[0]
            dti.loc[min_chi2_ind - 1, "variable"] = dti.loc[min_chi2_ind, "variable"]
            dti = dti.groupby("variable")[["negative", "positive"]].sum().reset_index()

        if self.print_process:
            time1 = time.time()
            print("==>完成分箱迭代耗时{:.2f}s".format(time1 - time0))
        return self._normalize_output(dti, var_name, is_numeric, mapping)

    def _calc_chi2(self, dti, row):
        ind0 = dti[dti['variable'] == row['variable']].index[0]
        if ind0 == dti.index.min():
            return np.inf
        ind1 = ind0 - 1
        a = dti.loc[ind1, 'negative']
        b = dti.loc[ind1, 'positive']
        c = dti.loc[ind0, 'negative']
        d = dti.loc[ind0, 'positive']
        return self._chi2(a, b, c, d)

    def _chi2(self, a, b, c, d):
        """
        如下横纵标对应的卡方计算公式为： K^2 = n (ad - bc) ^ 2 / [(a+b)(c+d)(a+c)(b+d)]　其中n=a+b+c+d为样本容量
            y1   y2
        x1  a    b
        x2  c    d
        :return: 卡方值
        """
        a, b, c, d = float(a), float(b), float(c), float(d)
        return ((a + b + c + d) * ((a * d - b * c) ** 2)) / ((a + b) * (c + d) * (b + d) * (a + c))


class BestKS(Discretization):
    """
    Best-KS 分箱法
    """

    def dsct(self, dat: pd.DataFrame, col: str, target: str):
        # todo 基于卡方分箱法实现特征离散化
        print("bestKS分箱法被调用")


class Univariate(object):
    """
    分装单变量分析的基本工具类
    """

    def __init__(self, max_bins=12, init_bins=100, init_method='qcut', precision=4, print_process=False):
        self.max_bins = max_bins
        self.init_bins = init_bins
        self.init_method = init_method
        self.precision = precision
        self.print_process = print_process
        self.is_numeric = True

    def base_info(elf, dat: pd.DataFrame, var_list: list, target: str):
        pass

    def IV(self, dat: pd.DataFrame, var_name: str, target: str):
        pass

    def woe_code(self, dat: pd.DataFrame, var_name: str, target: str, ):
        pass

    def feature_report(self, dat: pd.DataFrame, var_list: list, target: str):
        pass


class PlotUtils(object):
    def bubble(self, x, y):
        """
        绘制气泡图，记录撸
        """
        pass

    def auc(self, x, y):
        pass

    def ks(self, x, y):
        pass

    def auc_ks(self, x, y):
        pass


if __name__ == "__main__":
    # rs = Discretization.unique_noNA(pd.Series(np.random.randint(0, 100, 10000)))
    # print(rs)
    data = pd.read_excel("data/credit_data.xlsx")
    print(data.shape)
    chiMerge = ChiMerge(print_process=True)
    print(chiMerge.dsct(data, 'device_app_num', 'def_pd10'))
    print(chiMerge.dsct(data, 'continent', 'def_pd10'))

# bsetks = BestKS()
# print(bsetks.precision)
# bsetks.precision = 4
# print(bsetks.precision)
#     disct = BestKS()
#     df = pd.read_excel("credit_review_new_all_0922.xlsx")
#     # disct.bestDsct(df, "a", "b")
#     # col = "a"
#     # import time
#     #
#     # t1 = time.time()
#     # print(type_of_target(df["def_pd10"]))
#     # t2 = time.time()
#     # print(t2 - t2)
#     #
#     # t1 = time.time()
#     # print(disct._check_target_type(df["def_pd10"]))
#     # t2 = time.time()
#     # print(t2 - t2)
#     #
#     # t1 = time.time()
#     # print(is_numeric_dtype(df["continent"]))
#     # t2 = time.time()
#     # print(t2 - t2)
