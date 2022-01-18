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


class Discretization(metaclass=ABCMeta):
    """
    离散化基类，包含了基本的参数定义和特征预处理的方法
    Parameters
    ----------
    max_bin: int 最大分箱数量
    init_bin: int  初始化分箱的数量，若为空则钚进行初始化的分箱
    init_method : str 初始化方法默认为'qcut' , 初始化分箱方法，目前仅支持 'qcut' 和 'cut'
    precision : 小数精度，默认为3
    print_process : bool 是否答应分箱的过程信息
    positive_label : 正样本的定义，根据target的数据类型来定
    """

    def __init__(self, max_bin=12, init_bin=100, init_method='qcut', precision=3, print_process=False,
                 positive_label=1):
        self.max_bin = max_bin
        self.init_bin = init_bin
        self.init_method = init_method
        self.precision = precision
        self.print_process = print_process
        self.positive_label = positive_label
        self.is_numeric = True

    def data_init(self, dat, var_name, target):
        """
        init the input：
        1、check if the target variable is binary
        2、judge x is numeric dtype
        3、init the data todo
        """
        assert var_name in dat.columns, "数据中不包含变量%s，请检查数据" % (var_name)
        assert var_name == target, "因变量和自变量必须是不同的变量"

        self._y_check(dat[target])

        self.is_numeric = is_numeric_dtype(dat[var_name])
        self._x_check(dat[var_name])



        count_df = pd.crosstab(dat[var_name],dat[target])

    def _x_check(self, dat, var_name):
        """
        对自变量进行校验：校验是否存在缺失值
        ------------------------------
        Return
        若自变量中存在缺失值, 报错
        """
        x_na_count = pd.isna(dat[var_name]).sum()
        assert x_na_count != 0, f"自变量'{var_name}'中存在缺失值，自动分箱前请处理自变量中的缺失值"

    def _y_check(self, y):
        """
        判断目标变量是否为二分类变量
        ------------------------------
        Param
        y:exog variable,pandas Series contains binary variable
        ------------------------------
        Return
        若目标变量非二分类变量, 报错
        """
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('目标变量必须是二元的！')

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
        pass

    def bestDsct(self, dat: pd.DataFrame, var_name: str, target: str):
        """
        在基础分箱的方法上增加单调性校验功能
        """
        pass


class ChiMerge(Discretization):
    """
    卡方分箱法
    """

    def dsct(self, dat: pd.DataFrame, col: str, target: str):
        # todo 基于卡方分箱法实现特征离散化
        print("卡方分箱法被调用")


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

    def base_info(self, dat, cols, target):
        pass

    def IV(self, dat, var_name, target):
        pass

    def woe_code(self, dat, var_name, target, ):
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
    bsetks = BestKS()
    print(bsetks.precision)
    bsetks.precision = 4
    print(bsetks.precision)
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
