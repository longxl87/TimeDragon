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
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


class Kit(object):
    """
    常用工具类
    """

    def __init__(self, positive_label=1, negative_label=0):
        self.positive_label = positive_label
        self.negative_label = negative_label
        pass

    def cond_insert_ind(self, cond, ind):
        """
        为分箱结果中添加需要独立分箱的部分
        cond : 待处理的分箱结果
        arr : 需要独立分箱的数据集合，不管是数值型还是非数值型变量cond，请使用list添加
        """
        if isinstance(cond, list):
            cond = list(set(cond + ind))
            cond.sort()
        else:
            n = len(ind)
            arr = list(set(ind).difference(set(cond.keys())))
            for k, v in cond.items():
                cond[k] = v + n
            for i in range(len(arr)):
                cond[arr[i]] = i
        return cond

    def make_bin(self, df, var_name, cond, precision=3):
        """
        基于cond中的分箱条件，为df中var_name的变量匹配对应的分箱值
        """
        if isinstance(cond, list):
            df["bin"] = pd.cut(df[var_name], cond, duplicates='drop', precision=precision)
        elif isinstance(cond, dict):
            mapping = pd.Series(cond).reset_index().rename({"index": var_name, 0: "bin"}, axis=1)
            df = df[[var_name]].merge(mapping, on=var_name, how='left').set_index(df[[var_name]].index)
        else:
            raise ValueError("参数cond的类型只能为list或者dict")
        return df["bin"]

    def woe_code(self, df, var_name, woeDict):
        """
        对样本的数据进行woe编码，返回完成编码后的
        """
        if isinstance(df[var_name].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            mapping = pd.Series(woeDict).reset_index().rename({"index": var_name, 0: "woe"}, axis=1)
            breaks = mapping[var_name].to_list()
            breaks.insert(0, -np.inf)
            mapping[var_name] = pd.cut(mapping[var_name], breaks, duplicates="drop")
        else:
            mapping = pd.Series(woeDict).reset_index().rename({"index": var_name, 0: "woe"}, axis=1)
        df = df.merge(mapping, on=var_name, how='left').set_index(df.index)
        return df["woe"]

    def univerate(self, df, var_name, target, lamb=0.001, retWoeDict=False):
        """
        单变量分析函数，目前支持的计算指标为 IV,KS,LIFT
        建议用于编码后的数值型变量进行分析,若在前面使用了cond_insert方法调整了cond
        """
        #         dti = pd.crosstab(df[var_name], df[target])
        dti = df.groupby([var_name, target])[target].count().unstack().fillna(0)
        dti.rename({self.positive_label: "positive", self.negative_label: "negative"}, axis=1, inplace=True)
        dti["positive"] = dti["positive"].astype(int)
        dti["negative"] = dti["negative"].astype(int)
        p_t = dti["positive"].sum()
        n_t = dti["negative"].sum()
        t_t = p_t + n_t
        r_t = p_t / t_t
        dti["total"] = dti["positive"] + dti["negative"]
        dti["total_rate"] = dti["total"] / t_t
        dti["positive_rate"] = dti["positive"] / dti["total"]  # (rs["positive"] + rs["negative"])
        dti["negative_cum"] = dti["negative"].cumsum()
        dti["positive_cum"] = dti["positive"].cumsum()
        dti["woe"] = np.log(((dti["negative"] / n_t) + lamb) / ((dti["positive"] / p_t) + lamb))
        dti["LIFT"] = dti["positive_rate"] / r_t
        dti["KS"] = np.abs((dti["positive_cum"] / p_t) - (dti["negative_cum"] / n_t))
        dti["IV"] = (dti["negative"] / n_t - dti["positive"] / p_t) * dti['woe']
        IV = dti["IV"].sum()
        KS = dti["KS"].max()
        dti["IV"] = IV
        dti["KS"] = KS
        dti = dti.reset_index()
        dti.columns.name = None
        dti.rename({"Total": "num", var_name: "bin"}, axis=1, inplace=True)
        dti.insert(0, "target", [target] * dti.shape[0])
        dti.insert(0, "var", [var_name] * dti.shape[0])
        if retWoeDict:
            if isinstance(dti["bin"].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
                dti["v"] = dti["bin"].map(lambda x: x.right)
            else:
                dti["v"] = dti["bin"]
            woeDict = pd.Series(dti["woe"].values, index=dti["v"].values).to_dict()

            #             # 修正根据分箱后，空分组，对应的woe值
            #             if cond0 is not None:
            #                 right0 = set(cond0[1:])
            #                 right1 = set(woeDict.keys())
            #                 for key in right0.difference(right1):
            #                     woeDict[key] = 0

            dti.drop(columns=["negative_cum", "positive_cum", "v"], inplace=True)
            return dti, woeDict
        dti.drop(columns=["negative_cum", "positive_cum"], inplace=True)
        return dti

    def is_numeric(self, series):
        """
        判断变量是否为数值型变量
        """
        return is_numeric_dtype(series)

    def missing_count(self, series):
        """
        计算变量缺失率
        """
        missing_index = pd.isna(series)
        return missing_index.sum()

    def unique_count(self, series):
        """
        计算变量的枚举值数量
        """
        unique_arr = pd.unique(series)
        return unique_arr.size

    def csi(self, base, df, var_name):
        """
        计算不同数据集之间，同一个变量csi
        """
        count1 = base.groupby(var_name)[var_name].count()
        count2 = df.groupby(var_name)[var_name].count()
        t1 = count1.sum()
        t2 = count2.sum()
        c1 = count1 / t1
        c2 = count2 / t2
        csi = (c1 - c2) * np.log(c1 / c2)
        return csi.sum()

    def group_rs(self, data, group, sum_col=[], count_col=[], rate_tupes=[]):
        """
        业务分析工具类，同时对比计算多个target指标，查看结果
        data : 数据集
        sum_col : 需要group_sum的列
        count_col : 需要group_count的列
        rate_tupe : 需要除法计算的列 格式为 (字段1，字段2，新列名称) 或者 (字段，新列名称)
        """
        grouped = data.groupby(group)
        grouped_count = grouped[count_col].count()
        grouped_sum = grouped[sum_col].sum()
        grouped = pd.concat([grouped_count, grouped_sum], axis=1)
        for tup in rate_tupes:
            size = len(tup)
            if size == 3:
                grouped[tup[2]] = grouped[tup[0]] / grouped[tup[1]]
            if size == 2:
                grouped[tup[1]] = grouped[tup[0]] / grouped[tup[0]].sum()
        return grouped.reset_index()

    def batch_fillna(self, df, var_list, num_fill=-1, cate_fill="NA", suffix="_new"):
        """
        批量填充缺失值
        """
        for var_name in var_list:
            var_name_new = var_name + suffix
            if self.is_numeric(df[var_name]):
                df[var_name_new] = df[var_name].fillna(num_fill)
            else:
                df[var_name_new] = df[var_name].fillna(cate_fill)
        return df

    def varlist_suffix(self, var_list, suffix):
        return [var_name + suffix for var_name in var_list]

    def feature_engine(self, datas, var_list, target, discretize, max_bin=6, precision=4, num_fill=-1, cate_fill="NA",
                       num_ind=None, cate_ind=None, fill_suffix="_fill", bin_suffix="_bin", woe_suffix="_woe",
                       path=None):
        """
        批量对数据集进行自动化分箱和编码
        Parameters
        ----------
        datas: 数据集，为dataframe的list，第一个数据集为训练集
        var_list: 特征列表
        target : 目标值
        discretize : 分箱工具类
        max_bin : 最大分箱数
        num_fill : 数值型变量填充结果
        cate_fill : 类别型变量填充结果
        num_ind : 数值型变量中，需要独立插入的分箱 为 list
        cate_ind : 字符型变量中，需要独立进行分箱的值 为 list
        fill_suffix : 处理确实
        bin_suffix : 分箱后生成对应分箱的后缀
        woe_suffix : woe编码后的编码的后缀
        retInfoDict : 返回分箱后的变量信息，采用嵌套的dict格式，单个变量的相关信息如下：
                    变量名 ： { "cond" , "woeDict" }
        """
        assert len(datas) >= 1, "至少需要一个数据集"
        train = datas[0]
        all_data = pd.concat(datas, axis=1)
        info_dict = {}
        for var_name in var_list:
            print(f"开始处理变量:'{var_name}'")
            missing = self.missing_count(train[var_name])
            missing_rate = (missing * 1.0) / (train.shape[0])
            unique = self.unique_count(train[var_name])
            info_dict[var_name] = {}
            info_dict[var_name]['missing'] = missing
            info_dict[var_name]['missing_rate'] = missing_rate
            info_dict[var_name]['unique'] = unique

            is_numeric = self.is_numeric(train[var_name])
            var_name_new = var_name
            if is_numeric:
                if num_fill is not None:
                    var_name_new = var_name + fill_suffix
                    for df in datas:
                        df[var_name_new] = df[var_name].fillna(num_fill)
                cond = discretize.dsct(train, var_name, target, max_bin)
                if num_ind is not None:
                    cond = self.cond_insert_ind(cond, num_ind)
                type = 'numeric'
            else:
                if cate_fill is not None:
                    var_name_new = var_name + fill_suffix
                    for df in datas:
                        df[var_name_new] = df[var_name].fillna(cate_fill)
                check = []
                unique0 = set(train[var_name].unique())
                for df in datas[1:]:
                    diff = unique0.difference(df[var_name].unique())
                    check.append(diff)
                if len(check > 0):
                    cond = discretize.dsct(all_data, var_name_new, target, max_bin)
                    type = 'complex'
                else:
                    cond = discretize.dsct(train, var_name_new, target, max_bin)
                    type = 'category'
                if cate_ind is not None:
                    cond = self.cond_insert_ind(cond, cate_ind)

            info_dict[var_name]['type'] = type
            info_dict[var_name]['cond'] = cond

            var_name_bin = var_name + bin_suffix
            for df in datas:
                df[var_name_bin] = self.make_bin(df, var_name_new, cond, precision=precision)
            dti, woeDict = self.univerate(train, var_name_bin, target, retWoeDict=True)

            var_name_woe = var_name + woe_suffix
            for df in datas:
                df[var_name_woe] = self.woe_code(df, var_name_bin, woeDict)

            info_dict[var_name]['dti'] = dti
            info_dict[var_name]['woeDict'] = woeDict
            ks = dti.loc[0, 'KS']
            iv = dti.loc[0, 'IV']
            info_dict[var_name]['ks'] = ks
            info_dict[var_name]['iv'] = iv
        if path is not None:
            wb = Workbook()
            ws1 = wb.active
            ws1.title = '变量信息汇总'
            ws2 = wb.create_sheet(title='分箱信息')
            info = []
            dtis = []
            for v in info_dict.keys():
                info = info_dict[v]
                info.append(
                    [v, info['type'], info['missing'], info['missing_rate'], info['unique'], info['ks'], info['iv']])
                dtis.append(info_dict[v]['dti'])
            sheet1_data = pd.DataFrame(info,
                                       columns=['var_name', 'type', 'missing', 'missing_rate', 'unique', 'ks', 'iv'])
            sheet2_data = pd.concat(dtis, axis=1)
            sheet2_data['bin'] = sheet2_data['bin'].astype(str)
            for r in dataframe_to_rows(sheet1_data):
                ws1.append(r)
            for r in dataframe_to_rows(sheet2_data):
                ws2.append(r)
            wb.save(path)

        return [datas], info_dict

    def re_bin_woe(self, datas, var_name, target, cond, bin_suffix="_", woe_suffix="_woe"):
        """
        对相应的变量进行再分箱，并重新计算相应的woe,注意datas中的第一个数据集为用于计算woe的训练集
        """


class Discretize(metaclass=ABCMeta):
    """
    离散化基类，包含了基本的参数定义和特征预处理的方法
    注：分箱的预处理过程中就会剔除x变量中的缺失值，若要将缺失值也纳入分箱运算过程，请先在数据中进行填充
    Parameters
    ----------
    init_thredhold: int  初始化分箱的数量，若为空则钚进行初始化的分箱
    init_method : str 初始化方法默认为'qcut' , 初始化分箱方法，目前仅支持 'qcut' 和 'cut'
    print_process : bool 是否答应分箱的过程信息
    positive_label : 正样本的定义，根据target的数据类型来定
    negative_label : 负样本的定义，根据target的数据类型来定
    num_fillna ： 数字变量的缺失值填充，默认为None；若为None，在计算分箱的过程中会剔除缺失部分的数据
    cate_fillna ： 类别变量的缺失值填充，默认为None；若为None，在计算分箱的过程中会剔除缺失部分的数据
    """

    def __init__(self, init_thredhold=100, init_method='qcut', print_process=False,
                 positive_label=1, negative_label=0, num_fillna=None, cate_fillna=None):
        self.init_thredhold = init_thredhold
        self.init_method = init_method
        self.print_process = print_process
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.num_fillna = num_fillna
        self.cate_fillna = cate_fillna

    def _init_data(self, data, var_name, target, max_bin, precision):
        """
        init the input：
        1、校验自变量和因变量的类型
        2、判断x是否为数值类型
        3、初始化数据结果
        """
        if self.print_process: print("开始对变量'{}'使用'{}'分箱:".format(var_name, self.__class__.__name__))

        time0 = time.time()
        assert var_name in data.columns, "数据中不包含变量%s，请检查数据" % (var_name)
        assert var_name != target, "因变量和自变量必须是不同的变量"
        data = data[[var_name, target]].copy()

        self._y_check(data[data[target].notnull()][target])

        is_numeric = is_numeric_dtype(data[var_name])

        if is_numeric:
            if self.num_fillna is not None:
                data[var_name] = data[var_name].fillna(self.num_fillna)

            data = data[(data[var_name].notnull()) & (data[target].notnull())]
            if self.init_thredhold is not None:
                if self.init_method == 'qcut':
                    data.loc[:, var_name] = pd.qcut(data[var_name], self.init_thredhold, duplicates="drop",
                                                    precision=precision)
                elif self.init_method == 'cut':
                    data.loc[:, var_name] = pd.cut(data[var_name], self.init_thredhold, duplicates="drop",
                                                   precision=precision)
                else:
                    raise ValueError("init_method参数仅有qcut和cut两个选项")
                data.loc[:, var_name] = data[var_name].map(lambda x: x.right).astype(float)
            dti = pd.crosstab(data[var_name], data[target], dropna=False).reset_index()
            dti.rename({self.negative_label: "negative", self.positive_label: "positive"}, axis=1, inplace=True)
            dti["variable"] = dti[var_name]
            mapping = None
        else:
            if self.cate_fillna is not None:
                data[var_name] = data[var_name].fillna(self.cate_fillna)

            data = data[(data[var_name].notnull()) & (data[target].notnull())]
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

        while (len(dti) > max_bin) and (len(dti.query('(negative==0) or (positive==0)')) > 0):
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
        unique = y.unique()
        assert y_type in ['binary'], "目标必须是二分类"
        # assert not y.hasnans, "target中不能包含缺失值，请优先进行填充"
        assert self.positive_label in unique, "请根据target正确设定positive_label"
        assert self.negative_label in unique, "请根据target的结果正确设定negative_label"

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
    def dsct(self, df, var_name, target, max_bin=12, precision=4):
        """
        抽象接口，定义分箱方法的名称和入参，只能调用子类实现
        Parameters
        ----------
        dat: 数据集，格式必须为pd.DataFrame
        var_name:  待分箱的因变量
        target: 目标变量
        precision : 数值型变量的分箱进度
        """
        raise NotImplementedError("该方法只为定义基本的函数接口，不可直接调用，请使用子类实现的方法")


class ChiMerge(Discretize):
    """
    卡方分箱法
    """

    def dsct(self, df, var_name, target, max_bin=12, precision=4):
        dti, var_name, is_numeric, mapping = self._init_data(df, var_name, target, max_bin, precision)

        time0 = time.time()

        dti["chi2"] = dti.apply(lambda row: self._calc_chi2(dti, row), axis=1)
        while len(dti) > max_bin:
            min_chi2_ind = dti.query("chi2 == chi2.min()").index[0]
            if min_chi2_ind == dti.index.max():
                # 更新正负样本数量,将前一行的数据与此行的数据相加
                dti.loc[min_chi2_ind, "negative"] = dti.loc[min_chi2_ind, "negative"] + dti.loc[
                    min_chi2_ind - 1, "negative"]
                dti.loc[min_chi2_ind, "positive"] = dti.loc[min_chi2_ind, "positive"] + dti.loc[
                    min_chi2_ind - 1, "positive"]
                # 只需要更新当前行的chi2值
                a = dti.loc[min_chi2_ind, "negative"]
                b = dti.loc[min_chi2_ind, "positive"]
                c = dti.loc[min_chi2_ind - 2, "negative"]
                d = dti.loc[min_chi2_ind - 2, "positive"]
                dti.loc[min_chi2_ind, "chi2"] = self._chi2(a, b, c, d)
                # 删除前一行
                dti = dti.drop(index=min_chi2_ind - 1)
                dti.reset_index(drop=True, inplace=True)
            elif min_chi2_ind == dti.index.min() + 1:
                # 删除前一行前，需更新当前行的正负样本数量，以及当前行的chi2值
                dti.loc[min_chi2_ind, "negative"] = dti.loc[min_chi2_ind - 1, "negative"] + dti.loc[
                    min_chi2_ind, "negative"]
                dti.loc[min_chi2_ind, "positive"] = dti.loc[min_chi2_ind - 1, "positive"] + dti.loc[
                    min_chi2_ind, "positive"]
                dti.loc[min_chi2_ind, "chi2"] = np.inf
                # 更新后一行的chi2值
                a = dti.loc[min_chi2_ind, "negative"]
                b = dti.loc[min_chi2_ind, "positive"]
                c = dti.loc[min_chi2_ind + 1, "negative"]
                d = dti.loc[min_chi2_ind + 1, "positive"]
                dti.loc[min_chi2_ind + 1, "chi2"] = self._chi2(a, b, c, d)
                # 删除前一行
                dti = dti.drop(index=min_chi2_ind - 1)
                dti.reset_index(drop=True, inplace=True)
            else:
                # 删除前一行前，需更新当前行的正负样本数量，以及当前行的chi2值
                dti.loc[min_chi2_ind, "negative"] = dti.loc[min_chi2_ind - 1, "negative"] + dti.loc[
                    min_chi2_ind, "negative"]
                dti.loc[min_chi2_ind, "positive"] = dti.loc[min_chi2_ind - 1, "positive"] + dti.loc[
                    min_chi2_ind, "positive"]
                a = dti.loc[min_chi2_ind, "negative"]
                b = dti.loc[min_chi2_ind, "positive"]
                c = dti.loc[min_chi2_ind - 2, "negative"]
                d = dti.loc[min_chi2_ind - 2, "positive"]
                dti.loc[min_chi2_ind, "chi2"] = self._chi2(a, b, c, d)
                # 更新后一行的chi2值
                c = dti.loc[min_chi2_ind + 1, "negative"]
                d = dti.loc[min_chi2_ind + 1, "positive"]
                dti.loc[min_chi2_ind + 1, "chi2"] = self._chi2(a, b, c, d)
                # 删除前一行
                dti = dti.drop(index=min_chi2_ind - 1)
                dti.reset_index(drop=True, inplace=True)

            # dti.loc[min_chi2_ind - 1, "variable"] = dti.loc[min_chi2_ind, "variable"]
            # dti = dti.groupby("variable")[["negative", "positive"]].sum().reset_index()

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


class ChiMergeV0(Discretize):
    """
    卡方分箱法
    """

    def dsct(self, df, var_name, target, max_bin=12, precision=4):
        dti, var_name, is_numeric, mapping = self._init_data(df, var_name, target, max_bin, precision)

        time0 = time.time()

        while len(dti) > max_bin:
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


class BestKS(Discretize):
    """
    Best-KS 分箱法
    """

    def dsct(self, df, var_name, target, max_bin=12, precision=4):
        dti, var_name, is_numeric, mapping = self._init_data(df, var_name, target, max_bin, precision)
        time0 = time.time()

        dti["count"] = dti["negative"] + dti["positive"]
        dti["tmp"] = 0
        uni_tmp = pd.unique(dti["tmp"])
        while (len(uni_tmp) < max_bin) and (len(dti) > max_bin):
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

        if self.print_process:
            time1 = time.time()
            print("==>完成分箱迭代耗时{:.2f}s".format(time1 - time0))
        return self._normalize_output(dti, var_name, is_numeric, mapping)


class BestBin(Discretize):
    """
    考虑排序性的分箱法,基于传入分箱工具类，控制单调性的分箱法
    """

    def __init__(self, discretize, min_bin=3):
        self.discretize = discretize
        self.min_bin = min_bin
        self.kit = Kit()

    def check_posRate_monotone(self, dti):
        """
        校验分箱单调性
        """
        if dti.shape[0] <= 2:
            return True
        diff = dti["positive_rate"].diff()[1:]
        if len(diff[diff >= 0]) == len(diff) or len(diff[diff <= 0]) == len(diff):
            return True
        else:
            return False

    def dsct(self, df, var_name, target, max_bin=12, precision=4):
        """
        考虑单调性的分箱法，对原有的分箱方法进行风向
        """
        time0 = time.time()
        if self.discretize.print_process: print(
            "=========>>对变量'{}'启动'{}'的最优分箱".format(var_name, self.discretize.__class__.__name__))
        cond = self.discretize.dsct(df, var_name, target, max_bin=max_bin)
        var_name_new = var_name + "_bin"
        df[var_name_new] = self.kit.make_bin(df, var_name, cond)
        dti = self.kit.univerate(df, var_name_new, target)
        while (not self.check_posRate_monotone(dti)) and len(dti) > self.min_bin:
            max_bin = max_bin - 1
            cond = self.discretize.dsct(df, var_name, target, max_bin)
            df[var_name_new] = self.kit.make_bin(df, var_name, cond)
            dti = self.kit.univerate(df, var_name_new, target)
        time1 = time.time()
        if self.discretize.print_process: print("=========>>最优分箱完成，耗时{:.2f}s".format(time1 - time0))
        return cond