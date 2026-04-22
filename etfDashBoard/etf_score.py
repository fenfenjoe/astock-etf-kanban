# ETF 动量评分 - 计算当前分值并输出今日持仓建议
# 数据源: TickFlow

import math
import numpy as np
import pandas as pd
from datetime import date

from config import ETF_POOL, ETF_NAME_MAP, M_DAYS
from etf_utils import get_etf_close_akshare


def get_rank(etf_pool: list, etf_name_map: dict) -> list:
    """
    计算 ETF 池中每只 ETF 的动量评分，打印排行表，并返回按分值降序排列的代码列表。

    参数:
        etf_pool     - ETF 6位代码列表
        etf_name_map - {代码: 名称} 字典
    返回:
        rank_list - 按评分从高到低排列的 ETF 代码列表

    动量评分计算原理:
    ----------------------------------------
    1. 获取最近 M_DAYS 日的收盘价序列
    2. 计算对数收益率: y = ln(price)
    3. 对收益率序列进行线性回归: y = slope * x + intercept
       - slope > 0: 价格呈上升趋势
       - slope < 0: 价格呈下降趋势
    4. 计算年化收益率: annualized_returns = exp(slope)^250 - 1
       - 假设一年有250个交易日
       - 将日收益率转换为年化收益率
    5. 计算 R 平方（决定系数）: 衡量趋势的稳定性
       - R² 接近 1: 趋势稳定，线性拟合度高
       - R² 接近 0: 趋势不稳定，波动剧烈
    6. 最终动量评分: score = annualized_returns * r_squared
       - 只有当收益率高且趋势稳定时，评分才高
    """
    # 用于存储每只ETF的评分指标
    score_list = []               # 动量评分
    annualized_returns_list = []    # 年化收益率
    r_squared_list = []           # 决定系数（R²）

    # 遍历ETF池中的每只ETF
    for etf in etf_pool:
        # 获取最近 M_DAYS 日的收盘价数据
        # 返回 DataFrame，索引为日期，列为 'close'
        close = get_etf_close_akshare(etf, end_date=date.today(), count=M_DAYS)["close"]

        # 将收盘价转换为对数收益率
        # 对数收益率的优点：可以累加，便于计算复合收益
        y = np.log(close.values.astype(float))

        # 创建时间索引 (0, 1, 2, ..., M_DAYS-1)
        # x 代表时间，用于线性回归
        x = np.arange(len(y))

        # 线性回归拟合
        # np.polyfit(x, y, 1) 返回 [slope, intercept]
        # slope: 斜率，反映收益率的变化速度（每日收益率）
        # intercept: 截距
        slope, intercept = np.polyfit(x, y, 1)

        # 计算年化收益率
        # exp(slope) 是每日收益率的底数
        # 一年约有250个交易日，所以 pow(exp(slope), 250) 得到年化收益率底数
        # 减1得到年化收益率
        annualized_returns = math.pow(math.exp(slope), 250) - 1

        # 计算 R 平方（决定系数）
        # R² = 1 - (残差平方和 / 总平方和)
        # 残差平方和 = Σ(实际值 - 预测值)²
        # 总平方和 = Σ(实际值 - 均值)²
        # R² 越接近1，说明线性拟合越好，趋势越稳定
        r_squared = 1 - (
            sum((y - (slope * x + intercept)) ** 2)  # 残差平方和
            / ((len(y) - 1) * np.var(y, ddof=1))     # 总平方和（使用样本方差）
        )

        # 计算最终动量评分
        # 年化收益率衡量收益能力，R² 衡量趋势稳定性
        # 两者相乘得到综合评分，确保只有高收益且高稳定性的ETF才能获得高分
        score = annualized_returns * r_squared

        # 将结果添加到列表中
        score_list.append(score)
        annualized_returns_list.append(annualized_returns)
        r_squared_list.append(r_squared)

    # 创建结果 DataFrame，包含ETF名称和各评分指标
    result_df = pd.DataFrame(
        index=etf_pool,  # 使用ETF代码作为索引
        data={
            "etf_name": list(etf_name_map.values()),      # ETF名称
            "score": score_list,                            # 动量评分
            "annualized_returns": annualized_returns_list,  # 年化收益率
            "r_squared": r_squared_list,                    # 决定系数
        },
    )

    # 按动量评分降序排列
    result_df = result_df.sort_values(by="score", ascending=False)

    # 获取排序后的ETF代码列表
    rank_list = list(result_df.index)

    # 设置 pandas 显示选项，确保表格完整显示
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 100)

    # 打印结果表格
    print(result_df)

    # 重置显示选项
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")

    # 返回按评分降序排列的ETF代码列表
    return rank_list


if __name__ == "__main__":
    # 策略设置：持仓评分最高的ETF数量
    target_num = 1

    # 获取按评分降序排列的ETF代码列表
    target_list = get_rank(ETF_POOL, ETF_NAME_MAP)[:target_num]

    # 从配置中提取ETF简称
    # ETF_NAME_MAP 值格式为 "518880.黄金ETF"
    # 使用 split(".", 1)[-1] 取 "." 后的部分，得到 "黄金ETF"
    display_name = ETF_NAME_MAP[target_list[0]].split(".", 1)[-1]

    # 打印今日持仓建议
    print(
        "今天是：【{}】，应该持仓：【{}】".format(date.today(), display_name)
    )
