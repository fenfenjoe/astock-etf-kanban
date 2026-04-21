# ETF 动量评分 - 计算当前分值并输出今日持仓建议
# 数据源: AKShare

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
    """
    score_list = []
    annualized_returns_list = []
    r_squared_list = []

    for etf in etf_pool:
        close = get_etf_close_akshare(etf, end_date=date.today(), count=M_DAYS)["close"]

        y = np.log(close.values.astype(float))
        x = np.arange(len(y))

        slope, intercept = np.polyfit(x, y, 1)

        annualized_returns = math.pow(math.exp(slope), 250) - 1

        r_squared = 1 - (
            sum((y - (slope * x + intercept)) ** 2)
            / ((len(y) - 1) * np.var(y, ddof=1))
        )

        score = annualized_returns * r_squared

        score_list.append(score)
        annualized_returns_list.append(annualized_returns)
        r_squared_list.append(r_squared)

    result_df = pd.DataFrame(
        index=etf_pool,
        data={
            "etf_name": list(etf_name_map.values()),
            "score": score_list,
            "annualized_returns": annualized_returns_list,
            "r_squared": r_squared_list,
        },
    )
    result_df = result_df.sort_values(by="score", ascending=False)
    rank_list = list(result_df.index)

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 100)
    print(result_df)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")

    return rank_list


if __name__ == "__main__":
    target_num = 1
    target_list = get_rank(ETF_POOL, ETF_NAME_MAP)[:target_num]
    # ETF_NAME_MAP 值格式为 "518880.黄金ETF"，取 '.' 后的简称显示
    display_name = ETF_NAME_MAP[target_list[0]].split(".", 1)[-1]
    print(
        "今天是：【{}】，应该持仓：【{}】".format(date.today(), display_name)
    )


