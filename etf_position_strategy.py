# ETF 持仓策略可视化
# 数据源: AKShare

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

from config import ETF_POOL, ETF_NAME_MAP, M_DAYS
from etf_utils import get_etf_close_akshare


def get_rank(
    etf_pool: list, start_date: datetime, end_date: datetime
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    计算 ETF 池在 [start_date, end_date] 区间内每个交易日的动量评分，并记录每日最优持仓 ETF。

    参数:
        etf_pool   - ETF 6位代码列表
        start_date - 统计起始日期
        end_date   - 统计截止日期
    返回:
        result_df    - 每日各 ETF 动量评分
        max_score_df - 每日最高评分 ETF 代码及分值
        close_df     - 每日各 ETF 收盘价（含 max_etf 列，在 main 中追加）
    """
    full_data: dict[str, pd.DataFrame] = {}
    for etf in etf_pool:
        full_data[etf] = get_etf_close_akshare(etf, end_date, start_date=start_date)

    trading_days = None
    for etf in etf_pool:
        days = full_data[etf][full_data[etf].index >= pd.Timestamp(start_date)].index
        trading_days = days if trading_days is None else trading_days.intersection(days)

    result_df = pd.DataFrame(columns=etf_pool, index=pd.DatetimeIndex([]))
    max_score_df = pd.DataFrame(columns=["max_etf", "max_score"], index=pd.DatetimeIndex([]))

    close_df = pd.DataFrame(index=trading_days)
    for etf in etf_pool:
        close_df[etf] = full_data[etf].reindex(trading_days)["close"]

    for current_date in trading_days:
        daily_scores: dict[str, float] = {}
        for etf in etf_pool:
            hist = full_data[etf][full_data[etf].index <= current_date].iloc[-M_DAYS:]
            if len(hist) < 2:
                continue
            y = np.log(hist["close"].values.astype(float))
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            annualized_returns = math.pow(math.exp(slope), 250) - 1
            r_squared = 1 - (
                sum((y - (slope * x + intercept)) ** 2)
                / ((len(y) - 1) * np.var(y, ddof=1))
            )
            score = annualized_returns * r_squared
            result_df.loc[current_date, etf] = score
            daily_scores[etf] = score

        if daily_scores:
            max_etf = max(daily_scores, key=daily_scores.get)
            max_score_df.loc[current_date, "max_etf"] = max_etf
            max_score_df.loc[current_date, "max_score"] = daily_scores[max_etf]

    result_df = result_df.sort_index()
    result_df = result_df.apply(pd.to_numeric, errors="coerce")
    max_score_df = max_score_df.sort_index()
    close_df = close_df.sort_index()
    return result_df, max_score_df, close_df


if __name__ == "__main__":
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    df, max_scores_df, close_df = get_rank(ETF_POOL, start_date, end_date)

    df.rename(columns=ETF_NAME_MAP, inplace=True)
    close_df.rename(columns=ETF_NAME_MAP, inplace=True)
    max_scores_df["max_etf"] = max_scores_df["max_etf"].apply(ETF_NAME_MAP.get)
    close_df["max_etf"] = max_scores_df["max_etf"]
    close_df = close_df.sort_index()

    etf_columns = [col for col in close_df.columns if col != "max_etf"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, etf in enumerate(etf_columns):
        ax = axes[idx]
        dates = mdates.date2num(close_df.index.to_pydatetime())
        x = dates
        y = close_df[etf].values.astype(float)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        seg_colors = [
            "red" if close_df.iloc[i]["max_etf"] == etf else "purple"
            for i in range(len(close_df) - 1)
        ]

        lc = LineCollection(segments, colors=seg_colors, linewidth=2)
        ax.add_collection(lc)
        ax.set_xlim(dates.min() - 1, dates.max() + 1)

        data_min, data_max = y[~np.isnan(y)].min(), y[~np.isnan(y)].max()
        margin = (data_max - data_min) * 0.05
        ax.set_ylim(data_min - margin, data_max + margin)

        date_range = close_df.index.max() - close_df.index.min()
        if date_range.days <= 90:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        for i, (date_val, row) in enumerate(close_df.iterrows()):
            color = "red" if row["max_etf"] == etf else "purple"
            ax.scatter(
                mdates.date2num(date_val), row[etf],
                color=color, s=30, zorder=5, alpha=0.7,
            )

        ax.set_title(f"{etf}", fontsize=12)
        ax.set_ylabel("收盘价")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'ETF持仓策略可视化 ({start_date.strftime("%Y-%m-%d")} 至 {end_date.strftime("%Y-%m-%d")})',
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.2)
    fig.text(
        0.5, 0.01,
        f"红色线段：当日持仓ETF | 紫色线段：非持仓ETF | 数据区间：{len(close_df)}个交易日",
        ha="center", fontsize=10, style="italic",
    )
    plt.show()
