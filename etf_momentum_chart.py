# ETF 动量分值折线图 + 涨跌幅走势分析
# 数据源: AKShare

import math
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from config import ETF_POOL, ETF_NAME_MAP, M_DAYS
from etf_utils import get_etf_close_akshare


def get_rank(etf_pool: list, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    计算 ETF 池在 [start_date, end_date] 区间内每个交易日的动量评分。

    参数:
        etf_pool   - ETF 6位代码列表
        start_date - 统计起始日期
        end_date   - 统计截止日期
    返回:
        result_df  - 索引为日期，列为各 ETF 代码，值为当日动量评分
    """
    # 预先拉取全量数据（含前置窗口），减少循环内重复请求
    full_data: dict[str, pd.DataFrame] = {}
    for etf in etf_pool:
        full_data[etf] = get_etf_close_akshare(etf, end_date, start_date=start_date)

    # 取各 ETF 在 [start_date, end_date] 内都有数据的交易日
    trading_days = None
    for etf in etf_pool:
        days = full_data[etf][
            full_data[etf].index >= pd.Timestamp(start_date)
        ].index
        trading_days = days if trading_days is None else trading_days.intersection(days)

    result_df = pd.DataFrame(columns=etf_pool, index=pd.DatetimeIndex([]))

    for current_date in trading_days:
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
            result_df.loc[current_date, etf] = annualized_returns * r_squared

    result_df = result_df.sort_index()
    result_df = result_df.apply(pd.to_numeric, errors="coerce")
    return result_df


if __name__ == "__main__":
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    # --- 图1：动量分值折线图 ---
    df = get_rank(ETF_POOL, start_date, end_date)
    df.rename(columns=ETF_NAME_MAP, inplace=True)

    ax = df.plot(title="ETF动量分值", grid=True, figsize=(12, 6))
    ax.set_ylim(-2, 5)
    plt.tight_layout()

    # --- 图2：各ETF涨跌幅走势（2x2子图）---
    close_df = pd.DataFrame()
    for etf in ETF_POOL:
        hist = get_etf_close_akshare(etf, end_date, start_date=start_date)
        hist["daily_return"] = (hist["close"] / hist["close"].shift(1)) - 1
        close_df[etf] = hist["daily_return"]

    close_df.rename(columns=ETF_NAME_MAP, inplace=True)
    close_df.index = pd.DatetimeIndex(close_df.index).normalize()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    line_styles = ["-", "--", "-.", ":"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    fig.suptitle("ETF 涨跌幅走势分析", fontsize=16, fontweight="bold")

    for i, etf_col in enumerate(close_df.columns):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        dates = close_df.index
        returns = close_df[etf_col]
        cumulative_returns = (1 + returns).cumprod() - 1

        max_daily = returns.max()
        min_daily = returns.min()
        max_cumulative = cumulative_returns.max()
        min_cumulative = cumulative_returns.min()
        final_return = cumulative_returns.iloc[-1]

        ax.plot(dates, returns, label="每日涨跌幅", color=colors[i],
                linestyle=line_styles[0], linewidth=1.5, alpha=0.7)
        ax.plot(returns.idxmax(), max_daily, "^", markersize=8, color="red", label="日最大涨幅")
        ax.plot(returns.idxmin(), min_daily, "v", markersize=8, color="green", label="日最大跌幅")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        ax2 = ax.twinx()
        ax2.plot(dates, cumulative_returns, label="累计涨跌幅",
                 color="purple", linestyle=line_styles[1], linewidth=2.5, alpha=0.9)
        ax2.plot(cumulative_returns.idxmax(), max_cumulative, "o", markersize=8, color="blue", label="累计最高")
        ax2.plot(cumulative_returns.idxmin(), min_cumulative, "o", markersize=8, color="orange", label="累计最低")
        ax2.plot(dates[-1], final_return, "s", markersize=8, color="black", label="最终收益")

        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_title(f"{etf_col} 涨跌幅走势", fontsize=14)
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("每日涨跌幅", fontsize=12)
        ax2.set_ylabel("累计涨跌幅", fontsize=12)

        stats_text = (
            f"最终收益: {final_return:.2%}\n"
            f"最大单日涨幅: {max_daily:.2%}\n"
            f"最大单日跌幅: {min_daily:.2%}"
        )
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.8), verticalalignment="top")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="lower left",
                  framealpha=0.9, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


