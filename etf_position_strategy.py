# ETF 持仓策略可视化
# 数据源: TickFlow（通过 etf_utils.py 中的 get_etf_close_akshare 函数获取数据）

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
        result_df    - 每日各 ETF 动量评分（DataFrame）
        max_score_df - 每日最高评分 ETF 代码及分值（DataFrame）
        close_df     - 每日各 ETF 收盘价（DataFrame）

    计算逻辑:
    ----------------------------------------
    该函数用于可视化持仓策略，需要：
    1. 计算每个交易日各 ETF 的动量评分
    2. 找出每日评分最高的 ETF（应持仓）
    3. 获取每日各 ETF 的收盘价

    动量评分计算原理（与 etf_score.py 一致）：
    1. 获取当前交易日之前 M_DAYS 日的收盘价
    2. 计算对数收益率序列
    3. 线性回归拟合，计算斜率 slope
    4. 年化收益率 = exp(slope)^250 - 1
    5. R 平方 = 1 - (残差平方和 / 总平方和)
    6. 动量评分 = 年化收益率 × R 平方
    """
    # 预先拉取全量数据（含前置窗口），减少循环内重复请求
    full_data: dict[str, pd.DataFrame] = {}
    for etf in etf_pool:
        # 获取该 ETF 从 start_date 到 end_date 的所有数据
        # 实际返回的数据会向前多取 M_DAYS*3 天，以保证滑动窗口计算不会越界
        full_data[etf] = get_etf_close_akshare(etf, end_date, start_date=start_date)

    # 取各 ETF 在 [start_date, end_date] 内都有数据的交易日
    # 这确保了所有 ETF 在比较时使用的是同一组交易日
    trading_days = None
    for etf in etf_pool:
        # 过滤出在 start_date 之后的数据
        days = full_data[etf][full_data[etf].index >= pd.Timestamp(start_date)].index
        # 取交集，确保所有 ETF 都有该日期的数据
        trading_days = days if trading_days is None else trading_days.intersection(days)

    # 创建空的 DataFrame 用于存储每日动量评分
    result_df = pd.DataFrame(columns=etf_pool, index=pd.DatetimeIndex([]))
    # 创建 DataFrame 用于存储每日最高评分 ETF
    max_score_df = pd.DataFrame(columns=["max_etf", "max_score"], index=pd.DatetimeIndex([]))

    # 创建 DataFrame 用于存储收盘价
    close_df = pd.DataFrame(index=trading_days)
    for etf in etf_pool:
        # 将该 ETF 的收盘价重新索引到交易日
        close_df[etf] = full_data[etf].reindex(trading_days)["close"]

    # 遍历每个交易日，计算该日各 ETF 的动量评分
    for current_date in trading_days:
        # 存储当日各 ETF 的评分
        daily_scores: dict[str, float] = {}

        for etf in etf_pool:
            # 获取该 ETF 在 current_date 之前（包括 current_date）的最近 M_DAYS 日数据
            hist = full_data[etf][full_data[etf].index <= current_date].iloc[-M_DAYS:]

            # 如果数据不足 2 条，跳过该日期
            if len(hist) < 2:
                continue

            # 转换为对数收益率
            y = np.log(hist["close"].values.astype(float))
            # 时间索引
            x = np.arange(len(y))

            # 线性回归拟合
            slope, intercept = np.polyfit(x, y, 1)

            # 计算年化收益率
            annualized_returns = math.pow(math.exp(slope), 250) - 1

            # 计算 R 平方
            r_squared = 1 - (
                sum((y - (slope * x + intercept)) ** 2)
                / ((len(y) - 1) * np.var(y, ddof=1))
            )

            # 计算动量评分
            score = annualized_returns * r_squared

            # 存储评分
            result_df.loc[current_date, etf] = score
            daily_scores[etf] = score

        # 找出当日评分最高的 ETF
        if daily_scores:
            # max(daily_scores, key=daily_scores.get) 找出评分最高的键
            max_etf = max(daily_scores, key=daily_scores.get)
            max_score_df.loc[current_date, "max_etf"] = max_etf
            max_score_df.loc[current_date, "max_score"] = daily_scores[max_etf]

    # 按日期排序
    result_df = result_df.sort_index()
    # 将字符串转换为数值类型
    result_df = result_df.apply(pd.to_numeric, errors="coerce")
    max_score_df = max_score_df.sort_index()
    close_df = close_df.sort_index()

    return result_df, max_score_df, close_df


if __name__ == "__main__":
    # 设置时间范围：最近一年
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    # 获取每日动量评分、最高评分 ETF 和收盘价数据
    df, max_scores_df, close_df = get_rank(ETF_POOL, start_date, end_date)

    # 将列名从 ETF 代码替换为 ETF 名称
    df.rename(columns=ETF_NAME_MAP, inplace=True)
    close_df.rename(columns=ETF_NAME_MAP, inplace=True)
    # 将最高评分 ETF 代码替换为名称
    max_scores_df["max_etf"] = max_scores_df["max_etf"].apply(ETF_NAME_MAP.get)

    # 将最高评分 ETF 名称添加到收盘价 DataFrame
    close_df["max_etf"] = max_scores_df["max_etf"]
    close_df = close_df.sort_index()

    # 获取 ETF 列名列表（排除 max_etf 列）
    etf_columns = [col for col in close_df.columns if col != "max_etf"]

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # 将子图展平，方便遍历
    axes = axes.flatten()

    # 遍历每个 ETF，绘制持仓策略图
    for idx, etf in enumerate(etf_columns):
        ax = axes[idx]

        # 将日期转换为 matplotlib 的数值格式
        dates = mdates.date2num(close_df.index.to_pydatetime())
        x = dates
        y = close_df[etf].values.astype(float)

        # 将数据点重塑为 LineCollection 需要的格式
        # points 形状：(N, 1, 2)，每两个相邻点构成一个线段
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        # 连接相邻点形成线段 segments 形状：(N-1, 2, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 为每个线段设置颜色：
        # - 红色：该日 max_etf 等于当前 ETF（应持仓）
        # - 紫色：该日 max_etf 不等于当前 ETF（不应持仓）
        seg_colors = [
            "red" if close_df.iloc[i]["max_etf"] == etf else "purple"
            for i in range(len(close_df) - 1)
        ]

        # 创建线段集合并添加到图表
        lc = LineCollection(segments, colors=seg_colors, linewidth=2)
        ax.add_collection(lc)

        # 设置 X 轴范围
        ax.set_xlim(dates.min() - 1, dates.max() + 1)

        # 设置 Y 轴范围（考虑数据边界和边距）
        data_min, data_max = y[~np.isnan(y)].min(), y[~np.isnan(y)].max()
        margin = (data_max - data_min) * 0.05
        ax.set_ylim(data_min - margin, data_max + margin)

        # 设置 X 轴日期格式
        date_range = close_df.index.max() - close_df.index.min()
        if date_range.days <= 90:
            # 少于90天，显示每月
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        else:
            # 大于90天，每2个月显示
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # 旋转 X 轴标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 绘制散点，为每个数据点着色
        for i, (date_val, row) in enumerate(close_df.iterrows()):
            color = "red" if row["max_etf"] == etf else "purple"
            ax.scatter(
                mdates.date2num(date_val), row[etf],
                color=color, s=30, zorder=5, alpha=0.7,
            )

        # 设置子图标题和标签
        ax.set_title(f"{etf}", fontsize=12)
        ax.set_ylabel("收盘价")
        ax.grid(True, alpha=0.3)

    # 设置总标题
    plt.suptitle(
        f'ETF持仓策略可视化 ({start_date.strftime("%Y-%m-%d")} 至 {end_date.strftime("%Y-%m-%d")})',
        fontsize=16,
        fontweight="bold",
    )

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.2)

    # 在图表底部添加图例说明
    fig.text(
        0.5, 0.01,
        f"红色线段：当日持仓ETF | 紫色线段：非持仓ETF | 数据区间：{len(close_df)}个交易日",
        ha="center", fontsize=10, style="italic",
    )

    # 显示图表
    plt.show()
