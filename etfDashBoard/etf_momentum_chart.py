# ETF 动量分值折线图 + 涨跌幅走势分析
# 数据源: TickFlow（通过 etf_utils.py 中的 get_etf_close_akshare 函数获取数据）

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import mpld3

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

    计算逻辑:
    ----------------------------------------
    该函数用于绘制动量分值折线图，需要计算每个交易日各 ETF 的动量评分。

    动量评分计算原理（与 etf_score.py 一致）：
    1. 获取当前交易日之前 M_DAYS 日的收盘价
    2. 计算对数收益率序列
    3. 线性回归拟合，计算斜率 slope
    4. 年化收益率 = exp(slope)^250 - 1
    5. R 平方 = 1 - (残差平方和 / 总平方和)
    6. 动量评分 = 年化收益率 × R 平方
    """
    # 预先拉取全量数据（含前置窗口），减少循环内重复请求
    # 这样避免了在循环中每次都请求数据，提高效率
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
        days = full_data[etf][
            full_data[etf].index >= pd.Timestamp(start_date)
        ].index
        # 取交集，确保所有 ETF 都有该日期的数据
        trading_days = days if trading_days is None else trading_days.intersection(days)

    # 创建空的 DataFrame 用于存储每日动量评分
    result_df = pd.DataFrame(columns=etf_pool, index=pd.DatetimeIndex([]))

    # 遍历每个交易日，计算该日各 ETF 的动量评分
    for current_date in trading_days:
        for etf in etf_pool:
            # 获取该 ETF 在 current_date 之前（包括 current_date）的最近 M_DAYS 日数据
            # 使用 iloc[-M_DAYS:] 获取最后 M_DAYS 条记录
            hist = full_data[etf][full_data[etf].index <= current_date].iloc[-M_DAYS:]

            # 如果数据不足 2 条，跳过该日期（无法计算趋势）
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

            # 计算动量评分并存储
            result_df.loc[current_date, etf] = annualized_returns * r_squared

    # 按日期排序
    result_df = result_df.sort_index()
    # 将字符串转换为数值类型
    result_df = result_df.apply(pd.to_numeric, errors="coerce")
    return result_df


if __name__ == "__main__":
    # 设置时间范围：最近一年
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    # ===========================
    # 图1：动量分值折线图
    # ===========================
    # 计算各 ETF 在时间范围内的每日动量评分
    df = get_rank(ETF_POOL, start_date, end_date)

    # 将列名从 ETF 代码替换为 ETF 名称（方便图表显示）
    df.rename(columns=ETF_NAME_MAP, inplace=True)

    # 绘制折线图
    ax = df.plot(title="ETF动量分值", grid=True, figsize=(12, 6))
    # 设置 Y 轴范围
    ax.set_ylim(-2, 5)
    # 调整布局
    plt.tight_layout()

    # ===========================
    # 图2：各ETF涨跌幅走势（2x2子图）
    # ===========================
    # 创建空 DataFrame 存储每日收益率
    close_df = pd.DataFrame()

    # 计算每个 ETF 的每日涨跌幅
    for etf in ETF_POOL:
        # 获取历史数据
        hist = get_etf_close_akshare(etf, end_date, start_date=start_date)
        # 计算每日涨跌幅：(今日收盘价 / 昨日收盘价) - 1
        hist["daily_return"] = (hist["close"] / hist["close"].shift(1)) - 1
        # 添加到 DataFrame
        close_df[etf] = hist["daily_return"]

    # 将列名替换为 ETF 名称
    close_df.rename(columns=ETF_NAME_MAP, inplace=True)
    # 将索引标准化为日期（去除时间部分）
    close_df.index = pd.DatetimeIndex(close_df.index).normalize()

    # 设置图表样式
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # 各 ETF 的颜色
    line_styles = ["-", "--", "-.", ":"]  # 线条样式

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    # 设置总标题
    fig.suptitle("ETF 涨跌幅走势分析", fontsize=16, fontweight="bold")

    # 遍历每个 ETF，绘制涨跌幅图表
    for i, etf_col in enumerate(close_df.columns):
        # 计算子图位置
        row, col = i // 2, i % 2
        ax = axes[row, col]

        # 获取日期和涨跌幅数据
        dates = close_df.index
        returns = close_df[etf_col]

        # 计算累计涨跌幅：(1 + r1) * (1 + r2) * ... - 1
        cumulative_returns = (1 + returns).cumprod() - 1

        # 计算统计数据
        max_daily = returns.max()      # 最大单日涨幅
        min_daily = returns.min()      # 最大单日跌幅
        max_cumulative = cumulative_returns.max()  # 累计最高收益
        min_cumulative = cumulative_returns.min()  # 累计最低收益
        final_return = cumulative_returns.iloc[-1]  # 最终收益

        # 绘制每日涨跌幅折线
        ax.plot(dates, returns, label="每日涨跌幅", color=colors[i],
                linestyle=line_styles[0], linewidth=1.5, alpha=0.7)

        # 标记最大涨幅和最大跌幅点
        ax.plot(returns.idxmax(), max_daily, "^", markersize=8, color="red", label="日最大涨幅")
        ax.plot(returns.idxmin(), min_daily, "v", markersize=8, color="green", label="日最大跌幅")

        # 添加零线（参考线）
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # 创建双 Y 轴，用于显示累计涨跌幅
        ax2 = ax.twinx()
        # 绘制累计涨跌幅折线
        ax2.plot(dates, cumulative_returns, label="累计涨跌幅",
                 color="purple", linestyle=line_styles[1], linewidth=2.5, alpha=0.9)
        # 标记累计最高和最低点
        ax2.plot(cumulative_returns.idxmax(), max_cumulative, "o", markersize=8, color="blue", label="累计最高")
        ax2.plot(cumulative_returns.idxmin(), min_cumulative, "o", markersize=8, color="orange", label="累计最低")
        # 标记最终收益点
        ax2.plot(dates[-1], final_return, "s", markersize=8, color="black", label="最终收益")

        # 设置 Y 轴格式为百分比
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

        # 设置网格
        ax.grid(True, linestyle="--", alpha=0.7)

        # 设置子图标题和标签
        ax.set_title(f"{etf_col} 涨跌幅走势", fontsize=14)
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("每日涨跌幅", fontsize=12)
        ax2.set_ylabel("累计涨跌幅", fontsize=12)

        # 添加统计信息文本框
        stats_text = (
            f"最终收益: {final_return:.2%}\n"
            f"最大单日涨幅: {max_daily:.2%}\n"
            f"最大单日跌幅: {min_daily:.2%}"
        )
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.8), verticalalignment="top")

        # 设置 X 轴日期格式
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))

        # 合并图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="lower left",
                  framealpha=0.9, fontsize=9)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 输出为可交互 HTML
    output_path = "etf_momentum_chart.html"
    mpld3.save_html(plt.gcf(), output_path)
    print(f"图表已保存为 {output_path}，用浏览器打开即可交互查看")
