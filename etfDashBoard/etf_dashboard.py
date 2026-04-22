# ETF 每日轮动看板生成器
# -*- coding: utf-8 -*-
# 整合脚本1（评分排行）、脚本3（动量图表）、脚本4（持仓策略图）输出为单一 HTML

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.ticker import PercentFormatter
import mpld3
from datetime import datetime, date, timedelta

from config import ETF_POOL, ETF_NAME_MAP, M_DAYS
from etf_utils import get_etf_close_akshare

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def calc_score_data(etf_pool, etf_name_map):
    score_list, ann_list, r2_list = [], [], []
    today = date.today()
    for etf in etf_pool:
        close = get_etf_close_akshare(etf, end_date=today, count=M_DAYS)["close"]
        y = np.log(close.values.astype(float))
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        ann = math.pow(math.exp(slope), 250) - 1
        r2 = 1 - (sum((y - (slope * x + intercept)) ** 2) / ((len(y) - 1) * np.var(y, ddof=1)))
        score_list.append(ann * r2)
        ann_list.append(ann)
        r2_list.append(r2)
    result_df = pd.DataFrame(
        index=etf_pool,
        data={
            "etf_code": etf_pool,
            "etf_name": [etf_name_map[e].split(".", 1)[-1] for e in etf_pool],
            "score": score_list,
            "annualized_returns": ann_list,
            "r_squared": r2_list,
        },
    ).sort_values("score", ascending=False)
    top_name = result_df.iloc[0]["etf_name"]
    suggestion = f"今天是：【{today}】，应该持仓：【{top_name}】"
    return result_df, suggestion


def build_score_section(etf_pool, etf_name_map):
    print("[1/3] 计算今日动量评分...")
    df, suggestion = calc_score_data(etf_pool, etf_name_map)
    rows = ""
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        cls = ' class="top-row"' if rank == 1 else ""
        etf_display = f'{row["etf_name"]}({row["etf_code"]})'
        rows += f'<tr{cls}><td>{rank}</td><td>{etf_display}</td><td class="score-cell">{row["score"]:.4f}</td><td>{row["annualized_returns"]:.2%}</td><td>{row["r_squared"]:.4f}</td></tr>\n'
    return f"""
<div class="card">
  <h2>&#128203; 今日持仓建议</h2>
  <p class="suggestion">{suggestion}</p>
  <table>
    <thead><tr><th>排名</th><th>ETF</th><th class="score-header">动量评分</th><th>最近{M_DAYS}日年化收益率</th><th>R²</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def calc_momentum_rank(etf_pool, start_date, end_date):
    full_data = {}
    for etf in etf_pool:
        full_data[etf] = get_etf_close_akshare(etf, end_date, start_date=start_date)
    trading_days = None
    for etf in etf_pool:
        days = full_data[etf][full_data[etf].index >= pd.Timestamp(start_date)].index
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
            ann = math.pow(math.exp(slope), 250) - 1
            r2 = 1 - (sum((y - (slope * x + intercept)) ** 2) / ((len(y) - 1) * np.var(y, ddof=1)))
            result_df.loc[current_date, etf] = ann * r2
    result_df = result_df.sort_index().apply(pd.to_numeric, errors="coerce")
    return result_df, full_data


def build_momentum_chart_section(etf_pool, etf_name_map, start_date, end_date):
    print("[2/3] 绘制动量分值折线图 + 涨跌幅走势...")
    result_df, full_data = calc_momentum_rank(etf_pool, start_date, end_date)
    score_df = result_df.rename(columns=etf_name_map)
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for col in score_df.columns:
        ax1.plot(score_df.index, score_df[col], label=col)
    ax1.set_title("ETF 动量分值走势")
    ax1.set_ylim(-2, 5)
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()
    html_fig1 = mpld3.fig_to_html(fig1)
    plt.close(fig1)
    close_df = pd.DataFrame()
    for etf in etf_pool:
        hist = get_etf_close_akshare(etf, end_date, start_date=start_date)
        hist["daily_return"] = hist["close"] / hist["close"].shift(1) - 1
        close_df[etf] = hist["daily_return"]
    close_df.rename(columns=etf_name_map, inplace=True)
    close_df.index = pd.DatetimeIndex(close_df.index).normalize()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig2, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=100)
    fig2.suptitle("ETF 涨跌幅走势分析", fontsize=15, fontweight="bold")
    for i, etf_col in enumerate(close_df.columns):
        ax = axes[i // 2, i % 2]
        dates, returns = close_df.index, close_df[etf_col]
        cumret = (1 + returns).cumprod() - 1
        ax.plot(dates, returns, color=colors[i], linewidth=1.5, alpha=0.7, label="每日涨跌幅")
        ax.plot(returns.idxmax(), returns.max(), "^", markersize=7, color="red", label="日最大涨幅")
        ax.plot(returns.idxmin(), returns.min(), "v", markersize=7, color="green", label="日最大跌幅")
        ax.axhline(0, color="black", linewidth=1, alpha=0.5)
        ax2 = ax.twinx()
        ax2.plot(dates, cumret, color="purple", linestyle="--", linewidth=2, alpha=0.9, label="累计涨跌幅")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_title(f"{etf_col} 涨跌幅走势", fontsize=12)
        final = cumret.iloc[-1]
        ax.text(0.02, 0.95, f"最终收益: {final:.2%}\\n最大单日涨幅: {returns.max():.2%}\\n最大单日跌幅: {returns.min():.2%}", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.8), verticalalignment="top", fontsize=9)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))
        lines, labels = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(lines + l2, labels + lb2, loc="lower left", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    html_fig2 = mpld3.fig_to_html(fig2)
    plt.close(fig2)
    return f"""
<div class="card">
  <h2>&#128200; 动量分值折线图（近一年）</h2>
  <div class="chart-container">{html_fig1}</div>
</div>
<div class="card">
  <h2>&#128202; ETF 涨跌幅走势分析</h2>
  <div class="chart-container">{html_fig2}</div>
</div>"""


def calc_position_rank(etf_pool, start_date, end_date):
    full_data = {}
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
        daily_scores = {}
        for etf in etf_pool:
            hist = full_data[etf][full_data[etf].index <= current_date].iloc[-M_DAYS:]
            if len(hist) < 2:
                continue
            y = np.log(hist["close"].values.astype(float))
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            ann = math.pow(math.exp(slope), 250) - 1
            r2 = 1 - (sum((y - (slope * x + intercept)) ** 2) / ((len(y) - 1) * np.var(y, ddof=1)))
            score = ann * r2
            result_df.loc[current_date, etf] = score
            daily_scores[etf] = score
        if daily_scores:
            max_etf = max(daily_scores, key=daily_scores.get)
            max_score_df.loc[current_date, "max_etf"] = max_etf
            max_score_df.loc[current_date, "max_score"] = daily_scores[max_etf]
    result_df = result_df.sort_index().apply(pd.to_numeric, errors="coerce")
    max_score_df = max_score_df.sort_index()
    close_df = close_df.sort_index()
    return result_df, max_score_df, close_df


def build_position_strategy_section(etf_pool, etf_name_map, start_date, end_date):
    print("[3/3] 绘制持仓策略可视化...")
    df, max_scores_df, close_df = calc_position_rank(etf_pool, start_date, end_date)
    df.rename(columns=etf_name_map, inplace=True)
    close_df.rename(columns=etf_name_map, inplace=True)
    max_scores_df["max_etf"] = max_scores_df["max_etf"].apply(etf_name_map.get)
    close_df["max_etf"] = max_scores_df["max_etf"]
    close_df = close_df.sort_index()
    etf_columns = [col for col in close_df.columns if col != "max_etf"]
    # 改为 4行1列布局
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    for idx, etf in enumerate(etf_columns):
        ax = axes[idx]
        dates = mdates.date2num(close_df.index.to_pydatetime())
        x, y = dates, close_df[etf].values.astype(float)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = ["red" if close_df.iloc[i]["max_etf"] == etf else "purple" for i in range(len(close_df) - 1)]
        lc = LineCollection(segments, colors=seg_colors, linewidth=2)
        ax.add_collection(lc)
        ax.set_xlim(dates.min() - 1, dates.max() + 1)
        data_min, data_max = y[~np.isnan(y)].min(), y[~np.isnan(y)].max()
        margin = (data_max - data_min) * 0.05
        ax.set_ylim(data_min - margin, data_max + margin)
        # X轴以月为单位显示
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        for i, (date_val, row) in enumerate(close_df.iterrows()):
            color = "red" if row["max_etf"] == etf else "purple"
            ax.scatter(mdates.date2num(date_val), row[etf], color=color, s=30, zorder=5, alpha=0.7)
        ax.set_title(f"{etf}", fontsize=12, fontweight="bold")
        ax.set_ylabel("收盘价")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'ETF持仓策略可视化 ({start_date.strftime("%Y-%m-%d")} 至 {end_date.strftime("%Y-%m-%d")})', fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.97, hspace=0.35)
    fig.text(0.5, 0.01, f"红色线段：当日持仓ETF | 紫色线段：非持仓ETF | 数据区间：{len(close_df)}个交易日", ha="center", fontsize=10, style="italic")
    html_fig = mpld3.fig_to_html(fig)
    plt.close(fig)
    return f"""
<div class="card">
  <h2>&#127919; 持仓策略可视化（近一年）</h2>
  <div class="chart-container">{html_fig}</div>
</div>"""


def render_dashboard():
    today_str = date.today().strftime("%Y年%m月%d日")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    score_html = build_score_section(ETF_POOL, ETF_NAME_MAP)
    momentum_html = build_momentum_chart_section(ETF_POOL, ETF_NAME_MAP, start_date, end_date)
    position_html = build_position_strategy_section(ETF_POOL, ETF_NAME_MAP, start_date, end_date)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ETF 每日轮动看板 - {today_str}</title>
  <style>
    body {{ font-family: "Microsoft YaHei", sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 0; padding: 20px; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ text-align: center; color: white; font-size: 36px; margin-bottom: 10px; }}
    .subtitle {{ text-align: center; color: #f0f0f0; font-size: 16px; margin-bottom: 30px; }}
    .card {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
    .card h2 {{ color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; margin-top: 0; }}
    .suggestion {{ font-size: 20px; font-weight: bold; color: #d63031; text-align: center; padding: 15px; background: #fff3cd; border-radius: 8px; margin: 15px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
    th {{ background: #667eea; color: white; font-weight: bold; }}
    tr:hover {{ background: #f5f5f5; }}
    .top-row {{ background: #fff3cd; font-weight: bold; }}
    /* 动量评分红色样式 */
    .score-header {{ color: #ff0000 !important; }}
    .score-cell {{ color: #ff0000; font-weight: bold; }}
    /* 响应式图表容器 */
    .chart-container {{ width: 100%; overflow: hidden; position: relative; }}
    .chart-container svg {{ width: 100% !important; height: auto !important; display: block; }}
    .chart-container > div {{ width: 100% !important; }}
    /* 响应式布局 */
    @media (max-width: 768px) {{
      body {{ padding: 10px; }}
      h1 {{ font-size: 24px; }}
      .card {{ padding: 15px; }}
      .suggestion {{ font-size: 16px; padding: 10px; }}
      th, td {{ padding: 8px; font-size: 14px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>&#128202; ETF 每日轮动看板</h1>
    <p class="subtitle">生成时间：{today_str}</p>
    {score_html}
    {momentum_html}
    {position_html}
  </div>
</body>
</html>"""
    output_path = "etf_dashboard.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\\n✅ 看板生成完成：{output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("ETF 每日轮动看板生成器")
    print("=" * 60)
    render_dashboard()

    render_dashboard()
