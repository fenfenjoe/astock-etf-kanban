# 共用工具函数 - ETF 数据获取
# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, date, timedelta
from tickflow import TickFlow
import sys

from config import M_DAYS

# 修复编码问题，重定向标准输出
sys.stdout.reconfigure(encoding='utf-8')

# 创建 TickFlow 实例（使用免费服务）
try:
    tf = TickFlow.free()
except UnicodeEncodeError:
    # 捕获编码异常，手动创建客户端
    from tickflow.client import TickFlowClient
    from tickflow.config import FreeConfig
    config = FreeConfig()
    tf = TickFlowClient(config=config)


def get_etf_close_akshare(
    etf_code: str,
    end_date: datetime | date,
    start_date: datetime | date | None = None,
    count: int | None = None,
) -> pd.DataFrame:
    """
    使用 TickFlow 获取 ETF 日线收盘价，支持两种调用方式：

    方式一 —— 按条数（用于计算当日评分）:
        get_etf_close_akshare(etf_code, end_date, count=25)
        直接获取最近 N 条数据

    方式二 —— 按日期区间（用于历史回溯）:
        get_etf_close_akshare(etf_code, end_date, start_date=start_date)
        获取指定日期范围内的数据（会额外向前多取数据以保证滑动窗口不越界）

    参数:
        etf_code   - 6位ETF代码，如 '518880'
        end_date   - 数据截止日期
        count      - 取最近 N 条（与 start_date 二选一）
        start_date - 数据起始日期（与 count 二选一）
    返回:
        以日期为索引、列名为 'close' 的 DataFrame
    """
    # 构建 TickFlow 标的代码（添加市场后缀）
    # 假设 ETF 都在上海或深圳交易所
    if etf_code.startswith('5'):
        symbol = f"{etf_code}.SH"
    else:
        symbol = f"{etf_code}.SZ"
    
    if count is not None:
        # 使用 count 参数获取数据
        # 注意：TickFlow 返回的是所有历史数据，我们取最后 count 条
        df = tf.klines.get(symbol, period="1d", count=count, as_dataframe=True)

        # 确保列名正确
        if "trade_date" in df.columns:
            df = df.rename(columns={"trade_date": "date"})
        if "close" not in df.columns and "Close" in df.columns:
            df = df.rename(columns={"Close": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # 返回整个 DataFrame，保持与原函数一致的接口
        return df[['close']]
    elif start_date is not None:
        # 按日期区间获取数据
        # 由于 TickFlow 免费版不支持按日期范围查询，使用 count 估算
        days_diff = (end_date - start_date).days + M_DAYS * 3
        df = tf.klines.get(symbol, period="1d", count=days_diff, as_dataframe=True)
        # 确保列名正确
        if "trade_date" in df.columns:
            df = df.rename(columns={"trade_date": "date"})
        if "close" not in df.columns and "Close" in df.columns:
            df = df.rename(columns={"Close": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # 过滤日期范围
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        # 返回整个 DataFrame，保持与原函数一致的接口
        return df[['close']]
    else:
        raise ValueError("start_date 和 count 必须提供其中一个")
