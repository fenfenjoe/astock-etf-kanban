# 共用工具函数 - ETF 数据获取
import pandas as pd
import akshare as ak
from datetime import datetime, date, timedelta

from config import M_DAYS


def get_etf_close_akshare(
    etf_code: str,
    end_date: datetime | date,
    start_date: datetime | date | None = None,
    count: int | None = None,
) -> pd.DataFrame:
    """
    使用 AKShare 获取 ETF 日线收盘价，支持两种调用方式：

    方式一 —— 按日期区间（用于历史回溯）:
        get_etf_close_akshare(etf_code, end_date, start_date=start_date)
        会额外向前预取 M_DAYS*3 天数据，以保证滑动窗口不越界。

    方式二 —— 按条数（用于计算当日评分）:
        get_etf_close_akshare(etf_code, end_date, count=25)

    参数:
        etf_code   - 6位ETF代码，如 '518880'
        end_date   - 数据截止日期
        start_date - 数据起始日期（与 count 二选一）
        count      - 取最近 N 条（与 start_date 二选一）
    返回:
        以日期为索引、列名为 'close' 的 DataFrame
    """
    if count is not None:
        fetch_start = end_date - timedelta(days=count * 3)
    elif start_date is not None:
        fetch_start = start_date - timedelta(days=M_DAYS * 3)
    else:
        raise ValueError("start_date 和 count 必须提供其中一个")

    if isinstance(fetch_start, datetime):
        fetch_start_str = fetch_start.strftime("%Y%m%d")
    else:
        fetch_start_str = fetch_start.strftime("%Y%m%d")

    if isinstance(end_date, datetime):
        end_date_str = end_date.strftime("%Y%m%d")
    else:
        end_date_str = end_date.strftime("%Y%m%d")

    df = ak.fund_etf_hist_em(
        symbol=etf_code,
        period="daily",
        start_date=fetch_start_str,
        end_date=end_date_str,
        adjust="qfq",
    )
    df = df.rename(columns={"日期": "date", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()[["close"]]

    if count is not None:
        return df.iloc[-count:]
    return df
