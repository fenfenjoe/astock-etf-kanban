# ETF 收盘价展示脚本 - 获取并展示ETF池内各ETF的历史收盘价
# 数据源: TickFlow

import pandas as pd
from datetime import date

from config import ETF_POOL, ETF_NAME_MAP, M_DAYS
from etf_utils import get_etf_close_akshare


def get_etf_prices_table(etf_pool: list, etf_name_map: dict, days: int) -> pd.DataFrame:
    """
    获取 ETF 池中每只 ETF 最近 N 日的收盘价，并以表格形式展示。

    参数:
        etf_pool   - ETF 6位代码列表
        etf_name_map - {代码: 名称} 字典
        days       - 获取最近 N 日的收盘价
    返回:
        price_table - 宽表格式 DataFrame，列为ETF代码，索引为日期

    表格格式说明:
    ----------------------------------------
    返回一个宽表格式的 DataFrame：
    - 行索引：日期（从最早到最近）
    - 列名：ETF代码（如 "518880"、"159915"）
    - 单元格值：该ETF在该日期的收盘价

    例如：
              518880   513100   159915   510180
    2026-03-25  5.123    1.234    2.345    3.456
    2026-03-26  5.234    1.345    2.456    3.567
    ...
    """
    # 存储每只ETF的收盘价Series，键为ETF代码
    price_dict = {}

    # 遍历ETF池中的每只ETF
    for etf in etf_pool:
        # 获取最近 days 日的收盘价数据
        # 返回 DataFrame，索引为日期，列为 'close'
        df = get_etf_close_akshare(etf, end_date=date.today(), count=days)

        # 提取收盘价Series，并重命名为ETF代码
        # 这样便于后续合并为宽表
        price_dict[etf] = df["close"]

    # 将所有ETF的收盘价合并为一个宽表 DataFrame
    # concat 函数将多个Series按索引（日期）合并
    # axis=1 表示按列合并，即不同的ETF成为不同的列
    price_table = pd.concat(price_dict, axis=1)

    # 按日期升序排列（默认就是升序，但确保一下）
    price_table = price_table.sort_index()

    # 返回收盘价表格
    return price_table


def print_price_table(price_table: pd.DataFrame, etf_name_map: dict):
    """
    打印收盘价表格，并在列名处显示ETF简称。

    参数:
        price_table  - 收盘价宽表 DataFrame
        etf_name_map - {代码: 名称} 字典，用于显示简称
    """
    # 设置 pandas 显示选项
    # max_rows: 最多显示100行
    # precision: 收盘价显示2位小数
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.precision", 2)
    pd.set_option("display.width", 1000)

    # 创建简称映射：ETF代码 -> 简称
    # ETF_NAME_MAP 值格式为 "518880.黄金ETF"
    # 取 "." 后的部分作为简称
    short_names = {code: name.split(".", 1)[-1] for code, name in etf_name_map.items()}

    # 复制表格，将列名从代码替换为简称
    display_table = price_table.copy()
    display_table.columns = [short_names.get(col, col) for col in display_table.columns]

    # 打印标题
    print("\n" + "=" * 80)
    print("ETF 收盘价表格（最近 {} 个交易日）".format(len(price_table)))
    print("=" * 80)

    # 打印表格
    print(display_table)

    # 打印统计信息
    print("\n" + "-" * 80)
    print("统计信息：")
    print("-" * 80)
    print(display_table.describe().round(4))

    # 重置显示选项
    pd.reset_option("display.max_rows")
    pd.reset_option("display.precision")
    pd.reset_option("display.width")


if __name__ == "__main__":
    # 获取并展示收盘价表格
    price_table = get_etf_prices_table(ETF_POOL, ETF_NAME_MAP, M_DAYS)

    # 打印表格
    print_price_table(price_table, ETF_NAME_MAP)
