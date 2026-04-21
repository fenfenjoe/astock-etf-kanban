# A股ETF轮动看板

基于动量策略的 A 股 ETF 轮动分析工具，数据源使用 [AKShare](https://akshare.akfamily.xyz/)。

> 为什么选这 4 只 ETF 轮动，可看：https://www.joinquant.com/post/42673

---

## 项目结构

```
astock-etf-kanban/
├── config.py                 # 共用配置：ETF 池、名称映射、算法参数
├── etf_utils.py              # 共用工具：AKShare 数据获取函数
├── etf_score.py              # 脚本1：计算当前动量评分，输出今日持仓建议
├── etf_momentum_chart.py     # 脚本2：动量分值折线图 + 涨跌幅走势分析
├── etf_position_strategy.py  # 脚本3：持仓策略可视化（红/紫色线段）
├── pyproject.toml            # 项目依赖声明（uv 管理）
└── .python-version           # Python 版本锁定（3.11）
```

---

## 环境准备

项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖，首次使用需安装依赖：

```bash
# 安装 uv（如未安装）
pip install uv

# 安装项目依赖，自动创建 .venv 虚拟环境
uv sync
```

---

## 运行方式

### 脚本1 — 今日持仓建议（`etf_score.py`）

计算 ETF 池中每只 ETF 的近 25 日动量评分，打印排行表，并输出今天应持仓哪只 ETF。

```bash
uv run python etf_score.py
```

**示例输出：**

```
           etf_name     score  annualized_returns  r_squared
518880      黄金ETF     1.234               1.523      0.810
513100      纳指100     0.876               1.102      0.795
...
今天是：【2025-04-21】，应该持仓：【黄金ETF】
```

---

### 脚本2 — 动量分值 & 涨跌幅走势图（`etf_momentum_chart.py`）

绘制近一年内 4 只 ETF 的：

- **图1**：每日动量评分折线图（对比 4 只 ETF 的趋势强弱）
- **图2**：每只 ETF 的每日涨跌幅 + 累计涨跌幅（2×2 子图）

```bash
uv run python etf_momentum_chart.py
```

> ⚠️ 该脚本需逐日计算评分，数据量大，运行时间较长（约数分钟）。

---

### 脚本3 — 持仓策略可视化（`etf_position_strategy.py`）

绘制近一年内 4 只 ETF 的收盘价走势，并用颜色标记每日应持仓状态：

- 🔴 **红色线段**：该日动量评分最高，策略应持仓
- 🟣 **紫色线段**：该日非持仓

```bash
uv run python etf_position_strategy.py
```

> ⚠️ 同脚本2，逐日回溯计算，运行时间较长。

---

## 自定义配置

所有共用参数集中在 `config.py`，修改后三个脚本自动生效：

```python
# config.py

# 修改 ETF 池（6位纯数字代码）
ETF_POOL = ["518880", "513100", "159915", "510180"]

# 修改显示名称
ETF_NAME_MAP = {
    "518880": "518880.黄金ETF",
    # ...
}

# 修改动量计算窗口天数（默认25日）
M_DAYS = 25
```

---

## 依赖

| 包 | 用途 |
|----|------|
| `akshare` | A 股 ETF 历史行情数据 |
| `numpy` | 数值计算（线性拟合、对数收益率） |
| `pandas` | 数据处理与表格展示 |
| `matplotlib` | 图表绘制 |