# 交互式投资组合优化与回测仪表盘

这是一个已经封装好的 Streamlit 项目，支持中文 / English 切换。

## 一键运行

双击：

```text
start_dashboard.bat
```

运行后会自动打开：

```text
http://localhost:8501
```

如果是第一次在一台新电脑上运行，脚本会自动调用 `setup_env.bat` 创建虚拟环境并安装依赖。

## 新手入门流程

1. 第一次打开时，先保持“使用样例数据”开启。
2. 阅读页面顶部的“新手快速上手”。
3. 在侧边栏选择“资产预设”，例如“全球 ETF 示例”。
4. 在“组合优化”页先看有效前沿、最大夏普组合和最小方差组合。
5. 再进入“回测”页，比较优化组合、等权组合和基准指数的长期表现。

页面左侧顶部可以切换 `中文 / English`，也可以关闭“显示新手引导”。

## 停止运行

双击：

```text
stop_dashboard.bat
```

如果你是在命令行窗口中手动启动的，也可以在那个窗口按 `Ctrl + C` 停止。

## 手动启动

```powershell
cd D:\portfolio_optimizer_dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 安装或更新依赖

```text
setup_env.bat
```

## 运行测试

```text
run_tests.bat
```

## 主要功能

- 自由输入股票 / ETF ticker。
- 支持中文 / English 双语切换。
- 内置新手快速上手、术语速查、资产预设和控件提示。
- 支持 AI 分析：内置 DeepSeek 预设，也可以添加自定义 OpenAI-compatible 厂商。
- 支持本地记忆：关闭后重新打开，仍可恢复上次的侧边栏设置。
- 支持样例数据，方便离线演示。
- 自动获取历史价格数据。
- 计算最大夏普组合、最小方差组合。
- 绘制有效前沿和资本市场线。
- 支持滚动窗口回测，对比等权组合和基准指数。
- 支持简化 Black-Litterman 投资者观点输入。

## 文件说明

- `app.py`：Streamlit 网页入口。
- `portfolio/`：优化、回测、数据获取、Black-Litterman 逻辑。
- `portfolio/ai_analysis.py`：AI 厂商配置、提示词构造和 SDK 调用。
- `tests/`：核心单元测试。
- `requirements.txt`：Python 依赖。
- `start_dashboard.bat`：一键启动。
- `stop_dashboard.bat`：停止运行。
- `setup_env.bat`：安装或更新运行环境。
- `run_tests.bat`：运行测试。

## 注意

本项目用于课程展示、研究和学习，不构成投资建议。真实投资组合管理还需要加入交易成本、税费、流动性约束、数据校验和合规要求。

## AI 分析使用方法

1. 在侧边栏找到“AI 分析设置”。
2. 选择 `DeepSeek` 或 `自定义`。
3. DeepSeek 默认使用 `https://api.deepseek.com` 和 `deepseek-chat`；自定义厂商需要填写 OpenAI-compatible Base URL 和模型名。
4. 填写 API Key。
5. 在页面上方的独立“AI 分析”板块点击“生成 AI 分析”。

API Key 只保存在当前页面会话中，不会写入项目文件。AI 输出仅用于学习和研究辅助，不构成投资建议。

“高级设置”中可以设置温度、最大输出 tokens，上限为 `100000`，也可以自定义 system prompt。默认 system prompt 会把 AI 设定为专业市场分析师和投资组合策略师，引导其尽可能全面地分析组合。

## 本地记忆

仪表盘会自动把侧边栏设置保存到项目目录下的 `.dashboard_settings.json`，下次启动时自动恢复。记忆内容包括资产预设、股票/ETF、模型假设、回测设置、AI 厂商设置，以及中文和英文各自独立的 system prompt。

API Key 默认不会保存。只有打开“记住 API Key / Remember API Key”后，才会把 API Key 写入本机设置文件。
