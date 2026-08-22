# 商机录入与分析助手

基于 Streamlit 的 FDE CRM Demo：销售记录经 LLM 提取、Python 规则校验、AI 自检与人工复核后，才会进入商机台账和管理层视图。

## 支持的输入与输出

- 输入：销售手工记录、会议纪要、会议逐字稿、录音转写文本及其他文本。
- 对逐字稿和录音转写文本，系统会先进行去噪和事实摘要，再提取 CRM 字段。
- 输出：客户名称、客户需求、核心场景、预算、决策人、影响人、时间计划、商机阶段、风险、下一步行动，以及未确认项、原文依据和规则执行报告。

## 规则边界

- 只有客户明确表达或原始记录可直接观察的信息才会写入事实字段。
- 销售个人推测不会写成事实；缺失信息明确标为“未确认”。
- 预算、决策人、时间计划和阶段均保留原始依据；前后矛盾的信息并列保留并要求人工复核。
- 阶段由题目给定的 S0–S5 条件和 Python 规则引擎最终计算；AI Draft 不能直接入库。
- 归档前必须进行人工复核；台账和管理层统计只使用人工确认结果。

## 本地启动

建议使用独立虚拟环境，并在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

运行回归检查：

```powershell
python test_all.py
```

## DeepSeek 配置

生产/演示密钥仅从以下位置读取，优先级从前到后：

1. Streamlit Community Cloud 的 Secrets：`DEEPSEEK_API_KEY`
2. 本地环境变量：`DEEPSEEK_API_KEY`
3. 当前浏览器会话中在侧栏输入的 Key（仅当前会话使用）

默认接口为 `https://api.deepseek.com`，默认模型为 `deepseek-v4-flash`。未配置有效 Key 时，关闭 Mock 模式不会发起真实调用，并会提示配置方式。

## 部署说明

1. 将 `app.py`、`database.py`、`models.py`、`requirements.txt` 和测试文件提交至仓库根目录。
2. 在 Streamlit Community Cloud 创建应用，入口选择 `app.py`；部署页的 Advanced settings 中选择与本地验证一致的 Python 版本。
3. 在应用 Secrets 中配置：

   ```toml
   DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
   ```

4. 部署后，用“完整商机”“信息不足”“矛盾信息”三类输入分别验证。

SQLite 适合本地单人 Demo 和现场验收。若需云端多人长期保存，须将数据层替换为持久化云数据库或外部数据服务；不要把本地 SQLite 文件视为云端长期存储方案。
