"""
商机录入与分析助手 —— FDE 面试演示 Demo（产品级重构）
架构: Extractor (LLM) → Rule Engine (Python) → Critic (LLM) → Human Confirm → SQLite → Dashboard
=============================================
核心原则：
  AI Result = Draft / Suggestion
  Human Confirmed Result = System of Record
AI 不是最终事实来源。只有经过用户确认后的数据，才属于正式业务数据。
"""

import os
import re
import json
import uuid
import time
from datetime import datetime, timedelta
import streamlit as st

# ── Import our modules ────────────────────────────────────────────────
import database as db
from models import (
    OPPORTUNITY_FIELDS,
    CRITICAL_FIELDS,
    VALID_STAGE_LIST,
    VALID_STAGES,
    STAGE_DESC,
    STAGE_RULES,
    extract_value,
    extract_quote,
    clean_field_entry,
    CHANGE_AI_CONFIRMED,
    CHANGE_HUMAN_CORRECTED,
    CHANGE_HUMAN_ADDED,
    CHANGE_MISSING,
    infer_change_types,
)

# ──────────────────────────────────────────────
# DeepSeek (OpenAI 兼容接口) 配置
# ──────────────────────────────────────────────
BASE_URL = "https://api.deepseek.com"                 # DeepSeek API
MODEL_NAME = "deepseek-v4-flash"
LLM_TIMEOUT_SECONDS = 30.0
LLM_MAX_RETRIES = 1

# Mock 开关：True 时使用预置返回，False 时调真实 LLM
USE_MOCK = False

# Input type options
INPUT_TYPES = [
    "销售手工记录",
    "会议纪要",
    "会议逐字稿",
    "录音转写文本",
    "其他文本",
]


def _read_streamlit_secret(name: str) -> str:
    """Read one optional Streamlit secret without breaking local execution."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        # Local development and unit tests may have no secrets.toml at all.
        return ""
    return str(value or "").strip()


def _get_managed_api_key() -> str:
    """Return a server-managed key, if the deployment provides one.

    A managed key must never be paired with visitor-configurable endpoint or
    model settings. Otherwise, a public-app visitor could redirect the request
    to a server they control and receive the Authorization header.
    """
    for key in (
        _read_streamlit_secret("DEEPSEEK_API_KEY"),
        os.environ.get("DEEPSEEK_API_KEY", "").strip(),
        os.environ.get("DASHSCOPE_API_KEY", "").strip(),
    ):
        if key:
            return key
    return ""


def _get_api_key() -> str:
    """Return the deployment key, or a key supplied only for this session."""
    return _get_managed_api_key() or str(st.session_state.get("_api_key", "")).strip()


# ──────────────────────────────────────────────
# 预设测试用例
# ──────────────────────────────────────────────
PRESET_CASES = {
    "✅ 完整信息 — 优质商机": (
        "今天拜访了XX医疗的王总（副院长），他们正在规划全院级别的CDSS系统升级项目。"
        "王总明确说今年Q3要做选型，预算大概在200万左右，已经批了立项申请。\n\n"
        "除了王总，信息科的张主任和临床的李部长也很关注这个项目。张主任主要负责技术评估，"
        "李部长是未来系统的最终使用部门代表。我们已经跟张主任做了两轮技术演示，"
        "对方反馈不错。\n\n"
        "下一步：下周三安排一次给王总的方案汇报，由我司售前经理陈明参加。"
    ),
    "⚠️ 预算缺失 — 需追问": (
        "客户是某制造业企业，最近有MES系统改造的需求。生产部赵经理对我们产品感兴趣，"
        "说之前用过竞品A，但满意度不高。\n\n"
        "客户希望在年底前上线一期模块，先覆盖两条产线。时间比较紧，需要尽快推进。\n\n"
        "下一步：本周五带解决方案去客户现场做需求调研。"
    ),
    "⚠️ 人名缺失 — 联系人不明确": (
        "通过展会认识了一家零售连锁客户，他们对我们的会员营销平台有兴趣。"
        "对方说最近正好在找供应商做数字化升级。\n\n"
        "预计决策周期2个月左右，预算方面说是'可能有一百多万的盘子'。"
        "感觉对方挺着急的，希望我们尽快出方案。\n\n"
        "下一步：等对方发一下具体需求文档。"
    ),
    "❌ 信息不足 — 建议人工复核": (
        "聊了聊，客户好像有些需求。\n\n"
        "下次再跟进吧。"
    ),
    "❌ 矛盾信息 — 前后不一致": (
        "和张总确认了项目意向，他明确表态Q2一定会上线，预算500万没问题，已经走完内部审批。\n\n"
        "但从信息科侧面了解，目前连立项都没提交，预算还在写报告中，实际最早Q3才能开始可行性论证。\n\n"
        "张总个人很看好我们，承诺会全力推动。"
    ),
}

# ──────────────────────────────────────────────
# Prompt 模板（保留原有设计）
# ──────────────────────────────────────────────

EXTRACTOR_PROMPT = """你是一位严格遵守 CRM 商机规则的销售分析助手。请将以下销售拜访记录结构化提取为商机信息。

【字段定义】
| 字段 | 说明 |
|------|------|
| customer_name | 客户公司名称 |
| need | 客户需求描述 |
| scenario | 核心应用场景 |
| budget | 预算金额及来源状态 |
| decision_maker | 决策人（有拍板权的人） |
| influencer | 影响人（能影响决策的人） |
| timeline | 时间计划（关键里程碑与日期） |
| stage | 商机阶段（S0-S5） |
| risk | 风险与障碍 |
| next_step | 下一步行动计划 |

【必须遵守的业务规则】
1. **事实边界**：仅将客户明确表达或原始记录可直接观察的信息写入事实字段。每一个非空字段必须附上原始记录中的逐字 quote；quote 不得改写、概括或来自你的推测。
2. **推断处理**："可能、应该、感觉、挺感兴趣、我猜、我估计"等不确定表达不得写入预算、需求、时间或角色等事实字段。客户明确陈述的金额或时间范围中，“约/大概/左右”必须原样保留，并在字段值中标注“精确值待确认”（如“约80万，已批（金额精确值待确认）”）；不得把范围金额写成精确、已确认金额。
3. **缺失信息**：金额、姓名、时间、权限没有明确依据时，value 与 quote 均填 null，绝不补全。
4. **角色边界**：decision_maker 仅限有最终拍板权的角色；普通联系人、技术评估人或预算沟通人只能作为 influencer 或未确认信息。
5. **矛盾处理**：记录存在前后矛盾时，不能选择任意一方作为事实；在 _conflicts 并列列出互相冲突的原文依据，相关事实字段填 null 或标为待确认。_conflicts.fields 只能列出被冲突影响的事实字段（如 budget、timeline、stage），不能列 risk；risk 是承载冲突说明的结果字段。
6. **风险边界**：risk 只记录原文明确的风险、异议、阻碍或矛盾；不得仅因“约/大概/左右”这样的范围限定词就写“预算不确定”，也不得把已有最终拍板依据的人标为“决策人未确认”。
7. **下一步行动**：next_step 可以写两类原文已明确的行动：（a）客户共同约定的行动；（b）销售明确表述的后续跟进行动。不得把销售计划伪装成客户承诺；_next_step_plan 中拆出动作、负责人和时间。负责人/时间缺失时填 null，后续由系统标为“建议负责人”或“待确认”。

【阶段证据：只提取证据，不自由判定】
stage 最终由 Python 规则引擎计算。请输出 _stage_evidence 中的五项布尔证据；若为 true，quote 必须逐字来自原文：
- need_or_scenario：明确至少一个业务问题或使用场景。
- solution_validation：客户明确同意演示、试用、技术交流或方案评估。
- commercial_evaluation：已讨论预算、报价、采购流程或合同条款之一，且需求仍有效。
- decision_approval：明确进入内部立项、审批或供应商决策。
- contract_signed：已签合同或正式订单已确认。

stage 仍需输出一个候选 S0-S5 和 quote，但最终以系统规则计算为准。仅“正在选型/看了多家/将发送方案”不能单独构成 S2；“对方案感兴趣、夸界面、觉得不错、这正是想要的”等正向反馈本身不等于明确业务问题或场景，不能单独构成 S1；销售个人猜测的金额不能构成 S3；董事会仅审批预算不能自动推定最终决策人。

【输出要求】
严格输出 JSON，不要包含任何其他文字。除 10 个 CRM 字段外，必须包含 _stage_evidence、_conflicts、_next_step_plan：

格式示例：
```json
{{
  "customer_name": {{ "value": "...", "quote": "..." }},
  "need": {{ "value": "...", "quote": "..." }},
  "stage": {{ "value": "S2", "quote": "..." }},
  "_stage_evidence": {{
    "need_or_scenario": {{ "met": true, "quote": "..." }},
    "solution_validation": {{ "met": false, "quote": null }},
    "commercial_evaluation": {{ "met": false, "quote": null }},
    "decision_approval": {{ "met": false, "quote": null }},
    "contract_signed": {{ "met": false, "quote": null }}
  }},
  "_conflicts": [{{"fields": ["budget", "timeline"], "quotes": ["...", "..."], "description": "..."}}],
  "_next_step_plan": {{"action": "...", "owner": null, "time": null}}
}}
```

【原始记录】
{transcript}
"""

TRANSCRIPT_INPUT_TYPES = frozenset({"录音转写文本", "会议逐字稿"})

TRANSCRIPT_PREPROCESS_SYSTEM_PROMPT = """你是 B2B 销售会议纪要与事实梳理助手。
你的输出将被另一个模型用于提取 CRM 商机字段。只整理原文中已有的事实，绝不补充或猜测。"""

TRANSCRIPT_PREPROCESS_PROMPT = """请将以下{input_type}预处理为供 CRM 提取使用的 Markdown 事实摘要。

要求：
1. 删除寒暄、口头禅、重复表达和无业务含义的内容；保留所有业务事实、人员角色、金额、时间、承诺及原话中的不确定性。
2. 修正明显的同音错字和英文/产品名称大小写，但不得改变业务含义。
3. 必须按以下标题输出；没有依据的内容写“未提及”，不能编造：
   - 参与人及角色
   - 客户现状与痛点
   - 需求与核心场景
   - 预算与立项状态
   - 决策与影响角色
   - 时间计划与下一步
   - 风险、异议与矛盾信息
   - 可追溯原文摘录
4. 对存在冲突的说法并列呈现，明确标注“待核实”；不得自行选择其中一方。
5. “可追溯原文摘录”使用原文短句，供下游字段的 quote 引用。输出 Markdown 正文，不要解释你的处理过程。

【原始转写】
{raw_text}
"""

CRITIC_PROMPT = """你是一位严格的商机质量审查专家。请审核以下结构化商机数据，检查是否存在以下问题：

【审查规则】
1. **前后矛盾**：不同字段的陈述是否互相冲突（例如预算已批准 vs 立项未提交）
2. **不确定表述误判为确定**：将"可能/感觉/好像/大概"等不确定词当作事实陈述
3. **信息不足**：关键字段缺失数量 ≥ 3 个（customer_name, need, decision_maker, stage）时，需要列明真实缺失字段；若系统已明确标为“未确认”，这属于需人工补充的 REVIEW，而不是 AI 失败。
4. **阶段与角色越级**：阶段必须符合固定达成条件：S1 有明确问题/场景；S2 有明确同意演示、试用、技术交流或方案评估；S3 已讨论预算、报价、采购流程或合同条款且需求有效；S4 已进入立项、审批或供应商决策；S5 已签合同或订单确认。决策人必须有最终拍板权证据，不能把联系人直接当作决策人。
5. **销售猜测被事实化**：销售人员的“我猜/估计/应该”等未经客户确认的表达不得作为预算、时间或需求的字段值。
6. **范围金额的阶段含义**：客户明确讨论“约/大概/左右/区间”的预算时，金额精确值可待确认，但该范围预算仍是已讨论预算的证据；若需求仍有效，可支持 S3，不能因此判为阶段错误。

【输出要求】
严格输出 JSON，不要包含任何其他文字：
```json
{{
  "overall_status": "PASS | REVIEW | FAIL",
  "issues": [
    {{
      "type": "contradiction | uncertain_falsified | insufficient_info",
      "severity": "high | medium | low",
      "description": "具体问题描述",
      "affected_fields": ["field_names"]
    }}
  ],
  "missing_critical_fields": ["field_names"],
  "summary": "一句话总结"
}}
```

status 判断逻辑：
- PASS：无问题或仅低危问题
- REVIEW：存在矛盾、待确认事项或部分信息不完整；矛盾应提示人工复核，不应因为系统正确识别了矛盾而把 Draft 本身判为失败。
- FAIL：模型把无依据的内容写成确定事实、忽略已识别的高危矛盾，或关键字段严重不足且未明确标为未确认。

【商机数据】
{extracted_json}
"""

NEXTPROMPT_PROMPT = """你是一位顶级销售教练。根据以下商机信息的缺失项，生成一段建议销售人员在下次沟通时追问的话术。

【要求】
1. 语气自然，像老销售教新人的口吻
2. 针对每一个缺失的关键信息字段，写一个具体的追问话术
3. 总共不超过8句话，控制在200字以内
4. 直接输出话术内容，不需要标题或其他包装

【商机数据】
{extracted_json}

【审查结果】
{critic_result}
"""


# ──────────────────────────────────────────────
# JSON 清洗函数 — 处理 DeepSeek 思考标签与 Markdown 包裹
# ──────────────────────────────────────────────

def clean_and_parse_llm_json(raw_text: str):
    """
    清洗 LLM 返回的原始文本并尝试解析为 JSON。
    - 移除 <think>...</think> 标签
    - 提取 markdown code block 中的内容
    - 截取第一个 { 到最后一个 } 之间的子串
    - 返回 json.loads 的结果，解析失败则返回 None
    """
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1)
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx : end_idx + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────────
# LLM 调用（支持 Mock + Error Handling）
# ──────────────────────────────────────────────

def _call_llm_real(system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
    """Call DeepSeek safely, optionally enforcing a JSON response body."""
    api_key = _get_api_key()
    if not api_key:
        return (
            "[LLM调用异常] MissingAPIKey: 未配置 DeepSeek API Key。"
            "请在 Streamlit Secrets 或环境变量中设置 DEEPSEEK_API_KEY，"
            "或在当前会话侧栏输入 Key。"
        )
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            # Never accept an endpoint from a browser session. The OpenAI client
            # sends the API key as an Authorization header to this base URL.
            base_url=BASE_URL,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )

        request_options = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        if json_mode:
            # DeepSeek's JSON Output mode guarantees a JSON string for the
            # final message. Disabling thinking also avoids a long structured
            # extraction being cut off by reasoning before the JSON is emitted.
            request_options["response_format"] = {"type": "json_object"}
            request_options["extra_body"] = {"thinking": {"type": "disabled"}}

        resp = client.chat.completions.create(
            **request_options,
        )
        if not resp.choices:
            return "[LLM调用异常] EmptyResponse: API 未返回候选结果"
        content = resp.choices[0].message.content
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        if not content or not str(content).strip():
            return f"[LLM调用异常] EmptyResponse: API 返回空内容（finish_reason={finish_reason}）"
        if finish_reason == "length":
            return "[LLM调用异常] TruncatedResponse: JSON 输出超过长度限制，请缩短输入后重试"
        return str(content)
    except Exception as e:
        return f"[LLM调用异常] {type(e).__name__}: {e}"


def call_transcript_preprocessor(
    raw_text: str, input_type: str, use_mock: bool = False
) -> tuple[str, str | None]:
    """Denoise a transcript before field extraction, retaining the raw source separately."""
    if use_mock:
        return _mock_transcript_summary(raw_text), None

    summary = _call_llm_real(
        TRANSCRIPT_PREPROCESS_SYSTEM_PROMPT,
        TRANSCRIPT_PREPROCESS_PROMPT.format(input_type=input_type, raw_text=raw_text),
    ).strip()
    if summary.startswith("[LLM调用异常]"):
        return raw_text, summary
    return summary, None


def call_extractor(
    transcript: str, use_mock: bool = False, preprocessed_summary: str | None = None
) -> dict:
    """步骤一：Extractor — 调用大模型提取商机字段"""
    if use_mock:
        return _mock_extractor(transcript)
    source_text = transcript
    if preprocessed_summary:
        source_text = (
            "【预处理事实摘要（仅用于定位字段，不得直接作为 quote）】\n"
            f"{preprocessed_summary}\n\n"
            "【原始转写（唯一事实与 quote 来源；所有非空字段必须逐字引用这里）】\n"
            f"{transcript}"
        )
    result_text = _call_llm_real(
        "", EXTRACTOR_PROMPT.format(transcript=source_text), json_mode=True
    )

    if result_text.startswith("[LLM调用异常]"):
        return {"_raw_response": result_text, "_api_error": True}

    # Use cleaning function to parse JSON
    parsed = clean_and_parse_llm_json(result_text)
    if not isinstance(parsed, dict):
        return {
            "_raw_response": result_text,
            "_parse_error": True,
            "_llm_error": "LLM 返回的 JSON 必须是对象，不能是数组、字符串或标量。",
        }

    # Normalize field entries so every canonical field exists
    normalized = {}
    for f in OPPORTUNITY_FIELDS:
        entry = parsed.get(f)
        if isinstance(entry, dict):
            normalized[f] = clean_field_entry(entry)
        elif entry is not None:
            normalized[f] = {"value": str(entry), "quote": None}
        else:
            normalized[f] = {"value": None, "quote": None}

    # These are non-CRM analysis artefacts.  They are preserved for the
    # deterministic rule engine but never written as editable CRM fields.
    normalized["_stage_evidence"] = parsed.get("_stage_evidence", {})
    normalized["_conflicts"] = parsed.get("_conflicts", [])
    normalized["_next_step_plan"] = parsed.get("_next_step_plan", {})
    return normalized


def call_critic(extracted_data: dict, use_mock: bool = False) -> dict:
    """步骤三：Critic — 语义审核"""
    if use_mock:
        return {
            "overall_status": "PASS",
            "issues": [],
            "missing_critical_fields": [],
            "summary": "Mock mode: Critic skipped",
        }
    extracted_json = json.dumps(extracted_data, ensure_ascii=False, indent=2)
    result_text = _call_llm_real(
        "", CRITIC_PROMPT.format(extracted_json=extracted_json), json_mode=True
    )

    # Use cleaning function to parse JSON
    parsed = clean_and_parse_llm_json(result_text)
    if not isinstance(parsed, dict):
        return {
            "_raw_response": result_text,
            "_parse_error": True,
            "overall_status": "REVIEW",
            "issues": [],
            "missing_critical_fields": [],
            "summary": "Critic 解析失败，按 REVIEW 处理",
        }
    # Ensure keys exist
    parsed.setdefault("issues", [])
    parsed.setdefault("missing_critical_fields", [])
    parsed.setdefault("summary", "")
    return parsed


def reconcile_critic_with_rule_engine(critic: dict, extracted_data: dict) -> dict:
    """Keep Critic feedback aligned with the canonical rule-engine result.

    The Critic remains useful for semantic quality checks, but it must not
    label a correctly preserved range or a correctly detected source conflict
    as an AI failure.  The rule engine is the final authority for the fixed
    exercise rules, stage calculation and critical-field completeness.
    """
    if not isinstance(critic, dict):
        return {
            "overall_status": "REVIEW",
            "issues": [],
            "missing_critical_fields": [],
            "summary": "Critic 返回格式异常，建议人工复核。",
        }

    reconciled = dict(critic)
    raw_issues = critic.get("issues", [])
    raw_issues = raw_issues if isinstance(raw_issues, list) else []
    qualified_fields = extracted_data.get("_qualified_fields", {})
    qualified_fields = qualified_fields if isinstance(qualified_fields, dict) else {}
    critical_missing = [
        field
        for field in CRITICAL_FIELDS
        if not extract_value(extracted_data.get(field, {}))
    ]

    issues = []
    removed_issue = False
    downgraded_incomplete_issue = False
    stage_is_rule_derived = extract_value(extracted_data.get("stage", {})) is not None
    for issue in raw_issues:
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("type", ""))
        affected = issue.get("affected_fields", [])
        affected = affected if isinstance(affected, list) else []
        affected = {str(field) for field in affected}

        # A qualified range is deliberately preserved by the fixed rules.
        # Its derived stage is therefore not a "guess made factual".
        if issue_type == "uncertain_falsified" and affected and all(
            field in qualified_fields or (field == "stage" and stage_is_rule_derived)
            for field in affected
        ):
            removed_issue = True
            continue

        # The assignment defines serious incompleteness by critical fields
        # only. Non-critical gaps still remain in the explicit unconfirmed
        # list, but must not become a false FAIL.
        if issue_type == "insufficient_info" and len(critical_missing) < 3:
            removed_issue = True
            continue
        if issue_type == "insufficient_info":
            # Missing facts that the system has honestly labelled
            # "未确认" require follow-up, not a verdict that AI fabricated
            # a fact. Canonicalize both the count and the affected fields.
            normalized_issue = dict(issue)
            normalized_issue["severity"] = "medium"
            normalized_issue["affected_fields"] = critical_missing
            normalized_issue["description"] = (
                f"关键字段缺失 {len(critical_missing)} 项，均已明确标为未确认，需人工补充。"
            )
            issues.append(normalized_issue)
            downgraded_incomplete_issue = True
            continue
        issues.append(issue)

    reconciled["issues"] = issues
    reconciled["missing_critical_fields"] = critical_missing

    has_source_conflict = bool(extracted_data.get("_conflicts"))
    has_real_high_issue = any(
        str(issue.get("severity", "")).lower() == "high"
        and str(issue.get("type", "")) != "contradiction"
        for issue in issues
        if isinstance(issue, dict)
    )
    original_status = str(critic.get("overall_status", "REVIEW")).upper()

    # A correctly detected source conflict requires review, not an AI-failure
    # verdict.  Do not hide a separate, still-valid high-severity AI issue.
    should_review_for_safe_incompleteness = (
        downgraded_incomplete_issue and not has_real_high_issue
    )
    if (has_source_conflict or should_review_for_safe_incompleteness) and (
        original_status in {"PASS", "REVIEW"}
        or (original_status == "FAIL" and (removed_issue or downgraded_incomplete_issue) and not has_real_high_issue)
    ):
        reconciled["overall_status"] = "REVIEW"
        if has_source_conflict and should_review_for_safe_incompleteness:
            reconciled["summary"] = "原始记录存在关键信息冲突，且关键信息不足；系统已明确标为未确认，请人工复核并补充。"
        elif has_source_conflict:
            reconciled["summary"] = "检测到原始记录中的关键业务信息冲突；系统已保留双方依据并将相关字段标为未确认，请人工复核。"
        else:
            reconciled["summary"] = "商机关键信息不足，但均已明确标为未确认；请人工补充，不构成 AI 事实错误。"
    else:
        reconciled["overall_status"] = original_status if original_status in {"PASS", "REVIEW", "FAIL"} else "REVIEW"

    return reconciled


def call_nextprompts(extracted_data: dict, critic_result: dict, use_mock: bool = False) -> str:
    """Generate follow-up prompts (skipped in mock mode — uses _mock_nextprompt instead)."""
    if use_mock:
        return ""  # caller will use _mock_nextprompt directly
    extracted_json = json.dumps(extracted_data, ensure_ascii=False, indent=2)
    critic_json = json.dumps(critic_result, ensure_ascii=False, indent=2)
    result_text = _call_llm_real("", NEXTPROMPT_PROMPT.format(
        extracted_json=extracted_json, critic_result=critic_json
    ))
    if result_text.startswith("[LLM调用异常]"):
        return _local_nextprompt_fallback(extracted_data)

    # 清理可能的 markdown 包裹
    cleaned = result_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    return cleaned


_LOCAL_FOLLOW_UP_QUESTIONS = {
    "customer_name": "请确认客户公司的完整名称和所属主体。",
    "need": "当前最需要优先解决的业务问题是什么？",
    "scenario": "这个问题具体发生在哪个业务场景？",
    "budget": "目前是否已有预算范围、立项金额或审批计划？",
    "decision_maker": "该项目最终由哪位负责人拍板或审批？",
    "influencer": "还有哪些技术、业务或采购相关人员会影响决策？",
    "timeline": "预计何时启动、选型或上线？关键节点分别是什么？",
    "risk": "当前是否存在预算、立项、竞品、资源或进度风险？",
    "next_step": "下一步具体做什么、由谁负责、何时完成？",
}


def _local_nextprompt_fallback(extracted_data: dict) -> str:
    """Provide useful follow-up questions when the optional coaching call fails."""
    missing = []
    for item in extracted_data.get("_unconfirmed_items", []):
        if isinstance(item, dict) and item.get("field") in _LOCAL_FOLLOW_UP_QUESTIONS:
            missing.append(item["field"])
    if not missing:
        missing = [
            field for field in _LOCAL_FOLLOW_UP_QUESTIONS
            if not extract_value(extracted_data.get(field, {}))
        ]
    questions = []
    for field in dict.fromkeys(missing):
        questions.append(_LOCAL_FOLLOW_UP_QUESTIONS[field])
        if len(questions) == 5:
            break
    if not questions:
        return "当前商机信息较完整；建议继续确认方案、采购和决策节点。"
    return "💡 追问服务暂不可用，以下为本地建议：\n\n" + "\n".join(
        f"- {question}" for question in questions
    )


# ──────────────────────────────────────────────
# Mock 实现（保持不变）
# ──────────────────────────────────────────────

def _empty_mock_draft() -> dict:
    """Return an evidence-safe empty draft for unsupported Mock inputs."""
    return {field: {"value": None, "quote": None} for field in OPPORTUNITY_FIELDS}


def _mock_transcript_summary(raw_text: str) -> str:
    """Provide a deterministic stand-in for the transcript preprocessor in Mock mode."""
    if all(marker in raw_text for marker in ("工地", "物料", "王总", "赵工")):
        return """## 参与人及角色
- 王总：客户方运营副总；赵工：客户方现场项目经理；小李：我方销售。

## 客户现状与痛点
- 多个工地的钢筋、水泥等物料靠手工台账和照片录入 Excel，库存账实不符且无法实时联动。

## 需求与核心场景
- 希望将物料入库/出库与工程进度绑定，在现场可用平板或手机进行试用。

## 预算与立项状态
- 项目预算尚无准数，且尚未立项；“四五十万额度”被王总明确为 OA 升级预算，不能计入本项目。

## 决策与影响角色
- 是否能获得预算取决于方案能否打动李董；王总、赵工参与评估和试用。

## 时间计划与下一步
- 下周搭建测试环境并发送账号；王总下周出差，线上演示具体时间待微信确认。

## 风险、异议与矛盾信息
- 地下室网络较差，扫码能力需支持离线缓存；预算未立项且时间未定。

## 可追溯原文摘录
- “每天下面几个工地进出多少钢筋水泥，全靠赵工他们拿个大厚本子记。”
- “如果能把进度跟这个物料绑在一起看，确实能省不少事。”
- “咱们这个项目，预算还没立项呢。”
- “下周我安排我们售前，把测试环境搭好，把账号给两位发过去。”"""
    return raw_text


def _mock_transcript_extractor(transcript: str) -> dict | None:
    """Extract high-confidence facts from the bundled long-form recording demo."""
    if not all(marker in transcript for marker in ("工地", "物料", "王总", "赵工")):
        return None
    return {
        "customer_name": {"value": None, "quote": None},
        "need": {
            "value": "工地物料与工程进度协同管理",
            "quote": "下面几个工地每天进出多少钢筋水泥，全靠赵工他们拿个大厚本子记；如果能把进度跟这个物料绑在一起看，确实能省不少事",
        },
        "scenario": {
            "value": "多工地物料进出、库存盘点与工程进度协同",
            "quote": "报表上明明写着还有两百吨料，结果我去现场一看，毛都没有；纯手工或者半自动化 Excel 没法实时联动",
        },
        "budget": {"value": None, "quote": None},
        "decision_maker": {
            "value": "李董（是否为最终拍板人待确认）",
            "quote": "得看你们方案能不能打动李董",
        },
        "influencer": {
            "value": "王总（运营副总）、赵工（现场项目经理）",
            "quote": "王总（客户方 - 运营副总）；赵工（客户方 - 现场项目经理）",
        },
        "timeline": {
            "value": "下周准备测试环境，具体演示时间待确认",
            "quote": "下周我安排我们售前，把测试环境搭好，把账号给两位发过去；具体时间下周再说",
        },
        "stage": {
            "value": "S2",
            "quote": "针对物料对不上账这个场景，弄个专门的演示；搞个测试账号让老赵他们先去点一点",
        },
        "risk": {
            "value": "项目尚未立项、预算未确认；现场弱网可能影响扫码使用；演示时间待定",
            "quote": "咱们这个项目，预算还没立项呢；地下室没网，半天扫不出来；具体时间下周再说",
        },
        "next_step": {
            "value": "下周搭建测试环境并发送账号；微信确认线上演示时间",
            "quote": "下周我安排我们售前，把测试环境搭好，把账号给两位发过去；到时候微信上找我",
        },
    }


def _mock_generic_extractor(transcript: str) -> dict | None:
    """Conservative local fallback for custom notes in Mock mode.

    This is deliberately narrow: it extracts only explicit organization,
    product, and negative-follow-up signals instead of inventing a complete
    opportunity from arbitrary text.
    """
    organization_match = None
    for pattern in (
        r"老客户\s*([\u4e00-\u9fffA-Za-z0-9·]{2,30}(?:医疗|医院|集团|公司|科技|股份|企业))",
        r"客户(?:名称)?(?:是|为|：|:)\s*([\u4e00-\u9fffA-Za-z0-9·]{2,30}(?:医疗|医院|集团|公司|科技|股份|企业))",
        r"拜访(?:了)?\s*([\u4e00-\u9fffA-Za-z0-9·]{2,30}(?:医疗|医院|集团|公司|科技|股份|企业))",
    ):
        organization_match = re.search(pattern, transcript)
        if organization_match:
            break
    customer_name = organization_match.group(1) if organization_match else None
    products = [name for name in ("CDSS", "MES", "CRM", "ERP", "WMS") if name in transcript]
    product = products[0] if products else None
    rejection_terms = ("够用了", "不想折腾", "不需要跟进", "暂不跟进", "暂时不需要", "暂无需求", "没兴趣")
    has_rejection = any(term in transcript for term in rejection_terms)

    if not customer_name and not product:
        return None

    draft = _empty_mock_draft()
    if customer_name:
        draft["customer_name"] = {
            "value": customer_name,
            "quote": organization_match.group(0),
        }

    if product and has_rejection:
        draft["need"] = {
            "value": f"{product}产品评估（客户短期暂无升级需求）",
            "quote": "提了一嘴我们的新产品；现在的系统够用了，短期内不想折腾",
        }
        draft["stage"] = {
            "value": "S0",
            "quote": "对方没啥反应，说现在的系统够用了，短期内不想折腾",
        }
        draft["risk"] = {
            "value": "客户当前系统可用，短期无改造意愿",
            "quote": "现在的系统够用了，短期内不想折腾",
        }
        draft["next_step"] = {
            "value": "暂不跟进，后续观察需求变化",
            "quote": "暂时不需要跟进了",
        }
    elif product:
        draft["need"] = {
            "value": f"{product}产品需求待确认",
            "quote": f"提到{product}",
        }
        draft["stage"] = {"value": "S0", "quote": f"提到{product}"}

    return draft


def _mock_extractor(transcript: str) -> dict:
    """Mock extractor for the bundled scenarios; never reuse another scenario's data."""
    if len(transcript) < 30:
        return _empty_mock_draft()

    transcript_result = _mock_transcript_extractor(transcript)
    if transcript_result is not None:
        return transcript_result

    if "XX医疗" in transcript and "CDSS" in transcript:
        # 完整信息演示样例
        return {
            "customer_name": {"value": "XX医疗", "quote": "拜访了XX医疗"},
            "need": {"value": "CDSS系统升级", "quote": "全院级别的CDSS系统升级项目"},
            "scenario": {"value": "全院级临床决策支持", "quote": "全院级别的CDSS系统升级项目"},
            "budget": {"value": "约200万，已批立项", "quote": "预算大概在200万左右，已经批了立项申请"},
            "decision_maker": {"value": "王总（副院长）", "quote": "XX医疗的王总（副院长）"},
            "influencer": {"value": "张主任（信息科）、李部长（临床）", "quote": "信息科的张主任和临床的李部长"},
            "timeline": {"value": "Q3前完成选型", "quote": "今年Q3要做选型"},
            "stage": {"value": "S2", "quote": "已经跟张主任做了两轮技术演示"},
            "risk": {"value": None, "quote": None},
            "next_step": {"value": "下周三方案汇报，售前陈明参加", "quote": "下周三安排一次给王总的方案汇报，由我司售前经理陈明参加"},
        }

    if all(marker in transcript for marker in ("张总", "Q2", "Q3", "立项")):
        # Conflicting-information scenario: do not select a side or invent a customer.
        return {
            "customer_name": {"value": None, "quote": None},
            "need": {"value": "项目意向（具体需求待确认）", "quote": "确认了项目意向"},
            "scenario": {"value": None, "quote": None},
            "budget": {"value": None, "quote": None},
            "decision_maker": {"value": None, "quote": None},
            "influencer": {"value": "张总（角色待确认）", "quote": "张总个人很看好我们，承诺会全力推动"},
            "timeline": {"value": None, "quote": None},
            "stage": {"value": None, "quote": None},
            "risk": {
                "value": "预算、立项与时间计划存在矛盾，需人工核实",
                "quote": "Q2一定会上线，预算500万没问题，已经走完内部审批；目前连立项都没提交，预算还在写报告中，实际最早Q3才能开始可行性论证",
            },
            "next_step": {"value": None, "quote": None},
        }

    if "某制造业企业" in transcript and "MES" in transcript:
        # Budget-missing scenario
        return {
            "customer_name": {"value": "某制造业企业", "quote": "某制造业企业"},
            "need": {"value": "MES系统改造", "quote": "有MES系统改造的需求"},
            "scenario": {"value": "产线制造执行", "quote": "先覆盖两条产线"},
            "budget": {"value": None, "quote": None},
            "decision_maker": {"value": None, "quote": None},
            "influencer": {"value": "赵经理（生产部）", "quote": "生产部赵经理"},
            "timeline": {"value": "年底前上线一期", "quote": "希望在年底前上线一期模块"},
            "stage": {"value": "S1", "quote": "对我们产品感兴趣"},
            "risk": {"value": "时间紧张", "quote": "时间比较紧"},
            "next_step": {"value": "本周五现场需求调研", "quote": "本周五带解决方案去客户现场做需求调研"},
        }

    generic_result = _mock_generic_extractor(transcript)
    if generic_result is not None:
        return generic_result

    # Unsupported custom input remains unknown rather than borrowing values
    # from a predefined scenario.
    return _empty_mock_draft()


# ──────────────────────────────────────────────
# Rule Engine（修复返回值 bug）
# ──────────────────────────────────────────────

RULE_LOG_LIMIT = 20

FIELD_DISPLAY_NAMES = {
    "customer_name": "客户名称",
    "need": "客户需求",
    "scenario": "核心场景",
    "budget": "预算",
    "decision_maker": "决策人",
    "influencer": "影响人",
    "timeline": "时间计划",
    "stage": "商机阶段",
    "risk": "风险",
    "next_step": "下一步行动",
}

UNCONFIRMED_GUIDANCE = {
    "customer_name": "客户全称是什么？",
    "need": "当前最需要优先解决的业务问题是什么？",
    "scenario": "这个问题具体发生在哪个业务场景？",
    "budget": "目前是否已有预算范围、立项金额或审批计划？",
    "decision_maker": "该项目最终由哪位负责人拍板或审批？",
    "influencer": "还有哪些技术、业务或采购相关人员会影响决策？",
    "timeline": "预计何时启动、选型或上线？关键节点分别是什么？",
    "stage": "请确认客户目前所处的销售阶段及对应证据。",
    "risk": "当前是否存在预算、立项、竞品、资源或进度风险？",
    "next_step": "下一步具体做什么、由谁负责、何时完成？",
}

FACTUAL_FIELDS = {
    "customer_name", "need", "scenario", "budget", "decision_maker",
    "influencer", "timeline", "next_step",
}
UNCERTAIN_TERMS = ("可能", "应该", "感觉", "挺感兴趣", "我猜", "我估计")
STAGE_EVIDENCE_ORDER = (
    ("contract_signed", "S5"),
    ("decision_approval", "S4"),
    ("commercial_evaluation", "S3"),
    ("solution_validation", "S2"),
    ("need_or_scenario", "S1"),
)


def _normalized_evidence_text(value: object) -> str:
    """Compare quotes conservatively while ignoring formatting-only differences."""
    if value is None:
        return ""
    return re.sub(r"[\s\u3000\"'“”‘’`*_#>\-—–,，。！？!?：:；;（）()【】\[\]]+", "", str(value))


def _quote_is_grounded(quote: object, source_text: str) -> bool:
    normalized_quote = _normalized_evidence_text(quote)
    normalized_source = _normalized_evidence_text(source_text)
    return len(normalized_quote) >= 4 and normalized_quote in normalized_source


def _source_clause(source_text: str, match: re.Match) -> str:
    """Return the original sentence around a regex hit for auditable evidence."""
    start = max(source_text.rfind(mark, 0, match.start()) for mark in ("。", "！", "？", "\n")) + 1
    ends = [source_text.find(mark, match.end()) for mark in ("。", "！", "？", "\n")]
    end_candidates = [item for item in ends if item != -1]
    end = min(end_candidates) if end_candidates else len(source_text)
    return source_text[start:end].strip()


def _recover_qualified_budget(source_text: str) -> dict | None:
    """Recover a directly stated *range* budget without treating it as exact.

    A customer saying "大概 80 万" has discussed a budget, but has not
    confirmed its precise amount.  The displayed value therefore preserves the
    qualifier and makes the remaining uncertainty explicit.
    """
    for match in re.finditer(r"预算[^。！？\n]{0,80}", source_text):
        clause = _source_clause(source_text, match)
        # Do not turn the salesperson's own speculation into a CRM value.
        if any(term in clause for term in ("我猜", "我估计", "应该", "可能")):
            continue
        amount_match = re.search(
            r"(\d+(?:\.\d+)?(?:\s*[-~至到]\s*\d+(?:\.\d+)?)?\s*(?:万元|万|百万|千万|元))",
            clause,
        )
        if not amount_match:
            continue
        amount = re.sub(r"\s+", "", amount_match.group(1))
        amount_start, amount_end = amount_match.span()
        has_qualifier_word = (
            any(term in clause[:amount_end] for term in ("约", "大概", "大约", "预计"))
            or "左右" in clause[amount_start:]
        )
        is_qualified = has_qualifier_word or bool(re.search(r"[-~至到]", amount))
        is_approved = bool(re.search(r"(已经|已|获).{0,4}(批|批准|通过)|已批", clause))
        value = ("约" if has_qualifier_word else "") + amount
        if is_approved:
            value += "，已批"
        if is_qualified:
            value += "（金额精确值待确认）"
        return {
            "value": value,
            "quote": clause,
            "precision_pending": is_qualified,
        }
    return None


TIME_VALUE_PATTERN = (
    r"(?:约|大概|预计)?\s*"
    r"(?:(?:本周|这周|下周)[一二三四五六日天]?(?:前)?(?:上午|下午|晚上)?|"
    r"周[一二三四五六日天](?:上午|下午|晚上)?|下个月|今年[Qq][1-4]|"
    r"明年|年底|明天|后天|[Qq][1-4]|\d{1,2}月\d{1,2}[日号]?|\d{1,2}[日号])(?:左右)?"
)

# Dialogue labels are optional metadata in a sales note. When present, they
# are reliable enough to identify a participant and an explicitly named owner.
_DIALOGUE_SPEAKER_PATTERN = re.compile(
    r"(?m)^\s*(?:\*\*)?(?P<label>[^:\n：*]{1,40}?)[：:](?:\*\*)?\s*"
)
_SALES_SPEAKER_PATTERN = re.compile(r"(?:售前|销售|商务|客户经理|顾问|我方)")


def _dialogue_speaker_labels(source_text: str) -> set[str]:
    """Return normalized labels only from explicitly labelled dialogue turns."""
    return {
        _normalized_evidence_text(match.group("label"))
        for match in _DIALOGUE_SPEAKER_PATTERN.finditer(source_text)
        if _normalized_evidence_text(match.group("label"))
    }


def _extract_sales_follow_up_commitment(source_text: str) -> tuple[str | None, list[str]]:
    """Read a named sales speaker's literal follow-up commitment, if present.

    Unlabelled prose and customer turns are deliberately ignored. Without an
    explicit sales speaker, the UI must continue to show a suggested owner.
    """
    turns = list(_DIALOGUE_SPEAKER_PATTERN.finditer(source_text))
    for index, turn in enumerate(turns):
        speaker = turn.group("label").strip()
        if not _SALES_SPEAKER_PATTERN.search(speaker):
            continue
        end = turns[index + 1].start() if index + 1 < len(turns) else len(source_text)
        utterance = source_text[turn.end() : end]
        if not FOLLOW_UP_ACTION_PATTERN.search(utterance):
            continue
        times = [match.group(0).strip() for match in re.finditer(
            TIME_VALUE_PATTERN, utterance, flags=re.IGNORECASE
        )]
        return speaker, list(dict.fromkeys(times))
    return None, []


def _has_qualified_time(value: object) -> bool:
    """Whether a stated time is a range or too coarse to be an exact date."""
    text = str(value or "")
    normalized = re.sub(r"\s+", "", text)
    has_explicit_qualifier = bool(
        re.search(r"(?:约|大概|预计)\s*(?:本周|下周|下个月|今年|明年|年底|[Qq][1-4]|\d)", text)
        or re.search(r"(?:本周|下周|下个月|今年|明年|年底|[Qq][1-4]|\d[^。！？\n]{0,8})左右", text)
    )
    # “下周”“下个月”“Q3” state a valid period, but not a date. A weekday
    # or calendar date remains precise enough and therefore does not match.
    is_coarse_period = bool(re.fullmatch(
        r"(?:约|大概|预计)?(?:本周|下周|下个月|今年[Qq][1-4]|明年|年底|[Qq][1-4])(?:左右)?",
        normalized,
    ))
    return has_explicit_qualifier or is_coarse_period


def _recover_qualified_timeline(source_text: str) -> dict | None:
    """Preserve a customer-stated approximate time without claiming exactness."""
    for sentence in re.split(r"(?<=[。！？!?])|\n+", source_text):
        quote = sentence.strip()
        clause = quote.rstrip("。！？!?")
        if not clause or not re.search(TIME_VALUE_PATTERN, clause, flags=re.IGNORECASE):
            continue
        if not _has_qualified_time(clause):
            continue
        if any(term in clause for term in ("我猜", "我估计", "应该")):
            continue
        return {
            "value": f"{clause}（具体日期待确认）",
            "quote": quote,
            "precision_pending": True,
        }
    return None


FOLLOW_UP_ACTION_PATTERN = re.compile(
    r"(?:下一步|后续|回头|我(?:们)?(?:先|会|来|打算|安排|跟进|联系|找|约|发|带)|"
    r"(?:等|到)?(?:本周|下周|下个月|明天|后天).{0,20}(?:安排|跟进|联系|找|约|发|带|提交|同步|拉))"
)


def _recover_follow_up_action(source_text: str) -> dict | None:
    """Recover an explicitly stated sales follow-up that the model omitted.

    This recognizes only clear action cues in the original record. It does
    not turn a vague wish or a customer's reaction into an action, and the
    original phrase remains the auditable evidence.
    """
    for sentence in re.split(r"(?<=[。！？!?])|\n+", source_text):
        quote = sentence.strip()
        clause = quote.rstrip("。！？!?").strip()
        if not clause or not FOLLOW_UP_ACTION_PATTERN.search(clause):
            continue
        return {"value": clause, "quote": quote}
    return None


def _report_item(rule: str, status: str, field: str | None, result: str, action: str) -> dict:
    return {
        "rule": rule,
        "status": status,
        "field": FIELD_DISPLAY_NAMES.get(field or "", field or "—"),
        "result": result,
        "action": action,
    }


def _field_entry(value: object = None, quote: object = None) -> dict:
    return {"value": None if value is None else str(value).strip() or None,
            "quote": None if quote is None else str(quote).strip() or None}


def _stage_evidence_from_source(data: dict, source_text: str) -> dict:
    """Validate model evidence and fill only conservative deterministic fallbacks."""
    raw_evidence = data.get("_stage_evidence", {})
    evidence = {
        key: {"met": False, "quote": None}
        for key, _stage in STAGE_EVIDENCE_ORDER
    }

    if isinstance(raw_evidence, dict):
        for key in evidence:
            candidate = raw_evidence.get(key, {})
            if isinstance(candidate, dict) and candidate.get("met"):
                quote = candidate.get("quote")
                # A sales question or a customer's "我猜 / 可能" answer is
                # not commercial evaluation. It must not promote a deal to
                # S3 simply because it happens to mention a budget figure.
                commercial_guess = key == "commercial_evaluation" and any(
                    term in str(quote or "")
                    for term in (*UNCERTAIN_TERMS, "不好说", "没法给个准数")
                )
                if _quote_is_grounded(quote, source_text) and not commercial_guess:
                    evidence[key] = {"met": True, "quote": str(quote).strip()}

    # A need or scenario that has already passed the fact-boundary test is
    # direct evidence for S1.  It does not promote a record beyond S1.
    if not evidence["need_or_scenario"]["met"]:
        for field in ("need", "scenario"):
            entry = data.get(field, {})
            quote = extract_quote(entry) if isinstance(entry, dict) else None
            if extract_value(entry) and _quote_is_grounded(quote, source_text):
                evidence["need_or_scenario"] = {"met": True, "quote": quote}
                break

    # Commercial stage evidence must come from an already validated budget
    # fact or from non-speculative model evidence above. A broad keyword
    # match on “预算” would incorrectly treat questions and guesses as a
    # commercial commitment.
    if not evidence["commercial_evaluation"]["met"]:
        budget_entry = data.get("budget", {})
        budget_value = extract_value(budget_entry) if isinstance(budget_entry, dict) else None
        budget_quote = extract_quote(budget_entry) if isinstance(budget_entry, dict) else None
        if budget_value and _quote_is_grounded(budget_quote, source_text):
            evidence["commercial_evaluation"] = {"met": True, "quote": budget_quote}

    fallback_patterns = {
        "solution_validation": r"同意.{0,12}(演示|试用|技术交流|方案评估)|安排.{0,12}(演示|试用|技术交流)|测试账号|测试环境|技术交流|方案评估",
        "decision_approval": r"立项申请.{0,8}(已|获).{0,4}批|已.{0,8}立项|进入.{0,8}(立项|审批|供应商决策)|供应商.{0,8}(决策|选定|选择)",
        "contract_signed": r"(已签|签订).{0,8}合同|正式订单.{0,8}(已|确认)|订单.{0,8}已确认",
    }
    for key, pattern in fallback_patterns.items():
        if evidence[key]["met"]:
            continue
        match = re.search(pattern, source_text, flags=re.IGNORECASE)
        if match:
            evidence[key] = {"met": True, "quote": _source_clause(source_text, match)}
    return evidence


def _normalized_conflicts(data: dict, source_text: str) -> list[dict]:
    """Keep only conflicts with at least two distinct pieces of source evidence."""
    candidates = data.get("_conflicts", [])
    if not isinstance(candidates, list):
        candidates = []
    conflicts = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        fields = candidate.get("fields", [])
        if isinstance(fields, str):
            fields = [fields]
        # ``risk`` carries the conflict explanation; it is never itself a
        # contradictory source fact or an unconfirmed CRM field.
        fields = [
            field for field in fields
            if field in OPPORTUNITY_FIELDS and field != "risk"
        ]
        quotes = candidate.get("quotes", [])
        if isinstance(quotes, str):
            quotes = [quotes]
        valid_quotes = []
        for quote in quotes:
            if _quote_is_grounded(quote, source_text) and quote not in valid_quotes:
                valid_quotes.append(str(quote).strip())
        if fields and len(valid_quotes) >= 2:
            conflicts.append({
                "fields": fields,
                "quotes": valid_quotes,
                "description": str(candidate.get("description") or "原始记录存在相互矛盾的表述"),
            })

    # Conservative fallback for the common "已立项/未立项" contradiction.
    has_approved = re.search(r"(立项申请.{0,8}(已|获).{0,4}批|已.{0,8}立项|走完.{0,8}内部审批)", source_text)
    has_not_started = re.search(
        r"(未.{0,5}立项|立项.{0,5}(未|没)提交|连.{0,3}立项.{0,5}(未|没)提交|还没.{0,5}立项)",
        source_text,
    )
    if has_approved and has_not_started:
        quotes = [_source_clause(source_text, has_approved), _source_clause(source_text, has_not_started)]
        if not any(set(item["quotes"]) == set(quotes) for item in conflicts):
            conflicts.append({
                "fields": ["budget", "timeline", "stage"],
                "quotes": quotes,
                "description": "立项或审批状态前后矛盾，无法确认项目阶段",
            })
    return conflicts


def _build_next_step_plan(data: dict, source_text: str) -> dict:
    """Present a required action/owner/time plan without inventing facts."""
    next_step = data.get("next_step", {})
    action = extract_value(next_step) if isinstance(next_step, dict) else None
    next_step_quote = extract_quote(next_step) if isinstance(next_step, dict) else None
    raw_plan = data.get("_next_step_plan", {})
    raw_plan = raw_plan if isinstance(raw_plan, dict) else {}

    def supported_component(name: str) -> str | None:
        value = raw_plan.get(name)
        return str(value).strip() if _quote_is_grounded(value, source_text) else None

    owner = supported_component("owner")
    due_time = supported_component("time")
    source_owner, source_times = _extract_sales_follow_up_commitment(source_text)

    # Mock mode and older model responses may not provide the optional plan
    # object.  Recover only literal owner/time mentions from the already
    # grounded next-step quote; otherwise retain the explicit suggestion/
    # pending labels below.
    plan_quote = str(next_step_quote or "")
    if not owner and re.search(r"我(?:先|带|来|会|打算|负责|安排|跟进|联系|拉)", plan_quote):
        owner = "我方销售（记录者）"
    if not owner and re.search(
        r"^\s*等(?:本周|下周|下个月|明天|后天).{0,20}(?:安排|跟进|联系|找|约|发|带|提交|同步|拉)",
        plan_quote,
    ):
        # An ellipsized "等下周约…" in a sales note is a first-person
        # follow-up plan, not a customer commitment.
        owner = "我方销售（记录者）"
    if not owner and plan_quote:
        owner_match = re.search(
            r"由(?P<owner>[\u4e00-\u9fffA-Za-z0-9]{1,18}?)(?:负责|参加|跟进|执行|安排|，|。|；|$)",
            plan_quote,
        )
        if owner_match:
            owner = owner_match.group("owner").strip()
    if plan_quote:
        time_match = re.search(TIME_VALUE_PATTERN, plan_quote, flags=re.IGNORECASE)
        # Prefer the literal qualified time in the source over a model summary
        # that might have silently dropped “约 / 大概 / 左右”.
        if time_match and (_has_qualified_time(time_match.group(0)) or not due_time):
            due_time = time_match.group(0).strip()
    if not owner and source_owner:
        owner = source_owner

    time_values = [due_time] if due_time else []
    for source_time in source_times:
        if source_time not in time_values:
            time_values.append(source_time)
    time_precision_pending = any(_has_qualified_time(value) for value in time_values)
    displayed_values = [
        f"{value}（具体日期待确认）" if _has_qualified_time(value) else value
        for value in time_values
    ]
    displayed_time = "；".join(displayed_values) or None
    return {
        "action": action or "待确认下一步行动",
        "owner": owner or "我方销售（建议）",
        "time": displayed_time or "待确认",
        "action_confirmed": bool(action),
        "owner_confirmed": bool(owner),
        "time_stated": bool(due_time),
        "time_confirmed": bool(due_time) and not time_precision_pending,
        "time_precision_pending": time_precision_pending,
    }


def build_unconfirmed_items(data: dict, conflicts: list[dict] | None = None) -> list[dict]:
    """Create an explicit, traceable list required by the exercise output."""
    conflicts = conflicts or []
    qualified_fields = data.get("_qualified_fields", {})
    qualified_fields = qualified_fields if isinstance(qualified_fields, dict) else {}
    conflict_fields = {
        field
        for conflict in conflicts
        for field in conflict.get("fields", [])
    }
    items = []
    for field in OPPORTUNITY_FIELDS:
        entry = data.get(field, {})
        value = extract_value(entry) if isinstance(entry, dict) else None
        if field in conflict_fields:
            reason = "原始记录存在相互矛盾的表述，系统未选择任一版本。"
        elif field in qualified_fields:
            reason = str(qualified_fields[field])
        elif value is None:
            reason = "原始记录未提供可确认的依据。"
        else:
            continue
        items.append({
            "field": field,
            "label": FIELD_DISPLAY_NAMES[field],
            "reason": reason,
            "question": UNCONFIRMED_GUIDANCE[field],
        })
    return items


def rule_engine(data: dict, source_text: str = "") -> tuple[dict, list[str], list[dict], int]:
    """Apply the exercise rules before any field reaches human review.

    The model may extract evidence, but this function owns the final factual
    boundary, stage calculation, conflict handling and missing-information
    output.  Returns ``(data, short_log, report, rules_applied)``.
    """
    processed = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in data.items()
    }
    report: list[dict] = []
    short_log: list[str] = []
    rules_applied = 0

    # Rule 1–3: a factual CRM value needs a grounded quote and may not be a
    # salesperson's uncertainty expressed as a fact.
    for field in FACTUAL_FIELDS:
        entry = processed.get(field)
        if not isinstance(entry, dict):
            processed[field] = _field_entry()
            continue
        value = extract_value(entry)
        quote = extract_quote(entry)
        if value and not _quote_is_grounded(quote, source_text):
            processed[field] = _field_entry()
            report.append(_report_item(
                "事实边界", "拦截", field,
                "字段值缺少可追溯的原始记录依据。", "已改为未确认，等待人工复核。",
            ))
            short_log.append(f"⚠️ [{field}] 缺少可追溯原文依据，已标为未确认")
            rules_applied += 1
            continue
        if value and any(term in str(quote) for term in UNCERTAIN_TERMS):
            processed[field] = _field_entry()
            report.append(_report_item(
                "推断处理", "拦截", field,
                "原文包含不确定或销售猜测表达。", "已改为未确认，不作为 CRM 事实。",
            ))
            short_log.append(f"⚠️ [{field}] 命中不确定表达，已标为未确认")
            rules_applied += 1

    # A speaker label such as “张总（甲方）” identifies a person in a dialogue,
    # not the customer's legal or trade name. Reject only exact labelled
    # participants; ordinary company names remain untouched.
    customer_name = processed.get("customer_name", {})
    customer_value = extract_value(customer_name) if isinstance(customer_name, dict) else None
    if customer_value and _normalized_evidence_text(customer_value) in _dialogue_speaker_labels(source_text):
        processed["customer_name"] = _field_entry()
        report.append(_report_item(
            "实体边界", "拦截", "customer_name",
            "客户名称与原始对话中的参与人标签完全一致，无法作为组织名称确认。",
            "已改为未确认，等待补充客户公司全称。",
        ))
        short_log.append("⚠️ [customer_name] 参与人标签不能作为客户公司名称，已标为未确认")
        rules_applied += 1

    # A customer-stated range such as "大概 80 万" is useful commercial
    # evidence, but never an exact confirmed amount. Preserve the qualifier
    # and attach an explicit precision-pending marker even when the model
    # omitted the budget entirely.
    qualified_fields = {}
    recovered_budget = _recover_qualified_budget(source_text)
    if recovered_budget:
        processed["budget"] = _field_entry(
            recovered_budget["value"], recovered_budget["quote"]
        )
        if recovered_budget["precision_pending"]:
            qualified_fields["budget"] = "客户仅给出约数范围，金额精确值待确认。"
            report.append(_report_item(
                "推断处理", "待确认", "budget",
                "客户明确讨论预算，但金额带有“约/大概/左右”限定。",
                "保留范围金额，并标注金额精确值待确认。",
            ))
            short_log.append("⚠️ [budget] 已保留范围金额，精确值待确认")
            rules_applied += 1

    recovered_timeline = _recover_qualified_timeline(source_text)
    if recovered_timeline:
        processed["timeline"] = _field_entry(
            recovered_timeline["value"], recovered_timeline["quote"]
        )
        qualified_fields["timeline"] = "客户仅给出时间范围，具体日期待确认。"
        report.append(_report_item(
            "推断处理", "待确认", "timeline",
            "客户明确给出时间范围，但未约定精确日期。",
            "保留原始时间范围，并标注具体日期待确认。",
        ))
        short_log.append("⚠️ [timeline] 已保留时间范围，具体日期待确认")
        rules_applied += 1

    recovered_next_step = _recover_follow_up_action(source_text)
    if recovered_next_step and not extract_value(processed.get("next_step", {})):
        processed["next_step"] = _field_entry(
            recovered_next_step["value"], recovered_next_step["quote"]
        )
        report.append(_report_item(
            "下一步行动", "提取", "next_step",
            "原始记录包含明确的我方后续跟进行动。",
            "已保留原文行动；不将其视为客户承诺。",
        ))
        short_log.append("⚠️ [next_step] 已保留原始记录中的我方跟进行动")
        rules_applied += 1

    # A decision maker must have authority evidence, not only a contact name.
    decision = processed.get("decision_maker", {})
    decision_quote = extract_quote(decision) if isinstance(decision, dict) else None
    if extract_value(decision) and not re.search(r"(拍板|最终决策|最终审批|董事长|总经理|院长|董事会|负责人)", str(decision_quote)):
        processed["decision_maker"] = _field_entry()
        report.append(_report_item(
            "角色边界", "拦截", "decision_maker",
            "原文只证明联系人或影响角色，未证明最终拍板权。", "已改为未确认。",
        ))
        short_log.append("⚠️ [decision_maker] 未见最终拍板权依据，已标为未确认")
        rules_applied += 1

    # Do not let an LLM turn a qualified-but-stated budget or an already
    # evidenced decision maker into a fabricated risk. This targets only
    # unsupported "not confirmed" claims; explicit risks stay untouched.
    risk_entry = processed.get("risk", {})
    risk_value = extract_value(risk_entry) if isinstance(risk_entry, dict) else None
    risk_quote = extract_quote(risk_entry) if isinstance(risk_entry, dict) else None
    risk_says_budget_unknown = bool(
        risk_value and re.search(r"预算.{0,8}(不确定|未确认|待确认)", risk_value)
    )
    risk_says_decision_unknown = bool(
        risk_value and re.search(r"决策人.{0,8}(不确定|未确认|待确认)", risk_value)
    )
    source_has_explicit_uncertainty = bool(
        re.search(r"(未确认|待确认|未批|未立项|尚未|没有预算)", str(risk_quote or ""))
    )
    if (
        (risk_says_budget_unknown and "budget" in qualified_fields)
        or (risk_says_decision_unknown and extract_value(processed.get("decision_maker", {})))
    ) and not source_has_explicit_uncertainty:
        processed["risk"] = _field_entry()
        report.append(_report_item(
            "风险边界", "拦截", "risk",
            "风险结论与已保留的预算范围或最终拍板依据不一致。",
            "已清空无原始风险依据的推断。",
        ))
        short_log.append("⚠️ [risk] 已清空与已确认字段矛盾的风险推断")
        rules_applied += 1

    # Rule 5: preserve both sides of a contradiction and never select one.
    conflicts = _normalized_conflicts(processed, source_text)
    for conflict in conflicts:
        for field in conflict["fields"]:
            processed[field] = _field_entry()
            # A field cannot be both preserved as a qualified value and left
            # unresolved due to contradictory evidence. Conflict takes priority.
            qualified_fields.pop(field, None)
        conflict_risk = "待核实：" + conflict["description"]
        existing_risk = extract_value(processed.get("risk", {}))
        processed["risk"] = _field_entry(
            f"{existing_risk}；{conflict_risk}" if existing_risk else conflict_risk,
            "\n".join(conflict["quotes"]),
        )
        report.append(_report_item(
            "矛盾处理", "待复核", "、".join(conflict["fields"]),
            "；".join(f"“{quote}”" for quote in conflict["quotes"]),
            "相关字段已标为未确认，并写入风险。",
        ))
        short_log.append("⚠️ [conflict] 检测到相互矛盾的原文，未选择任一版本")
        rules_applied += 1

    # Stage is never accepted from the model as-is.  It is calculated from
    # validated evidence in the exact S0–S5 order supplied by the exercise.
    evidence = _stage_evidence_from_source(processed, source_text)
    stage_value, stage_quote = "S0", None
    if any("stage" in conflict["fields"] for conflict in conflicts):
        stage_value = None
        report.append(_report_item(
            "阶段判定", "待复核", "stage",
            "阶段相关证据存在矛盾。", "未自动判定阶段。",
        ))
        short_log.append("⚠️ [stage] 阶段证据存在矛盾，已标为未确认")
        rules_applied += 1
    else:
        for evidence_key, candidate_stage in STAGE_EVIDENCE_ORDER:
            # S3 additionally requires that a need or scenario is still
            # evidenced; a bare budget mention cannot create a valid deal.
            if candidate_stage == "S3" and not evidence["need_or_scenario"]["met"]:
                continue
            if evidence[evidence_key]["met"]:
                stage_value = candidate_stage
                stage_quote = evidence[evidence_key]["quote"]
                break
        report.append(_report_item(
            "阶段判定", "通过", "stage",
            f"按题干达成条件计算为 {stage_value} — {STAGE_RULES[stage_value]['name']}。",
            "已覆盖模型候选阶段。",
        ))
        rules_applied += 1
    processed["stage"] = _field_entry(stage_value, stage_quote)
    processed["_stage_evidence"] = evidence

    # Rule 6: action / recommended owner / time is visible separately so a
    # missing time never gets silently implied by the model.
    plan = _build_next_step_plan(processed, source_text)
    processed["_next_step_plan"] = plan
    missing_plan_parts = []
    if not plan["action_confirmed"]:
        missing_plan_parts.append("动作")
    if not plan["owner_confirmed"]:
        missing_plan_parts.append("负责人")
    if not plan["time_stated"]:
        missing_plan_parts.append("时间")
    elif plan["time_precision_pending"]:
        missing_plan_parts.append("时间精确值")
        qualified_fields["next_step"] = "下一步时间为范围表达，具体日期待确认。"
    report.append(_report_item(
        "下一步行动", "待补充" if missing_plan_parts else "通过", "next_step",
        "、".join(missing_plan_parts) + "未在原始记录中明确或未精确约定。" if missing_plan_parts else "动作、负责人和时间均有原始依据。",
        "负责人显示为建议值；时间范围保留原文并明确精确日期待确认。" if missing_plan_parts else "保留原始约定。",
    ))
    rules_applied += 1

    processed["_qualified_fields"] = qualified_fields
    unconfirmed = build_unconfirmed_items(processed, conflicts)
    processed["_conflicts"] = conflicts
    processed["_unconfirmed_items"] = unconfirmed
    if unconfirmed:
        report.append(_report_item(
            "缺失信息", "待补充", None,
            f"共 {len(unconfirmed)} 项未确认或待核实信息。", "已生成可追问清单。",
        ))
        short_log.append(f"⚠️ [missing] 已生成 {len(unconfirmed)} 项未确认信息")
        rules_applied += 1
    else:
        report.append(_report_item(
            "缺失信息", "通过", None, "题目要求的 CRM 字段均有可确认依据。", "无需补充。",
        ))

    return processed, short_log[-RULE_LOG_LIMIT:], report, rules_applied


# ──────────────────────────────────────────────
# 完整性评分
# ──────────────────────────────────────────────

def compute_quality_score(data: dict) -> tuple[int, int, list[str]]:
    """计算信息完整度评分，返回 (得分, 总分, 缺失字段列表)"""
    fields_to_check = list(CRITICAL_FIELDS) + [
        "scenario", "budget", "influencer", "timeline", "risk", "next_step"
    ]
    total = len(fields_to_check)
    score = 0
    missing = []
    for f in fields_to_check:
        entry = data.get(f)
        if isinstance(entry, dict):
            val = entry.get("value")
            if val and "缺" not in str(val) and "null" not in str(val).lower():
                score += 1
            else:
                missing.append(f)
    return score, total, missing


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

ANALYSIS_RULES = [
    ("Rule 1", "事实边界", "非空事实字段必须有可追溯原文依据；无依据即标为未确认"),
    ("Rule 2", "推断处理", "可能、应该、感觉、我猜等不进入 CRM 事实字段"),
    ("Rule 3", "证据与角色", "预算、决策人、时间和阶段保留依据；决策人必须有拍板权证据"),
    ("Rule 4", "矛盾处理", "冲突原文并列为风险/待确认，系统不选择任一版本"),
    ("Rule 5", "阶段判定", "按 S0~S5 达成条件由 Python 计算，不直接采纳模型结论"),
    ("Rule 6", "下一步行动", "明确展示动作、建议负责人和时间；未约定时间标为待确认"),
]


@st.cache_resource
def _initialize_database() -> None:
    """Run idempotent schema setup once per Streamlit server process."""
    db.init_db()


def main():
    st.set_page_config(page_title="商机录入与分析助手", page_icon="📊", layout="wide")

    st.markdown("""
    <h1 style='font-size:28px'>📊 商机录入与分析助手</h1>
    <p style='color:#666;font-size:15px'>面向 FDE 岗位面试演示 · Human-in-the-loop 商机信息录入与治理闭环</p>
    """, unsafe_allow_html=True)

    # ── Initialize DB once per server process ─────────────────────────
    try:
        _initialize_database()
    except Exception as exc:
        st.error(f"⚠️ 数据库初始化失败: {exc}")
        st.stop()

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ 配置")
        use_mock = st.checkbox("使用 Mock 模式（不调用真实API）", value=True, help="开启后使用本地模拟数据，无需 API Key")
        if _get_managed_api_key():
            st.caption("真实 API 已由服务端安全配置。")
            api_key = ""
        else:
            api_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                value="",
                key="api_key_input",
                placeholder="仅当前浏览器会话使用",
                help="仅用于本地或自管部署；公开应用请在服务端配置 DEEPSEEK_API_KEY。",
            )
            if api_key:
                st.session_state._api_key = api_key.strip()

        st.divider()
        st.header("📋 核心分析规则")
        for label, title, desc in ANALYSIS_RULES:
            st.caption(f"**{label} · {title}**\n{desc}")

        with st.expander("查看销售阶段定义", expanded=False):
            for stage, rule in STAGE_RULES.items():
                st.caption(f"**{stage} · {rule['name']}**：{rule['condition']}")

        st.divider()
        st.header("💡 业务痛点")
        st.info("""
        **传统商机录入的问题：**
        - 销售人员手写笔记格式混乱，关键字段遗漏率高
        - CRM 录入依赖主观判断，缺乏一致性标准
        - 管理者无法快速识别信息缺失和矛盾之处
        - 商机质量参差不齐， pipeline 可信度低
        """)

    # ── State initialization ──────────────────────────────────────────
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "extracted" not in st.session_state:
        st.session_state.extracted = None
    if "rule_log" not in st.session_state:
        st.session_state.rule_log = []
    if "rule_report" not in st.session_state:
        st.session_state.rule_report = []
    if "critic" not in st.session_state:
        st.session_state.critic = None
    if "nextprompt" not in st.session_state:
        st.session_state.nextprompt = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "record_id" not in st.session_state:
        st.session_state.record_id = None
    if "archive_success" not in st.session_state:
        st.session_state.archive_success = False
    if "archive_dedup" not in st.session_state:
        st.session_state.archive_dedup = False

    st.session_state._use_mock = use_mock
    if api_key:
        st.session_state._api_key = api_key

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_enter, tab_review, tab_ledger, tab_dashboard = st.tabs(
        ["📝 商机录入", "✅ 人工复核", "📒 商机台账", "📊 管理层视图"]
    )

    with tab_enter:
        _render_entry_flow(use_mock)

    with tab_review:
        _render_review_page()

    with tab_ledger:
        _render_ledger()

    with tab_dashboard:
        _render_dashboard()


# ── Entry Flow: Steps 1-5 ──────────────────────────────────────────

def _render_entry_flow(use_mock: bool) -> None:
    """Render the core Human-in-the-loop entry flow."""

    left_col, right_col = st.columns([1, 1.3], gap="large")

    # ═══ LEFT COLUMN: Input ═══
    with left_col:
        st.subheader("🎤 Step 1 — 原始输入", divider="blue")

        input_type = st.selectbox(
            "输入类型",
            options=INPUT_TYPES,
            index=INPUT_TYPES.index("销售手工记录"),
            key="input_type_selector",
        )

        # Mock samples are an optional demo aid, never a required step in the
        # sales-entry flow.  Load the selected sample into the normal input.
        if use_mock:
            with st.expander("🧪 加载演示样例", expanded=False):
                preset_keys = list(PRESET_CASES.keys())
                selected_preset = st.selectbox(
                    "演示样例",
                    options=["--- 自定义输入 ---"] + preset_keys,
                    key="preset_selector",
                )
                if selected_preset != "--- 自定义输入 ---":
                    if st.session_state.get("_loaded_preset") != selected_preset:
                        st.session_state.transcript = PRESET_CASES[selected_preset]
                        st.session_state._loaded_preset = selected_preset
                        _invalidate_current_draft()
        else:
            st.session_state.pop("_loaded_preset", None)

        transcript = st.text_area(
            "粘贴销售拜访记录（支持换行）",
            height=300,
            placeholder="在此粘贴销售人员的原始拜访记录...",
            key="transcript",
            on_change=_invalidate_current_draft,
        )

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            run_clicked = st.button(
                "🚀 Step 2 — AI 识别并生成 Draft",
                type="primary",
                width="stretch",
                disabled=not transcript.strip() or st.session_state.processing,
            )
        with btn_col2:
            clear_clicked = st.button("🗑️ 清空", width="stretch")

        if clear_clicked:
            _clear_all_state()

        # Run processing (inline to avoid Streamlit widget ordering issues)
        if run_clicked and not st.session_state.processing:
            _do_run_analysis(transcript, input_type, use_mock)

        # Progress indicator
        _render_progress()

    # ═══ RIGHT COLUMN: AI Draft ═══
    with right_col:
        extracted = st.session_state.extracted
        if extracted is None:
            st.info("👈 请在左侧输入拜访记录，点击「AI 识别并生成 Draft」按钮")
        else:
            preprocessed_summary = st.session_state.get("preprocessed_summary")
            if preprocessed_summary:
                with st.expander("🧹 查看录音预处理摘要", expanded=False):
                    st.markdown(preprocessed_summary)
            _render_extraction_results()
            st.info("AI Draft 已生成。请切换到右侧的「✅ 人工复核」完成确认、修正与归档。")

    # Archive success message
    if st.session_state.archive_success:
        rid = st.session_state.record_id
        st.success(f"✅ 已成功归档！Record ID: `{rid}`。可在「📊 管理层视图」中查看。")
        st.session_state.archive_success = False  # Show once per cycle

    if st.session_state.archive_dedup:
        st.warning("⚠️ 该记录已被确认归档，请勿重复提交。")

    # Bottom meta
    st.divider()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mock_label = " (Mock)" if st.session_state.get("_use_mock", USE_MOCK) else ""
    st.caption(f"分析时间: {ts}{mock_label} | 流程: 输入 → AI Draft → 人工复核 → 归档")


def _invalidate_current_draft() -> None:
    """Discard the current draft when its source text changes."""
    st.session_state.step = 0
    st.session_state.extracted = None
    st.session_state.rule_log = []
    st.session_state.rule_report = []
    st.session_state.rules_applied = 0
    st.session_state.critic = None
    st.session_state.quality_score = (0, 10, [])
    st.session_state.nextprompt = ""
    st.session_state.preprocessed_summary = None
    st.session_state.processing = False
    st.session_state.record_id = None
    st.session_state.archive_success = False
    st.session_state.archive_dedup = False
    st.session_state.human_values = {}

    for key in list(st.session_state.keys()):
        if key.startswith("confirm_") or key.startswith("_confirm_"):
            del st.session_state[key]


def _clear_all_state() -> None:
    """Clear workflow state and return the entry flow to its initial state."""
    _invalidate_current_draft()
    st.rerun()


# ── Inline analysis run (avoid sub-function for Streamlit widget safety) ──

def _do_run_analysis(transcript: str, input_type: str, use_mock: bool) -> None:
    """Run full pipeline directly (called from within render flow)."""
    st.session_state.processing = True
    # Use progress bar rendered in this same render cycle
    progress_container = st.empty()

    preprocessed_summary = None
    if input_type in TRANSCRIPT_INPUT_TYPES:
        progress_container.progress(15, "正在整理录音转写中的事实与矛盾信息...")
        preprocessed_summary, preprocess_error = call_transcript_preprocessor(
            transcript, input_type, use_mock
        )
        if preprocess_error:
            st.warning("录音预处理未完成，已使用原始文本继续提取。")

    time.sleep(0.2)
    progress_container.progress(30, "正在提取商机字段...")
    time.sleep(0.2)

    try:
        extracted = call_extractor(transcript, use_mock, preprocessed_summary)

        if extracted.get("_api_error") or extracted.get("_parse_error"):
            error_summary = (
                "⚠️ AI 服务调用失败，请检查 API Key、模型和网络后重试"
                if extracted.get("_api_error")
                else "⚠️ AI 返回内容不是有效 JSON，请查看原始输出后重试"
            )
            st.session_state.extracted = extracted
            st.session_state.critic = {
                "overall_status": "REVIEW",
                "issues": [],
                "missing_critical_fields": [],
                "summary": error_summary,
            }
            st.session_state.step = 1
            st.session_state.processing = False
            progress_container.progress(100, "⚠️ AI 解析失败，可查看错误信息")
            return

        # Step 2: Rule Engine
        time.sleep(0.2)
        progress_container.progress(70, "运行规则引擎校验...")
        extracted, rule_log, rule_report, rules_applied = rule_engine(extracted, transcript)

        # Step 3: Critic
        time.sleep(0.2)
        progress_container.progress(90, "语义审核中...")
        critic = call_critic(extracted, use_mock)
        critic = reconcile_critic_with_rule_engine(critic, extracted)

        score, total, missing = compute_quality_score(extracted)
        nextprompt = _mock_nextprompt(score, total, missing, critic) if use_mock else call_nextprompts(extracted, critic, use_mock)

        record_id = str(uuid.uuid4())[:8]

        st.session_state.extracted = extracted
        st.session_state.rule_log = rule_log
        st.session_state.rule_report = rule_report
        st.session_state.rules_applied = rules_applied
        st.session_state.critic = critic
        st.session_state.quality_score = (score, total, missing)
        st.session_state.nextprompt = nextprompt
        st.session_state.record_id = record_id
        st.session_state.input_type = input_type
        st.session_state.raw_transcript = transcript
        st.session_state.preprocessed_summary = preprocessed_summary
        st.session_state.step = 4
        st.session_state.processing = False

        progress_container.progress(100, "分析完成 ✅")

    except Exception:
        st.session_state.processing = False
        progress_container.progress(100, "⚠️ 分析未完成")
        st.error("分析过程中发生异常，请检查输入后重试。若问题持续，请联系应用维护者。")


def _render_progress() -> None:
    """Render step progress indicators."""
    steps = [
        ("① 原始输入", "填写销售记录"),
        ("② AI 识别", "LLM 提取商机字段"),
        ("③ Rule Engine", "Python 规则引擎校验"),
        ("④ Critic", "LLM 语义审核"),
        ("⑤ 人工复核", "修改并归档"),
    ]

    current_step = st.session_state.step
    cols = st.columns(len(steps))
    for i, (step_name, step_desc) in enumerate(steps):
        with cols[i]:
            if i <= min(current_step, 4):
                icon = "✅" if i < current_step else "⏳"
                color = "#52c41a" if i < current_step else "#faad14"
            else:
                icon = "⬜"
                color = "#d9d9d9"
            st.markdown(
                f"<div style='text-align:center;color:{color};font-size:12px'>"
                f"{icon} {step_name}<br><span style='font-size:10px'>{step_desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Extraction Results (Step 2-3 output) ───────────────────────────

def _is_low_priority_lead(extracted: dict) -> bool:
    """Recognize an explicit no-go response without treating it as an AI failure."""
    if extract_value(extracted.get("stage", {})) != "S0":
        return False
    signals = " ".join(
        str(extract_value(extracted.get(field, {})) or "")
        for field in ("need", "risk", "next_step")
    )
    no_go_terms = ("够用", "不想", "暂不", "不需要", "无更换", "无改造", "无需求", "不跟进")
    return any(term in signals for term in no_go_terms)


def _render_extraction_results() -> None:
    """Display AI-extracted draft results with evidence and critic info."""
    extracted = st.session_state.extracted
    critic = st.session_state.critic
    quality = st.session_state.get("quality_score", (0, 10, []))
    score, total, missing = quality

    # Service errors and malformed outputs are different operational issues.
    # Show them before any quality score so an API failure is never presented
    # as a 0/10 business-analysis result.
    if extracted.get("_api_error"):
        st.error("🔴 AI 服务调用失败", icon="🔴")
        st.caption("请检查 DeepSeek API Key、Base URL、模型名称或账户额度，然后重试。")
        with st.expander("📦 查看服务返回信息"):
            st.code(extracted.get("_raw_response", ""), language="text")
        return

    if extracted.get("_parse_error"):
        st.error("🔴 AI 返回内容无法解析为有效 JSON", icon="🔴")
        st.caption("系统已要求 DeepSeek 使用 JSON 输出模式；若仍出现，请查看原始输出。")
        with st.expander("📦 查看 AI 原始输出"):
            st.code(extracted.get("_raw_response", ""), language="text")
        return

    # Alert banner
    alert_status = critic.get("overall_status", "PASS") if isinstance(critic, dict) else "REVIEW"
    source_conflicts = extracted.get("_conflicts", [])
    if alert_status == "FAIL":
        st.error("🚨 AI Draft 存在严重问题 — 请仔细审查以下结果", icon="🚨")
    if source_conflicts:
        st.error("🚨 原始商机信息存在关键冲突 — 必须人工复核", icon="🚨")
    elif _is_low_priority_lead(extracted):
        st.info("ℹ️ 低优先级线索 — 客户当前暂无明确改造意愿，建议暂不推进并保留后续跟踪。")
    elif alert_status == "REVIEW":
        st.warning(f"⚠️ AI Draft 存在待确认事项 — 完整度 {score}/{total}", icon="⚠️")
    elif score < total * 0.6:
        st.warning(f"⚠️ AI Draft 信息完整度较低 — {score}/{total}", icon="⚠️")
    else:
        st.success(f"✅ AI 已完成结构化提取 — 完整度 {score}/{total}", icon="✅")

    # Field display with Evidence toggle
    display_fields = {
        "customer_name": "客户名称",
        "need": "客户需求",
        "scenario": "核心场景",
        "budget": "预算",
        "decision_maker": "决策人",
        "influencer": "影响人",
        "timeline": "时间计划",
        "stage": "商机阶段",
        "risk": "风险",
        "next_step": "下一步行动",
    }

    for field_en, field_cn in display_fields.items():
        entry = extracted.get(field_en, {})
        val = extract_value(entry) if isinstance(entry, dict) else None
        quote = extract_quote(entry) if isinstance(entry, dict) else None

        col_v, col_q = st.columns([2, 1])
        with col_v:
            st.markdown(f"**{field_cn}**")
            if val:
                st.caption(str(val))
            else:
                st.caption("未确认")
        with col_q:
            if quote:
                with st.expander("📎 查看 AI 分析依据", expanded=False):
                    st.caption(quote)
            elif field_en == "stage" and val == "S0":
                st.caption("系统依据：原始记录未出现明确需求或场景")
            else:
                st.caption("(无)")

    # The exercise asks for explicit missing-information output rather than
    # scattered empty fields.  This list is generated by the Python rules,
    # not by model prose.
    unconfirmed_items = extracted.get("_unconfirmed_items", [])
    if unconfirmed_items:
        with st.expander("❓ 未确认信息 / 建议追问", expanded=True):
            st.caption("以下项目缺少原始依据，或存在互相矛盾的表述；均未被写入确定事实。")
            for item in unconfirmed_items:
                st.markdown(f"**{item['label']}**：{item['reason']}")
                st.caption(f"建议追问：{item['question']}")
    else:
        st.success("✅ 未发现需要补充或待核实的 CRM 信息。")

    # Next steps are deliberately split into an action, a recommended owner
    # and a time.  "建议" / "待确认" prevents the system from fabricating a
    # responsible person or date when the source omitted it.
    next_step_plan = extracted.get("_next_step_plan", {})
    if isinstance(next_step_plan, dict):
        with st.expander("📌 下一步行动拆解", expanded=True):
            st.markdown(f"- **动作**：{next_step_plan.get('action', '待确认下一步行动')}")
            owner_label = "负责人" if next_step_plan.get("owner_confirmed") else "建议负责人"
            st.markdown(f"- **{owner_label}**：{next_step_plan.get('owner', '我方销售（建议）')}")
            st.markdown(f"- **时间**：{next_step_plan.get('time', '待确认')}")

    rule_report = st.session_state.get("rule_report", [])
    if rule_report:
        with st.expander("🛡️ 题目规则执行报告", expanded=False):
            st.caption("最终字段已经过 Python 规则校验；AI 的抽取结果不是直接入库的事实。")
            st.dataframe(
                [
                    {
                        "规则": item["rule"],
                        "状态": item["status"],
                        "字段": item["field"],
                        "检查结果": item["result"],
                        "系统动作": item["action"],
                    }
                    for item in rule_report
                ],
                hide_index=True,
                width="stretch",
            )

    # Critic summary in expander
    if isinstance(critic, dict):
        with st.expander("🔍 AI 自我审查报告", expanded=False):
            status_colors = {"PASS": "green", "REVIEW": "orange", "FAIL": "red"}
            c_color = status_colors.get(critic.get("overall_status", "PASS"), "gray")
            st.markdown(f"Overall: **{critic.get('overall_status', '?')}**")
            st.caption(critic.get("summary", ""))

            issues = critic.get("issues", [])
            if issues:
                for issue in issues:
                    sev = issue.get("severity", "low")
                    sev_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    st.markdown(
                        f"{sev_icons.get(sev, '⚪')} **{issue.get('type', '?')}** — "
                        f"{issue.get('description', '')}"
                    )
                    if issue.get("affected_fields"):
                        st.caption(f"涉及字段: {', '.join(issue['affected_fields'])}")
            else:
                st.success("未发现质量问题")

            missing_cf = critic.get("missing_critical_fields", [])
            if missing_cf:
                st.warning(f"关键字段缺失: {', '.join(missing_cf)}")

    # Next prompts
    if st.session_state.nextprompt:
        with st.expander("💬 AI 追问建议", expanded=False):
            st.info(st.session_state.nextprompt)

    # Raw JSON for debug
    with st.expander("🔧 调试信息 — 完整 AI Draft JSON", expanded=False):
        st.code(json.dumps(extracted, ensure_ascii=False, indent=2), language="json")


# ── Dashboard (Step 5 / Management View) ───────────────────────────

FIELD_LABELS = {
    "customer_name": ("客户名称", False),
    "need": ("客户需求", True),
    "scenario": ("核心场景", True),
    "budget": ("预算", False),
    "decision_maker": ("决策人", False),
    "influencer": ("影响人", False),
    "timeline": ("时间计划", False),
    "stage": ("商机阶段", False),
    "risk": ("风险", True),
    "next_step": ("下一步行动", True),
}
SHORT_FORM_FIELDS = ["customer_name", "budget", "decision_maker", "influencer", "timeline", "stage"]
LONG_FORM_FIELDS = ["need", "scenario", "risk", "next_step"]


def _as_nullable(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in ("", "null", "none", "n/a", "未知", "未确认") else text


def _human_widget_key(prefix: str, record_id: str, field: str) -> str:
    return f"{prefix}_{record_id}_{field}"


def _init_human_widget_values(prefix: str, record_id: str, values: dict) -> None:
    """Set AI-derived defaults once, without overwriting same-record edits."""
    initialized_key = f"_{prefix}_initialized_{record_id}"
    active_record_key = f"_{prefix}_active_record"
    widget_keys = [_human_widget_key(prefix, record_id, field) for field in OPPORTUNITY_FIELDS]
    # Streamlit discards widget state when a ledger record is not rendered.
    # Rehydrate it from SQLite when the user switches back, but never replace
    # live fields for the record that is currently being edited.
    if (
        st.session_state.get(initialized_key)
        and st.session_state.get(active_record_key) == record_id
        and all(
        widget_key in st.session_state for widget_key in widget_keys
        )
    ):
        return
    for field in OPPORTUNITY_FIELDS:
        value = _as_nullable(values.get(field))
        if field == "stage" and value not in VALID_STAGES:
            value = None
        st.session_state[_human_widget_key(prefix, record_id, field)] = value or ""
    st.session_state[initialized_key] = True
    st.session_state[active_record_key] = record_id


def _stage_format(value: str) -> str:
    return "未确认" if not value else f"{value} — {STAGE_DESC.get(value, value)}"


def _render_human_field(prefix: str, record_id: str, field: str, ai_value: str | None = None) -> dict:
    label, is_textarea = FIELD_LABELS[field]
    if ai_value is not None:
        st.caption(f"AI Draft: {ai_value}" if ai_value else "AI Draft: 未识别")
    widget_key = _human_widget_key(prefix, record_id, field)
    if field == "stage":
        value = st.selectbox(
            label,
            options=[""] + VALID_STAGE_LIST,
            format_func=_stage_format,
            key=widget_key,
        )
    elif is_textarea:
        value = st.text_area(label, height=90, key=widget_key)
    else:
        value = st.text_input(label, key=widget_key)
    return {"value": _as_nullable(value)}


def _render_human_fields(prefix: str, record_id: str, ai_values: dict | None = None) -> dict:
    """Use two columns for short fields and full width for long text."""
    human_fields = {}
    for left, right in zip(SHORT_FORM_FIELDS[::2], SHORT_FORM_FIELDS[1::2]):
        left_col, right_col = st.columns(2)
        with left_col:
            human_fields[left] = _render_human_field(
                prefix, record_id, left, (ai_values or {}).get(left)
            )
        with right_col:
            human_fields[right] = _render_human_field(
                prefix, record_id, right, (ai_values or {}).get(right)
            )
    for field in LONG_FORM_FIELDS:
        human_fields[field] = _render_human_field(
            prefix, record_id, field, (ai_values or {}).get(field)
        )
    return human_fields


def _render_review_page() -> None:
    """Render the human review sub-page for the current AI draft."""
    extracted = st.session_state.get("extracted")
    if extracted is None:
        st.info("请先在「📝 商机录入」输入记录并生成 AI Draft。")
        return
    _render_confirmation_form(extracted)


def _render_confirmation_form(extracted: dict) -> None:
    """Render the editable human confirmation form inside one st.form."""
    if st.session_state.archive_dedup:
        return
    record_id = st.session_state.record_id
    if extracted.get("_parse_error"):
        st.warning("⚠️ AI 解析失败，无法进入确认环节。请重试或开启 Mock 模式。")
        return
    if not record_id:
        st.error("内部错误：缺少 record_id，请重新运行分析。")
        return
    if db.get_record_by_id(record_id):
        st.session_state.archive_dedup = True
        st.warning("⚠️ 该记录已被确认归档，请勿重复提交。")
        return

    ai_values = {
        field: extract_value(extracted.get(field, {})) for field in OPPORTUNITY_FIELDS
    }
    _init_human_widget_values("confirm", record_id, ai_values)
    st.subheader("✅ 人工复核 / 修正", divider="violet")
    st.caption("AI 提取结果已预填；你可以修改、补充，或清空不正确的值。")
    with st.form(f"confirm_form_{record_id}"):
        human_fields = _render_human_fields("confirm", record_id, ai_values)
        archive_clicked = st.form_submit_button(
            "✅ Step 4 — 确认并归档", type="primary", width="stretch"
        )
    if archive_clicked:
        _handle_archive(extracted, record_id, human_fields)


def _handle_archive(extracted: dict, record_id: str, human_fields: dict) -> None:
    """Archive a record atomically after a final duplicate check."""
    if db.get_record_by_id(record_id):
        st.session_state.archive_dedup = True
        st.rerun()

    evidence = []
    for field in OPPORTUNITY_FIELDS:
        entry = extracted.get(field, {})
        quote = extract_quote(entry) if isinstance(entry, dict) else None
        if quote:
            evidence.append({
                "field": field,
                "value": extract_value(entry),
                "quote": quote,
            })
    try:
        db.archive_opportunity(
            record_id=record_id,
            raw_text=st.session_state.get("raw_transcript", ""),
            input_type=st.session_state.get("input_type", "其他文本"),
            ai_fields=extracted,
            evidence_json=evidence,
            human_fields=human_fields,
            change_types=infer_change_types(extracted, human_fields),
        )
    except Exception as exc:
        st.error(f"⚠️ 归档失败，未写入半成品数据：{type(exc).__name__}: {exc}")
        return

    st.session_state.archive_success = True
    st.session_state.archive_dedup = True
    st.rerun()


def _format_minute(timestamp: str | None) -> str:
    """Format ISO-like audit timestamps for the ledger's minute-level display."""
    if not timestamp:
        return "—"
    return str(timestamp).replace("T", " ")[:16]


def _delete_confirmed_opportunity(record_id: str) -> None:
    """Delete through the DB API, with a compatibility fallback for stale modules."""
    delete_fn = getattr(db, "delete_opportunity", None)
    if callable(delete_fn):
        delete_fn(record_id)
        return

    # The table is already known to be local SQLite because get_conn() is used
    # for reading the same ledger.  This branch is only for a stale module.
    conn = db.get_conn()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM records WHERE record_id=?", (record_id,))
            if cursor.rowcount != 1:
                raise ValueError(f"opportunity not found: {record_id}")
    finally:
        conn.close()


def _render_ledger() -> None:
    """List and edit only confirmed opportunities."""
    st.subheader("📒 商机台账")
    try:
        records = db.get_confirmed_records()
    except Exception as exc:
        st.error(f"⚠️ 台账读取失败: {exc}")
        return
    if not records:
        st.info("暂无已确认商机。请先在“商机录入”完成确认归档。")
        return

    table_data = [
        {
            "record_id": record["record_id"],
            "客户名称": record.get("customer_name_human"),
            "商机阶段": record.get("stage_human"),
            "预算": record.get("budget_human"),
            "决策人": record.get("decision_maker_human"),
            "时间计划": record.get("timeline_human"),
            "confirmed_at": _format_minute(record.get("confirmed_at")),
        }
        for record in records
    ]

    record_ids = [record["record_id"] for record in records]
    if st.session_state.get("ledger_selected_record") not in record_ids:
        st.session_state.ledger_selected_record = record_ids[0]

    st.caption("点击或双击表格中的任意单元格，可自动切换下方的查看与修改对象。")
    table_event = st.dataframe(
        table_data,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="ledger_table",
    )
    selected_rows = list(getattr(getattr(table_event, "selection", None), "rows", []) or [])
    if selected_rows:
        selected_row_index = selected_rows[0]
        if 0 <= selected_row_index < len(table_data):
            table_selected_id = table_data[selected_row_index]["record_id"]
            if table_selected_id != st.session_state.get("_ledger_last_table_selection"):
                st.session_state.ledger_selected_record = table_selected_id
                st.session_state._ledger_last_table_selection = table_selected_id
        else:
            # Streamlit can retain a row index from before a deletion.  It no
            # longer identifies a ledger record, so ignore it safely.
            st.session_state.pop("_ledger_last_table_selection", None)

    selected_id = st.selectbox("选择商机查看或修改", record_ids, key="ledger_selected_record")
    selected = db.get_record_by_id(selected_id)
    if not selected:
        st.error("所选商机不存在或尚未确认。")
        return

    st.caption(
        f"确认时间：{_format_minute(selected.get('confirmed_at'))} ｜ "
        f"最后更新：{_format_minute(selected.get('updated_at'))}"
    )
    defaults = {field: selected.get(f"{field}_human") for field in OPPORTUNITY_FIELDS}
    ai_values = {field: selected.get(f"{field}_ai") for field in OPPORTUNITY_FIELDS}
    _init_human_widget_values("ledger", selected_id, defaults)
    with st.form(f"ledger_edit_{selected_id}"):
        edited_fields = _render_human_fields("ledger", selected_id, ai_values)
        save_clicked = st.form_submit_button("保存修改", type="primary", width="stretch")
    if save_clicked:
        ai_fields = {field: {"value": ai_values[field]} for field in OPPORTUNITY_FIELDS}
        try:
            db.update_human_confirmation(
                selected_id,
                edited_fields,
                infer_change_types(ai_fields, edited_fields),
            )
        except Exception as exc:
            st.error(f"⚠️ 保存失败: {type(exc).__name__}: {exc}")
            return
        st.success("已更新 Human Confirmed Result；AI Draft 和原始记录未修改。")
        st.rerun()

    with st.expander("🗑️ 删除商机", expanded=False):
        st.warning("删除后将同时移除原始记录、AI Draft 与人工确认结果，且无法恢复。")
        delete_confirmed = st.checkbox(
            f"我确认删除商机 {selected_id}", key=f"ledger_delete_confirm_{selected_id}"
        )
        delete_clicked = st.button(
            "🗑️ 删除该商机",
            type="secondary",
            disabled=not delete_confirmed,
            width="stretch",
        )
    if delete_clicked:
        try:
            _delete_confirmed_opportunity(selected_id)
        except Exception as exc:
            st.error(f"⚠️ 删除失败: {type(exc).__name__}: {exc}")
            return
        st.session_state.pop("ledger_selected_record", None)
        st.session_state.pop("_ledger_last_table_selection", None)
        st.session_state.pop(f"ledger_delete_confirm_{selected_id}", None)
        st.success("已删除该商机及其关联数据。")
        st.rerun()


def _has_management_value(value: object) -> bool:
    """Whether a human-confirmed value is usable in management reporting."""
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "null", "none", "未确认", "未知", "—")


def _render_dashboard() -> None:
    """Render an actionable management view from Human Confirmed data only."""
    st.subheader("📊 管理层视图")
    st.caption("业务漏斗、数据可信度与需管理介入的商机；所有业务统计仅基于 Human Confirmed Result。")

    try:
        records = db.get_confirmed_records()
    except Exception as exc:
        st.error(f"⚠️ 管理数据读取失败: {exc}")
        return
    if not records:
        st.info("暂无已确认商机。请先完成至少一条人工复核并归档。")
        return

    total = len(records)
    stage_rank = {stage: index for index, stage in enumerate(VALID_STAGE_LIST)}
    stage_groups = {
        "早期探索（S0–S1）": {"stages": {"S0", "S1"}, "color": "🔴"},
        "方案推进（S2–S3）": {"stages": {"S2", "S3"}, "color": "🟡"},
        "成交确认（S4–S5）": {"stages": {"S4", "S5"}, "color": "🟢"},
    }
    group_counts = {
        label: sum(1 for record in records if record.get("stage_human") in meta["stages"])
        for label, meta in stage_groups.items()
    }
    week_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    week_start -= timedelta(days=week_start.weekday())
    weekly_new = 0
    for record in records:
        try:
            if datetime.fromisoformat(str(record.get("confirmed_at"))) >= week_start:
                weekly_new += 1
        except (TypeError, ValueError):
            continue

    # Layer 1 — business pipeline
    st.markdown("### 业务基本盘")
    headline_left, headline_right = st.columns(2)
    headline_left.metric("已归档商机", total)
    headline_right.metric("本周新增归档", weekly_new)
    funnel_columns = st.columns(3)
    for column, (label, meta) in zip(funnel_columns, stage_groups.items()):
        count = group_counts[label]
        with column:
            st.markdown(f"**{meta['color']} {label}**")
            st.progress(count / total if total else 0)
            st.caption(f"{count} 条 · {count / total:.0%}")

    # Layer 2 — trustworthy data and management gaps
    critical_fields = ("budget", "decision_maker")
    key_complete = sum(
        all(_has_management_value(record.get(f"{field}_human")) for field in critical_fields)
        for record in records
    )
    human_intervened = sum(
        any(
            record.get("change_types", {}).get(field)
            in (CHANGE_HUMAN_ADDED, CHANGE_HUMAN_CORRECTED)
            for field in OPPORTUNITY_FIELDS
        )
        for record in records
    )
    blind_spots = sum(
        any(not _has_management_value(record.get(f"{field}_human")) for field in critical_fields)
        for record in records
    )

    st.markdown("### 数据可信度与业务盲区")
    quality_columns = st.columns(3)
    quality_columns[0].metric("关键要素完备率", f"{key_complete / total:.0%}", f"{key_complete}/{total} 预算与决策人齐全")
    quality_columns[1].metric("人工校正覆盖率", f"{human_intervened / total:.0%}", f"{human_intervened}/{total} 条经人工补充或修正")
    quality_columns[2].metric("业务盲区率", f"{blind_spots / total:.0%}", f"{blind_spots}/{total} 条仍缺预算或决策人")

    critical_change_counts = {
        "AI Draft 经人工确认": 0,
        "人工补充 / 修正": 0,
        "人工复核后仍不明确": 0,
    }
    for record in records:
        for field in critical_fields:
            change_type = record.get("change_types", {}).get(field)
            human_value = record.get(f"{field}_human")
            if not _has_management_value(human_value):
                critical_change_counts["人工复核后仍不明确"] += 1
            elif change_type == CHANGE_AI_CONFIRMED:
                critical_change_counts["AI Draft 经人工确认"] += 1
            else:
                critical_change_counts["人工补充 / 修正"] += 1
    st.caption(
        "关键字段（预算、决策人）口径："
        + " ｜ ".join(f"{label} {count}" for label, count in critical_change_counts.items())
    )
    with st.expander("查看数据口径", expanded=False):
        st.markdown(
            "- **业务漏斗**：仅使用人工确认后的商机阶段。\n"
            "- **AI Draft 经人工确认**：AI 建议值与人工确认值一致。\n"
            "- **人工补充 / 修正**：人工补全 AI 未识别字段，或修正 AI Draft。\n"
            "- **业务盲区**：人工复核后，预算或最终决策人仍未确认。\n"
            "- 现有数据不保存“销售最初填报值”，因此不将任何指标表述为“拦截销售夸大”。"
        )

    # Layer 3 — manager action list
    st.markdown("### 需管理介入的高风险商机")
    risk_rows = []
    for record in records:
        stage = record.get("stage_human")
        if stage_rank.get(stage, -1) < stage_rank["S2"]:
            continue
        missing = [
            FIELD_LABELS[field][0]
            for field in critical_fields
            if not _has_management_value(record.get(f"{field}_human"))
        ]
        if not missing:
            continue
        risk_rows.append({
            "客户名称": record.get("customer_name_human") or "未确认",
            "当前阶段": stage,
            "待补关键信息": "、".join(missing),
            "下一步行动": record.get("next_step_human") or "未确认",
            "确认时间": _format_minute(record.get("confirmed_at")),
            "record_id": record["record_id"],
        })
    if risk_rows:
        st.warning(f"共有 {len(risk_rows)} 条 S2 及以上商机缺少预算或最终决策人，建议优先跟进。")
        st.dataframe(risk_rows, width="stretch", hide_index=True)
    else:
        st.success("S2 及以上商机的预算与最终决策人信息均已确认。")


# ──────────────────────────────────────────────
# Mock 追问话术（保持不变）
# ──────────────────────────────────────────────

def _mock_nextprompt(score: int, total: int, missing: list, critic: dict) -> str:
    """Mock 追问话术生成"""
    parts = []
    if "budget" in missing:
        parts.append("你可以问问：\"咱们这个项目大概的预算范围是多少？有没有已经获批的金额？\"")
    if "decision_maker" in missing:
        parts.append("记得搞清楚：\"这个项目的最终拍板人是哪位？还有谁会参与决策？\"")
    if "customer_name" in missing:
        parts.append("先把客户全名和公司背景搞清楚，\"方便我这边准备更有针对性的方案。\"")
    if "timeline" in missing:
        parts.append("问一句时间节点：\"您期望什么时候能看到系统上线？我们好提前排期。\"")
    if score <= 1:
        parts.insert(0, "这次沟通的信息量太少了，建议你补充了解一下客户的基本情况和需求背景后再做深入交流。")
    if critic and isinstance(critic, dict) and critic.get("overall_status") == "FAIL":
        parts.append("另外注意到信息存在一些矛盾之处，建议在下次沟通前先整理一轮已有的信息，确保口径一致。")

    if not parts:
        return "当前商机信息较为完整！继续保持沟通节奏，重点关注方案落地细节即可。"

    coach_voice = "💡 作为你的销售教练，我建议你在下次跟进时可以这样问：\n\n" + "\n".join(f"- {p}" for p in parts)
    return coach_voice


if __name__ == "__main__":
    main()
