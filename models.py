"""
FDE Demo — Canonical Opportunity Schema & constants.

All core modules (app, database, dashboard) reference these so that:
  LLM output == Human UI == SQLite == Dashboard
use consistent field names.
"""

# ── Canonical fields -------------------------------------------------
# Each field is stored in DB as both ai_{field} and human_{field}.
# NULL means "not known" — never use strings like "未确认" or "N/A" in DB.

OPPORTUNITY_FIELDS = [
    "customer_name",
    "need",
    "scenario",
    "budget",
    "decision_maker",
    "influencer",
    "timeline",
    "stage",
    "risk",
    "next_step",
]

# Critical fields used for completeness scoring
CRITICAL_FIELDS = {"customer_name", "need", "decision_maker", "stage"}

# Sales-stage source of truth.  These are the exact definitions supplied by
# the exercise; UI labels, prompts, Python validation and regression tests all
# derive from this single configuration.
STAGE_RULES = {
    "S0": {
        "name": "线索",
        "condition": "只有初步接触，无明确需求。",
    },
    "S1": {
        "name": "需求初探",
        "condition": "明确至少一个业务问题或使用场景。",
    },
    "S2": {
        "name": "方案验证",
        "condition": "客户明确同意演示、试用、技术交流或方案评估。",
    },
    "S3": {
        "name": "商务评估",
        "condition": "已讨论预算、报价、采购流程或合同条款之一，且需求仍有效。",
    },
    "S4": {
        "name": "决策审批",
        "condition": "明确进入内部立项、审批或供应商决策。",
    },
    "S5": {
        "name": "赢单/签约",
        "condition": "已签合同或正式订单已确认。",
    },
}

# Keep the ordered list separate from membership checks: UI rendering must not
# depend on unordered set iteration.
VALID_STAGE_LIST = list(STAGE_RULES)
VALID_STAGES = set(VALID_STAGE_LIST)
STAGE_DESC = {
    stage: f"{rule['name']}（{rule['condition']}）"
    for stage, rule in STAGE_RULES.items()
}

# Change type constants
CHANGE_AI_CONFIRMED = "ai_confirmed"       # AI had value, Human kept it
CHANGE_HUMAN_CORRECTED = "human_corrected"  # AI had value, Human changed it
CHANGE_HUMAN_ADDED = "human_added"          # AI was NULL, Human provided value
CHANGE_MISSING = "missing"                  # Both AI and Human are NULL


def infer_change_types(ai_fields: dict, human_fields: dict) -> dict:
    """Classify each human-confirmed value relative to the AI draft."""
    def normalize(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() in ("", "null", "none") else text

    change_types = {}
    for field in OPPORTUNITY_FIELDS:
        ai_entry = ai_fields.get(field, {}) if isinstance(ai_fields, dict) else {}
        human_entry = human_fields.get(field, {}) if isinstance(human_fields, dict) else {}
        ai_value = normalize(ai_entry.get("value") if isinstance(ai_entry, dict) else ai_entry)
        human_value = normalize(
            human_entry.get("value") if isinstance(human_entry, dict) else human_entry
        )
        if ai_value and human_value:
            change_types[field] = (
                CHANGE_AI_CONFIRMED if ai_value == human_value else CHANGE_HUMAN_CORRECTED
            )
        elif ai_value:
            change_types[field] = CHANGE_HUMAN_CORRECTED
        elif human_value:
            change_types[field] = CHANGE_HUMAN_ADDED
        else:
            change_types[field] = CHANGE_MISSING
    return change_types


# ── Extractor output helpers ------------------------------------------

def extract_value(entry: dict) -> object:
    """Return the 'value' from an extractor field entry, or None."""
    if not isinstance(entry, dict):
        return None
    v = entry.get("value")
    if v is not None and str(v).strip() != "" and str(v).lower() not in ("null", "none", ""):
        return str(v).strip()
    return None


def extract_quote(entry: dict) -> object:
    """Return the 'quote' from an extractor field entry, or None."""
    if not isinstance(entry, dict):
        return None
    q = entry.get("quote")
    if q is not None and str(q).strip() != "" and str(q).lower() not in ("null", "none", ""):
        return str(q).strip()
    return None


def clean_field_entry(entry: dict) -> dict:
    """Normalize a single field entry: ensure value/quote exist."""
    if not isinstance(entry, dict):
        return {"value": None, "quote": None}
    return {
        "value": extract_value(entry),
        "quote": extract_quote(entry),
    }
