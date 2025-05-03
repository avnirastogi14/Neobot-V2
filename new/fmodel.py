from transformers import pipeline
import re, logging, time, random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ml_recognition.log"), logging.StreamHandler()],
)
logger = logging.getLogger("fmodel")

# ────────────────────────────  INTENTS  ────────────────────────────
INTENTS_LIST = [
    # team / member management
    "assign_role",
    "update_team",
    "show_team_info",
    "remove_member",
    "list_teams",
    "get_member_info",
    "create_team",
    # help / small-talk
    "help",
    "greeting",
    # NEW:
    "create_role",               # ⇦ add server role
]

ROLE_KEYWORDS = [
    "developer", "lead", "manager", "designer", "architect", "tester", "qa",
    "frontend", "backend", "fullstack", "devops", "product owner", "scrum master",
    "head", "director", "engineer", "analyst", "admin", "coordinator", "ui",
    "ux", "project manager", "technical writer",
]

try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    ner = pipeline("ner", grouped_entities=True)
    logger.info("ML models loaded successfully")
except Exception as e:
    logger.error(f"Error loading ML models: {e}")
    classifier = lambda x, candidate_labels: {"labels": ["create_role"], "scores": [0.9]}
    ner = lambda x: []
    logger.warning("Using dummy pipelines")

# ────────────────────────────  Helpers  ────────────────────────────
def extract_entities(ner_results, text):
    """Extract entities for *both* team/member and role creation logic."""
    entities = {"name": None, "role": None, "team_name": None, "repo": None,
                "role_name": None, "colour": None}

    # — existing name/role/team extraction —
    person_ents = [e for e in ner_results if e.get("entity_group") == "PER"]
    if person_ents:
        entities["name"] = " ".join(e["word"] for e in person_ents)

    # team name with regex
    for pattern in [
        r"(?i)\bteam\s+([A-Za-z][\w\-]*)",
        r"(?i)\bin\s+([A-Za-z][\w\-]*)\s+team\b",
        r"(?i)\bfor\s+team\s+([A-Za-z][\w\-]*)",
    ]:
        if (m := re.search(pattern, text)):
            entities["team_name"] = m.group(1)
            break

    # repo URLs
    if (m := re.search(r"https?://(?:github|gitlab|bitbucket)\.com/\S+", text)):
        entities["repo"] = m.group(0)

    # roles from ROLE_KEYWORDS
    for kw in ROLE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text, re.I):
            full_role = re.search(rf"(?i)(senior|junior|lead|principal|chief)?\s*{kw}", text)
            if full_role:
                entities["role"] = full_role.group(0).strip()
                break

    # — NEW: server-role name & colour —
    # role name: word(s) after "role", excluding “in team …”
    if (m := re.search(r"(?i)(?:add|create|make)\s+(?:a\s+)?role\s+([\w\s\-]{2,30})", text)):
        entities["role_name"] = m.group(1).strip()

    # colour: hex like #ff8800
    if (m := re.search(r"#?[0-9a-fA-F]{6}", text)):
        entities["colour"] = m.group(0)

    # sanity-strip whitespace
    entities = {k: (v.strip() if isinstance(v, str) else v) for k, v in entities.items() if v}
    return entities


def estimate_intent_confidence(scores):
    if not scores:
        return "low"
    s = scores[0]
    return "high" if s > 0.8 else "medium" if s > 0.5 else "low"


# ────────────────────────────  MAIN PREDICT  ────────────────────────────
def predict(user_input: str):
    """Return dict: intent, entities, confidence, processing_time."""
    t0 = time.time()
    cleaned = user_input.strip()

    # quick rules for help & greeting
    if re.search(r"(?i)\b(help|what can you do|how to use)\b", cleaned):
        return {"intent": "help", "entities": {}, "confidence": "high", "processing_time": time.time() - t0}

    if re.fullmatch(r"(?i)(hi|hello|hey|good (morning|afternoon|evening))", cleaned):
        return {"intent": "greeting", "entities": {}, "confidence": "high", "processing_time": time.time() - t0}

    # rule-based detection for create_role – avoids expensive classifier for a very common phrase
    if re.search(r"(?i)\b(add|create|make)\s+(?:a\s+)?role\b", cleaned):
        ents = extract_entities([], cleaned)
        return {"intent": "create_role", "entities": ents, "confidence": "high", "processing_time": time.time() - t0}

    # classification
    result = classifier(cleaned, candidate_labels=INTENTS_LIST)
    intent = result["labels"][0]
    conf = estimate_intent_confidence(result["scores"])
    ner_results = ner(cleaned)
    ents = extract_entities(ner_results, cleaned)

    return {
        "intent": intent,
        "entities": ents,
        "confidence": conf,
        "processing_time": time.time() - t0,
    }


# ────────────────────────────  Dev quick-test  ────────────────────────────
if __name__ == "__main__":
    tests = [
        "Add a role Designer to the server",
        "Create role Moderator with colour #00ff7f",
        "Assign John as backend developer in team Apollo",
        "Create a new team Apollo",
    ]
    for txt in tests:
        print(predict(txt))
