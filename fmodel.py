from transformers import pipeline
import re
import logging
import time
import string
from typing import Dict, List, Any, Tuple, Optional
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fmodel")

INTENTS_LIST = [
    "assign_role",
    "update_team_repo",
    "update_team_members",
    "update_team_status",
    "update_team_role",
    "show_team_info",
    "remove_member",
    "list_teams",
    "get_member_info",
    "help",
    "greeting",
    "create_team",
    "delete_team"
]

INTENT_DESCRIPTIONS = {
    "list_teams": [
        "list all teams", "show teams", "display teams", "what teams do we have",
        "show all teams", "give me the teams", "list the teams", "show all the teams",
        "display all teams", "tell me the teams", "what are all the teams",
        "what teams exist", "teams list"
    ],
    "create_team": [
        "create a new team", "add a team", "make a team", "establish a team",
        "set up a team", "create team", "add team", "form a team", "build a team",
        "start a team", "new team"
    ],
    "delete_team": [
        "delete a team", "remove a team", "disband team", "eliminate team",
        "dissolve team", "deactivate team"
    ],
    "get_member_info": [
        "show information for person", "display info for member", "tell me about person",
        "what role does person have", "member information", "who is person",
        "information about person", "details on member"
    ],
    "show_team_info": [
        "show team information", "display team details", "get team info",
        "what is the team's status", "team details", "info about team"
    ],
    "assign_role": [
        "assign a role to someone", "give someone a position", "set someone's role",
        "allocate a role", "promote member to role"
    ],
    "update_team_repo": [
        "change the team's repository", "set the team's repo",
        "modify the team's code location", "update repo"
    ],
    "update_team_members": [
        "add members to the team", "remove members from the team",
        "change team membership", "modify team members"
    ],
    "update_team_status": [
        "change the team's status", "set the team to active",
        "mark the team as completed", "modify team status"
    ],
    "update_team_role": [
        "change the team's overall role", "set the team's function",
        "modify team purpose", "update team role"
    ],
    "remove_member": [
        "remove someone from the team", "delete a member",
        "kick someone off the team", "exclude member"
    ]
}

ROLE_KEYWORDS = [
    "developer", "lead", "manager", "designer", "architect",
    "tester", "qa", "frontend", "backend", "fullstack",
    "devops", "product owner", "scrum master", "head",
    "director", "engineer", "analyst", "admin", "coordinator",
    "ui", "ux", "project manager", "technical writer"
]

STATUS_KEYWORDS = [
    "active", "inactive", "on hold", "completed",
    "planning", "in progress", "pending", "archived",
    "paused", "delayed", "blocked", "complete"
]

# Load models with error handling
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    ner = pipeline("ner", grouped_entities=True)
    logger.info("ML models loaded successfully")
except Exception as e:
    logger.error(f"Error loading ML models: {e}")
    try:
        classifier = pipeline("zero-shot-classification")
        ner = pipeline("ner")
        logger.warning("Using fallback ML models")
    except Exception as e:
        logger.critical(f"Critical error loading fallback models: {e}")

        def dummy_classifier(text, candidate_labels, hypothesis_template=None):
            return {"labels": INTENTS_LIST, "scores": [0.1] * len(INTENTS_LIST)}

        def dummy_ner(text):
            return []

        classifier = dummy_classifier
        ner = dummy_ner
        logger.critical("Using dummy ML functions")

def preprocess_text(text: str) -> str:
    """Clean and standardize input text."""
    text = text.strip().lower()
    text = re.sub(r"[,.!?;]", " ", text) # Remove punctuation
    return " ".join(text.split()) # Normalize whitespace

def extract_team_name(text: str) -> Optional[str]:
    """Extract team name using regex patterns."""

    patterns = [
        r"(?:team|for team|in team|of team|to team)\s+(?P<team_name>[A-Za-z0-9_.-]+)",
        r"(?:team)\s+\"(?P<team_name>[^\"]+)\"",
        r"(?:team)\s+'(?P<team_name>[^']+)'",
        r"team\s+(?P<team_name>[A-Z][a-zA-Z0-9_.-]+)",
        r"(?:create|add|make|establish|set up)\s+(?:a\s+)?(?:new\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)",
        r"(?:delete|remove|disband)\s+team\s+(?P<team_name>[A-Za-z0-9_.-]+)",
        r"(?:update|change|set)\s+(?:the\s+)?(?:team|it)(?:\s+\w+)?\s+(?:to|as)\s+(?P<team_name>[A-Za-z0-9_.-]+)", # Added for status
        r"(?:show\s+(?:team\s+)?)?(?:\"(?P<team_name_quoted>[^\"]+)\"|(?P<team_name_simple>[A-Za-z0-9_.-]+))", # Added for show
        r"(?P<team_name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", # Capitalized words
        r"(?P<team_name>[A-Za-z0-9_.-]+)" # Alphanumeric
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group("team_name") or match.group("team_name_quoted") or match.group("team_name_simple")

    return None

def extract_members(text: str) -> Optional[List[str]]:
    """Extract team members from text."""

    patterns = [
    r"(?:members are|members to add are|add members)\s+(?P<members>.+?)(?:\.|\band\b|\|to\)",
    r"members\s*\:\s*(?P<members>.+?)(?:\.|\band\b|\|)",
    r"(?:with members|consisting of|comprised of)\s+(?P<members>.+?)(?:\.|\band\b|\|)",
    r"(?:update\|change)\s+members\s+(?:of\|for\|to)\s+(?P<members>.+?)(?:\.|\band\b|\|)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            members_text = match.group("members")
            return [m.strip() for m in re.split(r',|\sand\s|&', members_text) if m.strip()]

    return None

def extract_status(text: str) -> Optional[str]:
    """Extract team status from text."""

    patterns = [
        r"(?:status|state)\s+(?:to|as|of|is)\s+(?P<status>" + "|".join(map(re.escape, STATUS_KEYWORDS)) + ")",
        r"(?P<status>" + "|".join(map(re.escape, STATUS_KEYWORDS)) + ")\s+(?:status|state)",
        r"(?:set|mark|change|update)\s+(?:the\s+)?(?:team|it)(?:\s+\w+)?\s+(?:to|as)\s+(?P<status>" + "|".join(map(re.escape, STATUS_KEYWORDS)) + ")",
        r"(?:update|change|set)\s+team\s+[A-Za-z0-9_.-]+\s+status\s+to\s+(?P<status_free>[A-Za-z\s]+)",
        r"(?:update|change|set)\s+status\s+of\s+team\s+[A-Za-z0-9_.-]+\s+to\s+(?P<status_free>[A-Za-z\s]+)",
        r"(?:update|change|set)\s+the\s+status\s+for\s+team\s+[A-Za-z0-9_.-]+\s+to\s+(?P<status_free>[A-Za-z\s]+)",
        r"(?:update|change|set)\s+team\s+[A-Za-z0-9_.-]+\s+to\s+(?P<status_free>[A-Za-z\s]+)\s+status",
        r"(?:update|change|set)\s+status\s+to\s+(?P<status_free>[A-Za-z\s]+)\s+for\s+team\s+[A-Za-z0-9_.-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group("status") or match.group("status_free")

    return None

def extract_repo(text: str) -> Optional[str]:
    """Extract repository URL from text."""

    patterns = [
        r'https?://\S+', # Direct URL
        r"(?:repo|repository|link)\s+(?:is|to|as|of)\s+(?P<repo>https?://\S+)",
        r"(?:repo|repository|link):\s*(?P<repo>https?://\S+)",
        r"(?:update|change|set)\s+(?:the\s+)?repo(?:sitory)?\s+(?:to|as|of)\s+(?P<repo>https?://\S+)",
        r"(?:add|assign)\s+(?:a\s+)?repo(?:sitory)?\s+(?P<repo>https?://\S+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group("repo")

    return None

def extract_role(text: str) -> Optional[str]:
    """Extract role information from text."""

    patterns = [
        r"(?:as|to be|to|is|as a|as an|a|an)\s+(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")",
        r"(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")\s+(?:role|position|title)",
        r"role\s+(?:of|as|to)\s+(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")",
        r"(?:promote|assign)\s+[A-Za-z]+\s+(?:to|as)\s+(?P<role_free>[a-zA-Z\s]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group("role") or match.group("role_free")

    return None

def extract_person_name(text: str, ner_results: List[Dict]) -> Optional[str]:
    """Extract person name from text using NER and patterns."""

    # 1. NER results (highest priority)
    for ent in ner_results:
        if ent["entity_group"] == "PER":
            return ent["word"].strip()

    # 2. Name patterns
    patterns = [
        r"(?:assign|make|set|give|promote)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:as|to be|to|the role of|to)\s+",
        r"(?:add|assign|make)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:to|as a member of|as)",
        r"(?:remove|delete)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:from)",
        r"(?:information|info|details)\s+(?:for|about|of|on)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:who is|show|get|display)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:'s|s)?\s+(?:role|position|info|information)",
        r"(?:role|position|info|information)\s+(?:of|for|about)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:what|which)\s+(?:role|position|title)\s+(?:does|is|has)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"promote\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+to\s+"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group("name").strip()

    # 3. Simple capitalized name in short queries (last resort)
    words = text.split()
    if len(words) <= 4:
        for word in words:
            if re.match(r"^[A-Z][a-z]+$", word) and word.lower() not in ["team", "info", "role", "show", "get", "promote", "to"]:
                return word

    return None

def extract_entities(text: str, ner_results: List[Dict]) -> Dict[str, Any]:
    """Extract all entities from text using multiple methods."""

    entities = {
        "name": extract_person_name(text, ner_results),
        "role": extract_role(text),
        "team_name": extract_team_name(text),
        "repo": extract_repo(text),
        "members": extract_members(text),
        "status": extract_status(text)
    }

    return {k: v for k, v in entities.items() if v is not None} # Filter out None

def enhanced_intent_classification(text: str) -> Tuple[str, float]:
    """Enhance intent classification using semantic patterns and zero-shot."""

    cleaned_text = preprocess_text(text)

    intent_patterns = [
        ("list_teams", [
            r"(?i)(show|list|display|give|what|get)\s+(all|the|all the|)\s*(teams|team)",
            r"(?i)(show\s+all\s+teams)",
            r"(?i)(list\s+all\s+teams )",
            r"(?i)(show\s+all\s+the\s+teams )",
            r"(?i)(list\s+all\s+the\s+teams )",
            r"(?i)(what|which)\s+(teams|team)\s+(do we have|exist|are there)",
            r"(?i)(what|which)\s+(teams|team)\s+(are\s+there)",
            r"(?i)teams\s+(list|show)",
            r"(?i)all\s+teams",
            r"(?i)teams" # Short query
        ]),
        ("create_team", [
            r"(?i)(create|add|make|establish|set up)\s+(a\s+)?(new\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)(create|add|make|establish|set up)\s+team",
            r"(?i)new\s+team"
        ]),
        ("delete_team", [
            r"(?i)(delete|remove|disband|eliminate|dissolve|deactivate)\s+(a\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)(delete|remove|disband|eliminate|dissolve|deactivate)\s+team"
        ]),
        ("get_member_info", [
            r"(?i)(show|display|get)\s+(information|info|details)\s+(for|about|of|on)\s+(?P<name>[A-Za-z]+)",
            r"(?i)(what|which)\s+(role|position|title)\s+(does|is|has)\s+(?P<name>[A-Za-z]+)",
            r"(?i)(member|person)\s+information\s+(for|about|of|on)\s+(?P<name>[A-Za-z]+)",
            r"(?i)who is\s+(?P<name>[A-Za-z]+)",
            r"(?i)tell me about\s+(?P<name>[A-Za-z]+)",
            r"(?i)info\s+(for|about|of|on)\s+(?P<name>[A-Za-z]+)", # Short query
            r"(?i)(show|get|display)\s+(info|information|details)\s+(for|about|of)\s+([A-Za-z]+)"
        ]),
       ("show_team_info", [
            r"(?i)(show|list|display|get)\s+(all\s+)?teams?",
            r"(?i)(what|which)\s+teams?",
            r"(?i)teams\s+(list|show|display)?",
            r"(?i)(show|display|get)\s+(team\s+)?information\s+(for\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)(what is|show|display|get)\s+(the\s+)?(team's|team)\s+(status|details|info)\s+(for\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)team\s+(information|details|info)\s+(for\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)(team\s+)?status\s+(for\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)team\s+info\s+(for\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)(team's|team)\s+(status|details|info)\s+(for\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)(show|display|get)\s+(team\s+)?info", # Short query
            r"(?i)(what is|show|display|get)\s+(the\s+)?team's\s+status", # Short query
            r"(?i)team\s+information", # Short query
            r"(?i)team\s+details", # Short query
            r"(?i)team\s+info" # Short query
        ]),
        ("assign_role", [
            r"(?i)(assign|give|set|allocate|promote)\s+(a\s+)?role\s+(to|for)\s+(?P<name>[A-Za-z]+)\s+(?:in\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)?\s+(?:as|to be|to|is)\s+(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")",
            r"(?i)(assign|give|set|allocate|promote)\s+(?P<name>[A-Za-z]+)\s+(?:to|as)\s+(a\s+)?(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")\s+(?:in\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)?",
            r"(?i)(assign|give|set|allocate|promote)\s+(?P<name>[A-Za-z]+)\s+(a\s+)?(?P<role_free>[a-zA-Z\s]+)\s+(?:role|position|title)\s+(?:in\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)?",
            r"(?i)(assign|give|set|allocate|promote)\s+(?P<name>[A-Za-z]+)\s+(?:in\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)?\s+(?:to|as)\s+(a\s+)?(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")",
            r"(?i)(assign|give|set|allocate|promote)\s+(a\s+)?role\s+(to|for)\s+(?P<name>[A-Za-z]+)", # Short query
            r"(?i)(assign|give|set|allocate|promote)\s+(?P<name>[A-Za-z]+)\s+(?:to|as)\s+(a\s+)?(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")", # Short query
            r"(?i)(assign|give|set|allocate|promote)\s+(?P<name>[A-Za-z]+)\s+(a\s+)?(?P<role_free>[a-zA-Z\s]+)\s+(?:role|position|title)" # Short query
        ]),
        ("update_team_repo", [
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(repo(?:sitory)?|code location|link)\s+(to|as|is)\s+(?P<repo>https?://\S+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(repo(?:sitory)?|code location|link)\s+(?P<repo>https?://\S+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?repo(?:sitory)?\s+(of\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as|is)\s+(?P<repo>https?://\S+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?repo(?:sitory)?\s+(of\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)\s+(?P<repo>https?://\S+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as|is)\s+(?P<repo>https?://\S+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(?P<repo>https?://\S+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?repo(?:sitory)?\s+(to|as|is)\s+(?P<repo>https?://\S+)", # Short query
            r"(?i)(change|set|modify|update)\s+(the\s+)?repo(?:sitory)?\s+(?P<repo>https?://\S+)" # Short query
        ]),
        ("update_team_members", [
            r"(?i)(add|remove|change|modify|update)\s+(the\s+)?members\s+(of\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as|with)\s+(?P<members>.+)",
            r"(?i)(add|remove|change|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(members|membership)\s+(to|as|with)\s+(?P<members>.+)",
            r"(?i)(add|remove|change|modify|update)\s+(the\s+)?members\s+(to|as|with)\s+(?P<members>.+)\s+(of\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)(add|remove|change|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as|with)\s+(?P<members>.+)",
            r"(?i)(add|remove|change|modify|update)\s+(members)\s+(to|as|with)\s+(?P<members>.+)", # Short query
            r"(?i)(add|remove|change|modify|update)\s+(members)\s+(of\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)" # Short query
        ]),
        ("update_team_status", [
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+status\s+(to|as)\s+(?P<status>" + "|".join(map(re.escape, STATUS_KEYWORDS)) + ")",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+status\s+(to|as)\s+(?P<status_free>[a-zA-Z\s]+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?status\s+(of\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as)\s+(?P<status>" + "|".join(map(re.escape, STATUS_KEYWORDS)) + ")",
            r"(?i)(change|set|modify|update)\s+(the\s+)?status\s+(of\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as)\s+(?P<status_free>[a-zA-Z\s]+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as)\s+(?P<status>" + "|".join(map(re.escape, STATUS_KEYWORDS)) + ")",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as)\s+(?P<status_free>[a-zA-Z\s]+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(?P<status>" + "|".join(map(re.escape, STATUS_KEYWORDS)) + ")",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(?P<status_free>[a-zA-Z\s]+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?status\s+(to|as)\s+(?P<status>" + "|".join(map(re.escape, STATUS_KEYWORDS)) + ")", # Short query
            r"(?i)(change|set|modify|update)\s+(the\s+)?status\s+(to|as)\s+(?P<status_free>[a-zA-Z\s]+)", # Short query
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+status", # Short query
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(?P<status_keyword>active|inactive|on hold|completed|planning|in progress|pending|archived|paused|delayed|blocked|complete)",
            r"(?i)(change|set|modify|update)\s+status\s+of\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+to\s+(?P<status_keyword>active|inactive|on hold|completed|planning|in progress|pending|archived|paused|delayed|blocked|complete)"
        ]),
        ("update_team_role", [
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+role\s+(to|as)\s+(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+role\s+(to|as)\s+(?P<role_free>[a-zA-Z\s]+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?role\s+(of\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as)\s+(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")",
            r"(?i)(change|set|modify|update)\s+(the\s+)?role\s+(of\s+team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as)\s+(?P<role_free>[a-zA-Z\s]+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as)\s+(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(to|as)\s+(?P<role_free>[a-zA-Z\s]+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")",
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+(?P<role_free>[a-zA-Z\s]+)",
            r"(?i)(change|set|modify|update)\s+(the\s+)?role\s+(to|as)\s+(?P<role>" + "|".join(map(re.escape, ROLE_KEYWORDS)) + ")", # Short query
            r"(?i)(change|set|modify|update)\s+(the\s+)?role\s+(to|as)\s+(?P<role_free>[a-zA-Z\s]+)", # Short query
            r"(?i)(change|set|modify|update)\s+(the\s+)?team\s+(?P<team_name>[A-Za-z0-9_.-]+)\s+role" # Short query
        ]),
        ("remove_member", [
            r"(?i)(remove|delete|kick out|exclude)\s+(?P<name>[A-Za-z]+)\s+from\s+(?:team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)(remove|delete|kick out|exclude)\s+(?P<name>[A-Za-z]+)\s+from\s+the\s+team",
            r"(?i)(remove|delete|kick out|exclude)\s+member\s+(?P<name>[A-Za-z]+)\s+from\s+(?:team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)",
            r"(?i)(remove|delete|kick out|exclude)\s+member\s+(?P<name>[A-Za-z]+)\s+from\s+the\s+team",
            r"(?i)(remove|delete|kick out|exclude)\s+from\s+(?:team\s+)?(?P<team_name>[A-Za-z0-9_.-]+)\s+(?P<name>[A-Za-z]+)",
            r"(?i)(remove|delete|kick out|exclude)\s+(?P<name>[A-Za-z]+)\s+from\s+team", # Short query
            r"(?i)remove\s+(?P<name>[A-Za-z]+)", # Very short query
            r"(?i)delete\s+(?P<name>[A-Za-z]+)"  # Very short query
        ]),
        ("help", [
            r"(?i)help",
            r"(?i)what can you do",
            r"(?i)commands",
            r"(?i)what are the commands"
        ]),
        ("greeting", [
            r"(?i)hello",
            r"(?i)hi",
            r"(?i)hey",
            r"(?i)greetings"
        ])
    ]

    for intent, patterns in intent_patterns:
        for pattern in patterns:
            match = re.search(pattern, cleaned_text)
            if match:
                logger.info(f"Intent '{intent}' matched with pattern: '{pattern}' for text: '{cleaned_text}'")
                return intent, 0.95 # High confidence for pattern match

    # Fallback to zero-shot classification if no pattern matches
    try:
        start_time = time.time()
        zero_shot_result = classifier(cleaned_text, candidate_labels=INTENTS_LIST, hypothesis_template="The user wants to {}.")
        predicted_intent = zero_shot_result['labels'][0]
        confidence = zero_shot_result['scores'][0]
        end_time = time.time()
        logger.info(f"Zero-shot classification predicted intent: '{predicted_intent}' with confidence: {confidence:.2f} for text: '{cleaned_text}' (took {end_time - start_time:.2f} seconds)")
        return predicted_intent, confidence
    except Exception as e:
        logger.error(f"Error during zero-shot classification: {e}")
        return "unknown", 0.0

def predict(text: str) -> Tuple[str, Dict[str, Any]]:
    """Predict intent and extract entities from the input text."""
    cleaned_text = preprocess_text(text)
    intent, confidence = enhanced_intent_classification(cleaned_text)
    ner_results = []
    try:
        if classifier != dummy_classifier and ner != dummy_ner:
            ner_results = ner(cleaned_text)
            logger.info(f"NER results for '{cleaned_text}': {ner_results}")
    except Exception as e:
        logger.warning(f"Error during NER: {e}")

    entities = extract_entities(cleaned_text, ner_results)

    logger.info(f"Predicted intent: '{intent}' with confidence: {confidence:.2f}, extracted entities: {entities} for text: '{cleaned_text}'")
    return intent, entities

if __name__ == '__main__':
    test_commands = [
        "list all teams",
        "create a new team Project Phoenix",
        "delete team Alpha",
        "show information for Alice",
        "what is the status of team Beta",
        "assign the role of lead to Bob in team Gamma",
        "change the team Delta's repository to https://github.com/example/delta",
        "add members Carol and David to the team Epsilon",
        "update the status of team Zeta to in progress",
        "change the team Eta's role to administrator",
        "remove Frank from team Theta",
        "help me",
        "hello bot",
        "update team Omega status to completed",
        "set status of team Sigma to on hold",
        "change status to planning for team Lambda"
    ]

    for cmd in test_commands:
        intent, entities = predict(cmd)
        print(f"Command: '{cmd}' -> Intent: '{intent}', Entities: {entities}")