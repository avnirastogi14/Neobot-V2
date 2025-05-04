from transformers import pipeline
import re
import logging
import time
import string
from typing import Dict, List, Any, Tuple, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
    "assign_role",          # For assigning roles to team members
    "update_team_repo",     # For updating team repository
    "update_team_members",  # For updating the list of team members
    "update_team_status",   # For updating the status of a team
    "update_team_role",     # For updating the overall role of a team
    "show_team_info",       # For showing details of a team
    "remove_member",        # For removing team members
    "list_teams",           # For listing all teams
    "get_member_info",      # For getting info about a specific member
    "help",                 # For help requests
    "greeting",             # For greetings/small talk
    "create_team",          # For creating a new team
    "delete_team"           # For deleting a team
]

# Enhanced intent descriptions for better zero-shot classification
INTENT_DESCRIPTIONS = {
    "list_teams": [
        "list all teams",
        "show teams",
        "display teams",
        "what teams do we have",
        "show all teams",
        "give me the teams",
        "list the teams",
        "show all the teams",
        "display all teams",
        "tell me the teams",
        "what are all the teams",
        "what teams exist",
        "teams list"
    ],
    "create_team": [
        "create a new team",
        "add a team",
        "make a team",
        "establish a team",
        "set up a team",
        "create team",
        "add team",
        "form a team",
        "build a team",
        "start a team",
        "new team"
    ],
    "delete_team": [
        "delete a team",
        "remove a team",
        "disband team",
        "eliminate team",
        "dissolve team",
        "deactivate team"
    ],
    "get_member_info": [
        "show information for person",
        "display info for member",
        "tell me about person",
        "what role does person have",
        "member information",
        "who is person",
        "information about person",
        "details on member"
    ],
    "show_team_info": [
        "show team information",
        "display team details",
        "get team info",
        "what is the team's status",
        "team details",
        "info about team"
    ],
    "assign_role": [
        "assign a role to someone",
        "give someone a position",
        "set someone's role",
        "allocate a role",
        "promote member to role"
    ],
    "update_team_repo": [
        "change the team's repository",
        "set the team's repo",
        "modify the team's code location",
        "update repo"
    ],
    "update_team_members": [
        "add members to the team",
        "remove members from the team",
        "change team membership",
        "modify team members"
    ],
    "update_team_status": [
        "change the team's status",
        "set the team to active",
        "mark the team as completed",
        "modify team status"
    ],
    "update_team_role": [
        "change the team's overall role",
        "set the team's function",
        "modify team purpose",
        "update team role"
    ],
    "remove_member": [
        "remove someone from the team",
        "delete a member",
        "kick someone off the team",
        "exclude member"
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
    """Clean and standardize input text for better processing"""
    # Convert to lowercase and remove extra whitespace
    text = text.strip().lower()

    # Remove punctuation that isn't meaningful for our context
    for c in ",.!?;":
        text = text.replace(c, " ")

    # Normalize whitespace
    text = " ".join(text.split())

    return text

def extract_team_name(text: str) -> Optional[str]:
    """Extract team name using various patterns"""
    # Direct team name matching with various patterns
    team_patterns = [
    r"(?i)(?:team|for team|in team|of team|to team)\s+([A-Za-z0-9_.-]+)",
    r"(?i)(?:team)\s+\"([^\"]+)\"",
    r"(?i)(?:team)\s+'([^']+)'",
    r"(?i)team\s+([A-Z][a-zA-Z0-9_.-]+)",
    r"(?i)(?:create|add|make|establish|set up)\s+(?:a\s+)?(?:new\s+)?team\s+([A-Za-z0-9_.-]+)",
    r"(?i)(?:delete|remove|disband)\s+team\s+([A-Za-z0-9_.-]+)"
    ]

    for pattern in team_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Look for capitalized words that might be team names
    words = text.split()
    for i, word in enumerate(words):
        if i < len(words) - 1 and word.lower() == "team" and words[i+1][0].isupper():
            return words[i+1]
        if word[0].isupper() and len(word) > 1 and "team" in text.lower():
            potential_name = word.strip(string.punctuation)
            if potential_name:
                return potential_name

    return None

def extract_members(text: str) -> Optional[List[str]]:
    """Extract team members from text"""
    members_patterns = [
        r"(?i)(?:members are|members to add are|add members)\s+(.+?)(?:\.|\band\b|$|to)",
        r"(?i)members\s*:\s*(.+?)(?:\.|\band\b|$)",
        r"(?i)(?:with members|consisting of|comprised of)\s+(.+?)(?:\.|\band\b|$)",
        r"(?i)(?:update|change)\s+members\s+(?:of|for|to)\s+(.+?)(?:\.|\band\b|$)"
    ]

    for pattern in members_patterns:
        match = re.search(pattern, text)
        if match:
            members_text = match.group(1)
            # Split by common separators (comma, and, &)
            members = re.split(r',|\sand\s|&', members_text)
            return [m.strip() for m in members if m.strip()]

    return None

def extract_status(text: str) -> Optional[str]:
    """Extract team status from text"""
    for status in STATUS_KEYWORDS:
        pattern = rf"(?i)(?:status|state)\s+(?:to|as|of|is)\s+{re.escape(status)}"
        if re.search(pattern, text):
            return status

        pattern = rf"(?i){re.escape(status)}\s+(?:status|state)"
        if re.search(pattern, text):
            return status

        # Direct status mention with team context
        pattern = rf"(?i)(?:set|mark|change|update)\s+(?:the\s+)?(?:team|it)(?:\s+\w+)?\s+(?:to|as)\s+{re.escape(status)}"
        if re.search(pattern, text):
            return status

    return None

def extract_repo(text: str) -> Optional[str]:
    """Extract repository URL from text"""
    # First try to find URL pattern
    url_match = re.search(r'https?://\S+', text)
    if url_match:
        return url_match.group(0)

    # Try to find repo mentions
    repo_patterns = [
        r"(?i)(?:repo|repository|link)\s+(?:is|to|as|of)\s+(https?://\S+)",
        r"(?i)(?:repo|repository|link):\s*(https?://\S+)",
        r"(?i)(?:update|change|set)\s+(?:the\s+)?repo(?:sitory)?\s+(?:to|as|of)\s+(https?://\S+)",
        r"(?i)(?:add|assign)\s+(?:a\s+)?repo(?:sitory)?\s+(https?://\S+)"
    ]

    for pattern in repo_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return None

def extract_role(text: str) -> Optional[str]:
    """Extract role information from text"""
    # Check for specific role patterns
    for role in ROLE_KEYWORDS:
        patterns = [
            rf"(?i)(?:as|to be|to|is|as a|as an|a|an)\s+{re.escape(role)}",
            rf"(?i){re.escape(role)}\s+(?:role|position|title)",
            rf"(?i)role\s+(?:of|as|to)\s+{re.escape(role)}"
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return role

    return None

def extract_person_name(text: str, ner_results: List[Dict]) -> Optional[str]:
    """Extract person name from text using NER and patterns"""
    # First check NER results
    for ent in ner_results:
        if ent["entity_group"] == "PER":
            return ent["word"].strip()

    # Check for name patterns in various contexts
    name_patterns = [
        r"(?i)(?:assign|make|set|give|promote)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:as|to be|to|the role of|to)\s+",
        r"(?i)(?:add|assign|make)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:to|as a member of|as)",
        r"(?i)(?:remove|delete)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:from)",
        r"(?i)(?:information|info|details)\s+(?:for|about|of|on)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?i)(?:who is|show|get|display)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:'s|s)?\s+(?:role|position|info|information)",
        r"(?i)(?:role|position|info|information)\s+(?:of|for|about)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?i)(?:what|which)\s+(?:role|position|title)\s+(?:does|is|has)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?i)promote\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+to\s+"
    ]

    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Check for simple capitalized name in short queries
    words = text.split()
    if len(words) <= 4:
        for word in words:
            # Look for capitalized words that might be names
            if re.match(r"^[A-Z][a-z]+$", word) and word.lower() not in ["team", "info", "role", "show", "get", "promote", "to"]:
                return word

    return None

def extract_entities(text: str, ner_results: List[Dict]) -> Dict[str, Any]:
    """Extract all entities from text using multiple methods"""
    entities = {
        "name": None,
        "role": None,
        "team_name": None,
        "repo": None,
        "members": None,
        "status": None
    }

    # Extract person name
    entities["name"] = extract_person_name(text, ner_results)

    # Extract team name
    entities["team_name"] = extract_team_name(text)

    # Extract role
    entities["role"] = extract_role(text)

    # Extract repository
    entities["repo"] = extract_repo(text)

    # Extract members
    members = extract_members(text)
    if members:
        entities["members"] = ", ".join(members)

    # Extract status
    entities["status"] = extract_status(text)

    # Filter out None values
    return {k: v for k, v in entities.items() if v is not None}

def enhanced_intent_classification(text: str) -> Tuple[str, float]:
    """Enhance intent classification using semantic patterns and zero-shot"""
    cleaned_text = preprocess_text(text)

    # Check for direct and expanded list teams patterns
    list_teams_patterns = [
        r"(?i)(show|list|display|give|what|get)\s+(all|the|all the|)\s*(teams|team)",
        r"(?i)(show\s+all\s+teams)",
        r"(?i)(list\s+all\s+teams )",
        r"(?i)(show\s+all\s+the\s+teams )",
        r"(?i)(list\s+all\s+the\s+teams )",
        r"(?i)(what|which)\s+(teams|team)\s+(do we have|exist|are there)",
        r"(?i)(teams|team)\s+(list|listing)",
        r"(?i)^teams$",
    ]
    for pattern in list_teams_patterns:
        if re.search(pattern, cleaned_text):
            return "list_teams", 0.95

    # Check for create team patterns
    create_team_patterns = [
        r"(?i)(create|add|make|establish|set up|build|form|insert|initialise)\s+(a\s+)?(new\s+)?(team)",
        r"(?i)(new team)",
    ]

    for pattern in create_team_patterns:
        if re.search(pattern, cleaned_text):
            return "create_team", 0.92

    # Check for delete team patterns
    delete_team_patterns = [
        r"(?i)(delete|remove|disband|dissolve|eliminate)\s+(a\s+)?(team)",
    ]
    for pattern in delete_team_patterns:
        if re.search(pattern, cleaned_text):
            return "delete_team", 0.91

    # Check for show team info patterns
    team_info_patterns = [
        r"(?i)(show|get|display|what is|tell me about)\s+(information|info|details|team)\s+(for|about|of)\s+(team\s+)?([A-Za-z]+)",
        r"(?i)(team)\s+([A-Za-z]+)\s+(information|info|details)",
        r"(?i)(what|which|who)\s+(is|are)\s+(in|the members of|part of)\s+(team\s+)?([A-Za-z]+)",
        r"(?i)(show)+([A-Za-z]+)+(info)",
        r"(?i)(show|tell me|what is)\s+(team)\s+([A-Za-z]+)"
    ]
    for pattern in team_info_patterns:
        if re.search(pattern, cleaned_text):
            return "show_team_info", 0.9

    # Check for assign role patterns, including "promote"
    assign_role_patterns = [
    r"(?i)(add|assign|make).+(role|as|to be).+(in|to|for)\s+team",
    r"(?i)(set)\s+([A-Za-z]+).+as.+(in|to)\s+team",
    r"(?i)promote\s+([A-Za-z]+)\s+to\s+([a-zA-Z\s]+)(?:\s+in\s+team)?",
    r"(?i)assign\s+([A-Za-z]+)\s+([a-zA-Z\s]+)\s+as\s+([a-zA-Z\s]+)(?:\s+in\s+team)?", 
    r"(?i)give\s+([A-Za-z]+)\s+the\s+role\s+of\s+([a-zA-Z\s]+)(?:\s+in\s+team)?"
    ]
    
    for pattern in assign_role_patterns:
        if re.search(pattern, cleaned_text):
            return "assign_role", 0.88

    # Check for update team repo patterns
    update_repo_patterns = [
        r"(?i)(update|change|set).+(repo|repository).+(to|as)",
    ]
    for pattern in update_repo_patterns:
        if re.search(pattern, cleaned_text):
            return "update_team_repo", 0.87

    # Check for update team members patterns
    update_members_patterns = [
        r"(?i)(update|change|add|set).+(members|member list).+(to|as|of)",
        r"(?i)(update|add)+(team)+([A-Za-z]+)\s+(member)",
        r"(?i)(put)+([A-Za-z]+)\s+(to)+(team)+([A-Za-z]+)\s",
        r"(?i)(update|change|set)\s+(?:team\s+)?([A-Za-z0-9_.-]+)\s+status\s+(?:to|as|of)\s+([A-Za-z\s]+)",
        r"(?i)(update|change|set)\s+status\s+of\s+(?:team\s+)?([A-Za-z0-9_.-]+)\s+to\s+([A-Za-z\s]+)",
        r"(?i)(update|change|set)\s+(?:the\s+)?status\s+for\s+(?:team\s+)?([A-Za-z0-9_.-]+)\s+to\s+([A-Za-z\s]+)"
    ]
    for pattern in update_members_patterns:
        if re.search(pattern, cleaned_text):
            return "update_team_members", 0.93

    # Check for update team status patterns
    update_status_patterns = [
    r"(?i)(update|change|set).+(status|state).+(to|as|of)",
    r"(?i)(update).+([A-Za-z]+)\s+(status)"
    ]

    for pattern in update_status_patterns:
        if re.search(pattern, cleaned_text):
            return "update_team_status", 0.87

    # Check for remove member patterns
    remove_member_patterns = [
        r"(?i)(remove|delete).+(from team|from the team)",
    ]
    for pattern in remove_member_patterns:
        if re.search(pattern, cleaned_text):
            return "remove_member", 0.87

    # Check for other direct matches
    if re.search(r"(?i)(!\s*help|how to use|what can you do|help me|help|bothelp|!bothelp)", cleaned_text):
        return "help", 0.95

    if re.search(r"(?i)(!\s*ping|are you there|ping)", cleaned_text):
        return "ping", 0.95

    if re.search(r"(?i)(!\s*exit|quit|bye|exit)", cleaned_text):
        return "exit", 0.95

    if re.search(r"(?i)(hello|hi|hey|greetings|good morning|good afternoon)", cleaned_text) and len(cleaned_text.split()) < 5:
        return "greeting", 0.9

    # Check for single entity mentions that imply an intent
    words = cleaned_text.split()
    if len(words) <= 3:
        # Simple entity mentions
        if re.search(r"(?i)^([A-Z][a-z]+)$", text):  # Single capitalized word
            # Likely a person or team name
            return "get_member_info", 0.75
        elif re.search(r"(?i)^team\s+([A-Z][a-z]+)$", text):  # Just "team X"
            return "show_team_info", 0.85

    # If no direct matches, use zero-shot classification with an improved hypothesis template
    hypothesis_template = "This text is asking to {}"
    result = classifier(cleaned_text, INTENTS_LIST, hypothesis_template=hypothesis_template)
    return result["labels"][0], result["scores"][0]

def estimate_intent_confidence(score: float) -> str:
    """Convert numerical confidence score to categorical level"""
    if score > 0.8:
        return "high"
    elif score > 0.5:
        return "medium"
    else:
        return "low"

def predict(user_input: str) -> Dict[str, Any]:
    """Predict intent and extract entities from user input"""
    start_time = time.time()

    if not user_input or len(user_input.strip()) < 3:
        return {"intent": "unknown", "entities": {}, "confidence": "low", "processing_time": 0}

    # Preprocess input
    cleaned_input = preprocess_text(user_input)

    # Get intent classification with enhanced approach
    intent, confidence_score = enhanced_intent_classification(user_input)

    # Extract entities
    ner_results = ner(user_input)
    extracted_entities = extract_entities(user_input, ner_results)

    # Post-process: Make contextual corrections based on intent and entities

    # Special handling for get_member_info with no extracted name but name in input
    if intent == "get_member_info" and "name" not in extracted_entities:
        # Try harder to find a name in a simple query
        words = user_input.split()
        for word in words:
            if re.match(r"^[A-Z][a-z]+$", word) and word.lower() not in ["team", "info", "show", "get", "display"]:
                extracted_entities["name"] = word
                break

    # Special handling for name-only queries
    if re.match(r"^[A-Z][a-z]+$", user_input.strip()):
        intent = "get_member_info"
        extracted_entities["name"] = user_input.strip()
        confidence_score = 0.85

    # If someone asks for information about a person
    if re.search(r"(?i)(?:information|info|details)\s+(?:for|about|of|on)\s+([A-Z][a-z]+)", user_input):
        intent = "get_member_info"
        # Ensure the name is extracted
        if "name" not in extracted_entities:
            match = re.search(r"(?i)(?:information|info|details)\s+(?:for|about|of|on)\s+([A-Z][a-z]+)", user_input)
            if match:
                extracted_entities["name"] = match.group(1)
        confidence_score = 0.88

    # If someone says "show teams Alpha", they probably want info about team Alpha
    if intent == "list_teams" and "team_name" in extracted_entities:
        intent = "show_team_info"

    # Handle ambiguous cases with "team" followed by a name
    if re.search(r"(?i)team\s+[A-Z][a-z]+", user_input) and intent not in ["create_team", "delete_team", "show_team_info", "update_team_repo", "update_team_status"]:
        # Default to showing team info if just the team name is mentioned
        if len(cleaned_input.split()) < 4:
            intent = "show_team_info"
            # Extract team name if not already present
            if "team_name" not in extracted_entities:
                match = re.search(r"(?i)team\s+([A-Z][a-z]+)", user_input)
                if match:
                    extracted_entities["team_name"] = match.group(1)

    # If the intent is "assign_role" and we have a "promote" pattern, adjust entities
    if intent == "assign_role" and re.search(r"(?i)promote\s+([A-Za-z]+)\s+to\s+([a-zA-Z\s]+)", user_input):
        promote_match = re.search(r"(?i)promote\s+([A-Za-z]+)\s+to\s+([a-zA-Z\s]+)", user_input)
        if promote_match:
            extracted_entities["name"] = promote_match.group(1).strip()
            extracted_entities["role"] = promote_match.group(2).strip()

    # Final check to ensure common intents have high confidence
    if intent in ["list_teams", "show_team_info", "get_member_info", "assign_role"] and confidence_score < 0.7:
        confidence_score = max(confidence_score, 0.7)

    # Convert numerical confidence to categorical
    confidence_category = estimate_intent_confidence(confidence_score)

    result = {
        "intent": intent,
        "entities": extracted_entities,
        "confidence": confidence_category,
        "confidence_score": confidence_score,
        "processing_time": time.time() - start_time
    }

    logger.info(f"Predicted intent: {intent}, Confidence: {confidence_category} ({confidence_score:.4f}), Entities: {extracted_entities} for input: '{user_input}'")

    return result

def test_prediction(test_input: str) -> Dict[str, Any]:
    """Test the prediction system with a sample input"""
    print(f"Input: '{test_input}'")
    result = predict(test_input)
    print(f"Predicted intent: {result['intent']}")
    print(f"Entities: {result['entities']}")
    print(f"Confidence: {result['confidence']} ({result.get('confidence_score', 0):.4f})")
    print(f"Processing time: {result['processing_time']:.4f} seconds")
    print("-" * 50)
    return result

if __name__ == "__main__":
    # Run test cases
    test_cases = [
        "Create a new team Alpha",
        "add a new team Beta",
        "Assign John as developer in team Alpha",
        "Make Sarah lead developer in Bravo",
        "Who are the members of team Alpha?",
        "Show details for team Charlie",
        "Update repo of Alpha to https://github.com/company/alpha-project",
        "Update status of Bravo to inactive",
        "Remove Alex from team Delta",
        "List all teams",
        "Show all teams",
        "What role does John have?",
        "Hello there",
        "Help me use this system",
        "Set role of Mary as QA engineer in team Echo",
        "Update role of Zeta to Senior Developer",
        "show team Cosmic Creators",
        "delete team Avengers",
        "remove team Innovators",
        "add members Alice, Bob to team Gamma", # Test adding members
        "team Beta members are Carol, David", # Another way to specify members
        "change status of Lambda to on hold", # Test changing status
        "set the status of Mu to completed", # Another way to set status
        "update the repository of Sigma to https://gitlab.com/org/sigma", # Another repo update
        "assign the role of project manager to Eve in team Omega", # More natural assign role
        "what is the repository of team Tau?", # Should ideally go to show_team_info
        "who is the lead in team Kappa?", # Should ideally go to show_team_info
        "update members of team Zeta to Frank, Grace, Heidi", # Test direct member update
        "set team Delta status to planning", # Another status update
        "change team Alpha's repository to http://example.com/alpha", # Another repo update
        "assign a developer to Iris in team Theta", # Should extract role 'developer'
        "create team Phoenix",
        # Additional test cases for problem areas
        "show teams",
        "give me all teams",
        "list the teams we have",
        "display teams",
        "show me the teams",
        "create team Guardians",
        "make a new team called Defenders",
        "team Avengers",
        "promote Alice to lead in team Beta",
        "promote Bob to senior developer",
        "promote Charlie to project manager in team Gamma"
    ]

    print("Running test cases:")
    for test in test_cases:
        test_prediction(test)