from transformers import pipeline
import re
import logging
import time

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
    "update_team",          # For updating team details
    "show_team_info",       # For showing roles in a team
    "remove_member",        # For removing team members
    "list_teams",           # For listing all teams
    "get_member_info",      # For getting info about a specific member
    "help",                 # For help requests
    "greeting",             # For greetings/small talk
    "create_team",          # For creating a new team
    "delete_team"           # For deleting a team
]

ROLE_KEYWORDS = [
    "developer", "lead", "manager", "designer", "architect",
    "tester", "qa", "frontend", "backend", "fullstack",
    "devops", "product owner", "scrum master", "head",
    "director", "engineer", "analyst", "admin", "coordinator",
    "ui", "ux", "project manager", "technical writer"
]

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
        def dummy_classifier(text, candidate_labels):
            return {"labels": ["assign_role", "create_team", "update_team", "show_team_info", "delete_team"], "scores": [0.9, 0.1, 0.05, 0.7, 0.2]} # Include delete_team
        def dummy_ner(text):
            return []
        classifier = dummy_classifier
        ner = dummy_ner
        logger.critical("Using dummy ML functions")

def extract_entities(ner_results, text):
    # FIX: Standardize entity names to match what's used in bot2.py
    entities = {"name": None, "role": None, "team_name": None, "repo": None}

    # Extract person names from NER
    person_entities = [ent for ent in ner_results if ent["entity_group"] == "PER"]
    if person_entities:
        entities["name"] = " ".join([ent["word"] for ent in person_entities]).strip()

    # Extract team names using regex patterns
    team_patterns = [
        r"(?i)team\s+([A-Za-z][a-zA-Z0-9_.-]*)",          # Team X, Team Alpha, Team My-Project
        r"(?i)in\s+([A-Za-z][a-zA-Z0-9_.-]*)\s+team",      # in Alpha team
        r"(?i)for\s+team\s+([A-Za-z][a-zA-Z0-9_.-]*)",      # for team Alpha
        r"(?i)^(?:show\s+)?team\s+([A-Za-z][a-zA-Z0-9_.-]+)$", # Matches "show team Cosmic Creators"
        r"(?i)(create|make|new)\s+(?:new\s+)?team\s+([A-Za-z][a-zA-Z0-9_.-]+)", # Captures team name after create
        r"(?i)(delete|remove)\s+(?:team\s+)?([A-Za-z][a-zA-Z0-9_.-]+)" # Captures team name after delete/remove
    ]

    for pattern in team_patterns:
        team_match = re.search(pattern, text)
        if team_match:
            # Prioritize the team name found directly after "team" or in the "create team" pattern
            if len(team_match.groups()) > 1:
                entities["team_name"] = team_match.group(team_match.lastindex).strip()
            else:
                entities["team_name"] = team_match.group(1).strip()
            break

    # Extract roles using keywords and patterns
    for role in ROLE_KEYWORDS:
        if re.search(f"\\b{role}\\b", text.lower()):
            # Find the complete role phrase
            role_pattern = re.search(f"(?i)(senior|junior|lead|principal|chief)? ?{role}", text)
            if role_pattern:
                entities["role"] = role_pattern.group(0).strip()
                break

    # Extract repository URLs
    repo_patterns = [
        r"https?://github\.com/\S+",          # GitHub URLs
        r"https?://gitlab\.com/\S+",          # GitLab URLs
        r"https?://bitbucket\.org/\S+"       # Bitbucket URLs
    ]

    for pattern in repo_patterns:
        repo_match = re.search(pattern, text)
        if repo_match:
            entities["repo"] = repo_match.group(0)
            break

    # Try to find missing names if not found by NER
    if not entities["name"]:
        # Look for common patterns like "add [name] as [role]"
        name_patterns = [
            r"(?i)add\s+(\w+)\s+as",
            r"(?i)assign\s+(\w+)\s+as",
            r"(?i)for\s+(\w+)\s+as",
            r"(?i)remove\s+(\w+)\s+from" # Added for remove member
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, text)
            if name_match:
                entities["name"] = name_match.group(1)
                break

    # Clean up extracted entities
    for key, value in entities.items():
        if value:
            entities[key] = value.strip()

    return entities

def estimate_intent_confidence(scores):
    """Evaluate confidence in the intent classification"""
    if not scores:
        return "low"

    top_score = scores[0]
    if top_score > 0.8:
        return "high"
    elif top_score > 0.5:
        return "medium"
    else:
        return "low"

def predict(user_input):
    """
    Predicts intent and extracts entities from user input

    Args:
        user_input: String containing user's message

    Returns:
        dict: Contains 'intent', 'entities', 'confidence', and 'processing_time'
    """
    start_time = time.time()

    try:
        # Clean and prepare input
        cleaned_input = user_input.strip()

        # Handle empty or too short inputs
        if not cleaned_input or len(cleaned_input) < 3:
            return {
                "intent": "unknown",
                "entities": {},
                "confidence": "low",
                "processing_time": 0
            }

        # Check for help or greeting intents with simple rules first
        if re.search(r"(?i)\b(help|how to use|what can you do)\b", cleaned_input):
            return {
                "intent": "help",
                "entities": {},
                "confidence": "high",
                "processing_time": time.time() - start_time
            }

        if re.search(r"(?i)\b(hello|hi|hey|greetings|good morning|good afternoon)\b", cleaned_input) and len(cleaned_input.split()) < 4:
            return {
                "intent": "greeting",
                "entities": {},
                "confidence": "high",
                "processing_time": time.time() - start_time
            }

        # FIX: Prioritize "show team [team_name]" for "show_team_info"
        show_team_match = re.search(r"(?i)^(?:show\s+)?team\s+([A-Za-z][a-zA-Z0-9_.-]+)$", cleaned_input)
        if show_team_match:
            return {
                "intent": "show_team_info",
                "entities": {"team_name": show_team_match.group(1).strip()},
                "confidence": "high",
                "processing_time": time.time() - start_time
            }

        # FIX: Better detection of "create_team" intent
        create_team_match = re.search(r"(?i)(create|make|new)\s+(?:new\s+)?team(?:,\s+)?([A-Za-z0-9_.-]+)", cleaned_input)
        if create_team_match:
            return {
                "intent": "create_team",
                "entities": {"team_name": create_team_match.group(2).strip()},
                "confidence": "high",
                "processing_time": time.time() - start_time
            }

        # FIX: Detect "delete team" intent
        delete_team_match = re.search(r"(?i)(delete|remove)\s+team\s+([A-Za-z0-9_.-]+)", cleaned_input)
        if delete_team_match:
            return {
                "intent": "delete_team",
                "entities": {"team_name": delete_team_match.group(2).strip()},
                "confidence": "high",
                "processing_time": time.time() - start_time
            }
        delete_single_team_match = re.search(r"(?i)delete\s+([A-Za-z0-9_.-]+)\s+team", cleaned_input)
        if delete_single_team_match:
            return {
                "intent": "delete_team",
                "entities": {"team_name": delete_single_team_match.group(1).strip()},
                "confidence": "high",
                "processing_time": time.time() - start_time
            }

        # Do intent classification
        intent_result = classifier(cleaned_input, candidate_labels=INTENTS_LIST)
        predicted_intent = intent_result["labels"][0]
        confidence_scores = intent_result["scores"]

        # Extract entities
        ner_results = ner(cleaned_input)

        # Add a safety check here to ensure ner_results is a list
        if not isinstance(ner_results, list):
            logger.error(f"NER pipeline returned unexpected type: {type(ner_results)}, input: '{cleaned_input}'")
            extracted_entities = {} # Assign an empty dictionary in case of error
        else:
            extracted_entities = extract_entities(ner_results, cleaned_input)

        # FIX: Log the extracted entities to help with debugging
        logger.info(f"Extracted entities: {extracted_entities}")

        # Domain-specific post-processing based on intent
        if predicted_intent == "list_teams" and extracted_entities.get("team_name"):
            predicted_intent = "show_team_info"  # If team is mentioned in list_teams, it's likely show_team_info

        if predicted_intent == "assign_role" and not extracted_entities.get("name") and not extracted_entities.get("role") and extracted_entities.get("team_name"):
            # Might be asking about team info if only team name is present
            if re.search(r"(?i)\b(what are|who is in|members of)\b", cleaned_input):
                predicted_intent = "show_team_info"

        if predicted_intent == "assign_role" and not extracted_entities.get("team_name") and extracted_entities.get("name") and extracted_entities.get("role"):
            # Inferring context - assigning role to a member, need team
            pass # Keep as assign_role, handler will ask for team if needed

        # Improve detection of member removal
        if re.search(r"(?i)\b(remove|delete|kick|eliminate)\b", cleaned_input) and extracted_entities.get("name"):
            predicted_intent = "remove_member"

        # Try to infer "assign_role" if the pattern "make [name] [role] in [team]" is found
        make_role_match = re.search(r"(?i)make\s+(\w+)\s+([a-zA-Z\s]+)\s+in\s+([A-Za-z0-9_.-]+)", cleaned_input)
        if make_role_match:
            return {
                "intent": "assign_role",
                "entities": {
                    "name": make_role_match.group(1),
                    "role": make_role_match.group(2).strip(),
                    "team_name": make_role_match.group(3)
                },
                "confidence": "high",
                "processing_time": time.time() - start_time
            }

        # Try to infer "assign_role" if the pattern "assign [name] [role] to [team]" is found
        assign_to_match = re.search(r"(?i)assign\s+(\w+)\s+([a-zA-Z\s]+)\s+to\s+([A-Za-z0-9_.-]+)", cleaned_input)
        if assign_to_match:
            return {
                "intent": "assign_role",
                "entities": {
                    "name": assign_to_match.group(1),
                    "role": assign_to_match.group(2).strip(),
                    "team_name": assign_to_match.group(3)
                },
                "confidence": "high",
                "processing_time": time.time() - start_time
            }

        # Try to infer "set role of [name] as [role] in [team]"
        set_role_match = re.search(r"(?i)set\s+role\s+of\s+(\w+)\s+as\s+([a-zA-Z\s]+)\s+in\s+([A-Za-z0-9_.-]+)", cleaned_input)
        if set_role_match:
            return {
                "intent": "assign_role",
                "entities": {
                    "name": set_role_match.group(1),
                    "role": set_role_match.group(2).strip(),
                    "team_name": set_role_match.group(3)
                },
                "confidence": "high",
                "processing_time": time.time() - start_time
            }

        # Try to infer "update_team" for "update repo of [team] to [url]"
        update_repo_match = re.search(r"(?i)update\s+repo\s+of\s+([A-Za-z0-9_.-]+)\s+to\s+(\S+)", cleaned_input)
        if update_repo_match:
            return {
                "intent": "update_team",
                "entities": {
                    "team_name": update_repo_match.group(1),
                    "repo": update_repo_match.group(2)
                },
                "confidence": "high",
                "processing_time": time.time() - start_time
            }

        # Try to infer "update_team" for "update [field] of [team] to [value]"
        update_field_match = re.search(r"(?i)update\s+(role|repo|status)\s+of\s+([A-Za-z0-9_.-]+)\s+to\s+(\S+)", cleaned_input)
        if update_field_match:
            return {
                "intent": "update_team",
                "entities": {
                    "field": update_field_match.group(1),
                    "team_name": update_field_match.group(2),
                    "value": update_field_match.group(3)
                },
                "confidence": "medium",
                "processing_time": time.time() - start_time
            }

        # Check for member info requests
        member_info_match = re.search(r"(?i)info\s+(?:about|on)\s+(\w+)", cleaned_input)
        if member_info_match:
            return {
                "intent": "get_member_info",
                "entities": {
                    "name": member_info_match.group(1)
                },
                "confidence": "high",
                "processing_time": time.time() - start_time
            }
        
        # Process the results
        confidence = estimate_intent_confidence(confidence_scores)

        # Return the final result
        return {
            "intent": predicted_intent,
            "entities": extracted_entities,
            "confidence": confidence,
            "processing_time": time.time() - start_time
        }

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {
            "intent": "unknown",
            "entities": {},
            "confidence": "low",
            "processing_time": time.time() - start_time,
            "error": str(e)
        }

def test_prediction(test_input):
    """
    Test function for prediction
    """
    print(f"Input: '{test_input}'")
    result = predict(test_input)
    print(f"Predicted intent: {result['intent']}")
    print(f"Entities: {result['entities']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Processing time: {result['processing_time']:.4f} seconds")
    print("-" * 50)
    return result

if __name__ == "__main__":
    # Run some tests
    test_cases = [
        "Create a new team Alpha",
        "Assign John as developer in team Alpha",
        "Make Sarah lead developer in Bravo",
        "Who are the members of team Alpha?",
        "Update repo of Alpha to https://github.com/company/alpha-project",
        "Update status of Bravo to inactive",
        "Remove Alex from team Delta",
        "List all teams",
        "What role does John have?",
        "Hello there",
        "Help me use this system",
        "Set role of Mary as QA engineer in team Echo",
        "Update role of Zeta to Senior Developer",
        "show team Cosmic Creators",  # Test case for showing team info
        "delete team Avengers",      # Test case for deleting a team
        "remove team Innovators"     # Test case for deleting a team using remove
    ]

    print("Running test cases:")
    for test in test_cases:
        test_prediction(test)