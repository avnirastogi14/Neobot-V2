import discord
from discord.ext import commands
from pymongo import MongoClient
from fmodel import predict, INTENTS_LIST
import asyncio
import random
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
import re

load_dotenv(dotenv_path='C:/Users/Hrida/OneDrive/Documents/Desktop/Avni_College/foss_p/tesserx/data.env')

intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix="!", intents=intents)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bot2")

# Global variables for team creation process
TEAM_CREATION_USER = None
TEAM_CREATION_DATA = {}
TEAM_CREATION_FIELDS = ["team_name", "role", "members", "repo", "status"]
TEAM_CREATION_INDEX = 0

# Global variable to track if a command is being executed
IS_COMMAND_RUNNING = False

try:
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["discord_bot"]
    collection = db["Data"]
    logger.info("✅ Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"❌ MongoDB connection error: {e}")

@client.event
async def on_ready():
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="commands | !ping"))
    logger.info(f"✅ {client.user} is online | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Connected to {len(client.guilds)} server(s)")

@client.command()
async def ping(ctx):
    embed = discord.Embed(
        title="🏓 Pong!",
        description=f"Bot latency: {round(client.latency * 1000)}ms",
        color=discord.Color.green()
    )
    embed.set_footer(text=f"Requested by {ctx.author.display_name}")
    await ctx.send(embed=embed)

@client.command()
async def start(ctx):
    """Resumes the bot's message processing."""
    global bot_paused
    if 'bot_paused' in globals() and bot_paused:
        bot_paused = False
        await ctx.send("Bot has been resumed. I will now process messages.")
    else:
        await ctx.send("Bot is already running.")

@client.command()
async def end(ctx):
    """Pauses the bot, ignoring new messages."""
    global bot_paused
    bot_paused = True
    await ctx.send("Bot has been paused. I will not process new messages until '!start' is used.")

@client.command()
async def exit(ctx):
    """Exits the current command execution."""
    global IS_COMMAND_RUNNING, TEAM_CREATION_USER, TEAM_CREATION_DATA, TEAM_CREATION_INDEX
    if IS_COMMAND_RUNNING:
        IS_COMMAND_RUNNING = False
        await ctx.send("⌚❌ Exiting current operation - Execution Aborted!")
        TEAM_CREATION_USER = None
        TEAM_CREATION_DATA = {}
        TEAM_CREATION_INDEX = 0
    else:
        await ctx.send("⚠️ No command is currently running to exit.")

@client.command()
async def reset(ctx):
    """Resets any ongoing team creation process."""
    global TEAM_CREATION_USER, TEAM_CREATION_DATA, TEAM_CREATION_INDEX
    if ctx.author.guild_permissions.administrator or ctx.author.guild_permissions.manage_guild:
        old_user = TEAM_CREATION_USER
        TEAM_CREATION_USER = None
        TEAM_CREATION_DATA = {}
        TEAM_CREATION_INDEX = 0
        if old_user:
            await ctx.send("✅ Successfully reset the ongoing team creation process.")
        else:
            await ctx.send("ℹ️ There was no active team creation process to reset.")
    else:
        await ctx.send("⚠️ You need administrator permissions to reset team creation processes.")

@client.command()
async def bothelp(ctx):
    embed = discord.Embed(
        title="NeoBot Help",
        description="I understand natural language commands for team and role management.",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="Example Commands",
        value=( """ **Available Commands**:
    • `Create a new team`.
    • `Assign role <role> to <member> in <team>`.
    • `Update <team>'s repository to <URL>`.
    • `Update <team>'s members to <member1, member2, ...>`.
    • `Show details for team <team>`.
    • `Remove <member> from team <team>`.
    • `List all teams`.
    • `Delete team <team>`.
    • `!exit`: To exit from current command.
    • `!bothelp`: Show this help message. """),
        inline=False
    )
    embed.add_field(name="Technical Commands", value="• !ping - Check if bot is responsive", inline=False)
    embed.set_footer(text="I use ML to understand your requests")
    await ctx.send(embed=embed)

@client.event
async def on_message(message):
    global IS_COMMAND_RUNNING, TEAM_CREATION_USER, TEAM_CREATION_DATA, TEAM_CREATION_INDEX

    if message.author == client.user:
        return

    # Check if the bot is mentioned
    if client.user.mentioned_in(message):
        text = re.sub(r'<@!?\d+>', '', message.content).strip()  # Clean the message

        if text.lower() == "!exit" and IS_COMMAND_RUNNING:
            IS_COMMAND_RUNNING = False
            await message.channel.send("⌚❌ Exiting current operation - Execution Aborted!")
            TEAM_CREATION_USER = None
            TEAM_CREATION_DATA = {}
            TEAM_CREATION_INDEX = 0
            return

        await client.process_commands(message)  # Still process commands (e.g., !ping)

        if text.startswith(client.command_prefix):
            return

        # Check for ongoing team creation process
        if TEAM_CREATION_USER == message.author and TEAM_CREATION_INDEX < len(TEAM_CREATION_FIELDS):
            field = TEAM_CREATION_FIELDS[TEAM_CREATION_INDEX]
            TEAM_CREATION_DATA[field] = text  # Use the cleaned text
            TEAM_CREATION_INDEX += 1

            if TEAM_CREATION_INDEX < len(TEAM_CREATION_FIELDS):
                await message.channel.send(f"Alright, next up: the **{TEAM_CREATION_FIELDS[TEAM_CREATION_INDEX].replace('_', ' ')}**? (or type 'skip' to leave empty)")
            else:
                await handle_create_team_interactive(message, TEAM_CREATION_DATA)
                TEAM_CREATION_USER = None
                TEAM_CREATION_DATA = {}
                TEAM_CREATION_INDEX = 0
            return

        # Cache to avoid repeat processing (keep this)
        cache_key = f"{message.channel.id}:{message.id}"
        if not hasattr(client, 'processed_messages'):
            client.processed_messages = set()
        if cache_key in client.processed_messages:
            return
        client.processed_messages.add(cache_key)
        if len(client.processed_messages) > 100:
            client.processed_messages = set(list(client.processed_messages)[-80:])

        # ML Prediction
        try:
            prediction_result = predict(text)  # Use the cleaned text
            intent = prediction_result.get("intent")
            entities = prediction_result.get("entities", {})
            confidence = prediction_result.get("confidence", "low")
            logger.info(f"Intent predicted: {intent}, Entities: {entities}, Confidence: {confidence}")

            if intent == "help" and confidence == "high":
                await client.get_command('bothelp').invoke(await client.get_context(message))
                return
            elif intent == "exit" and confidence == "high":
                await message.channel.send("⌚❌ Exiting Command - Command Aborted!")
                return
            elif not intent or intent == "unknown" or confidence == "low":
                # Provide a more helpful "unknown command" response
                responses = [
                    "Hmm, I'm not quite sure what you're asking. Could you rephrase?",
                    "Sorry, I didn't understand that command. Try `!bothelp` for available commands.",
                    "That's an interesting request! However, I don't have a function for that yet. Check `!bothelp`.",
                    "My apologies, but I couldn't process your request. Please see `!bothelp` for guidance.",
                    "Could you please clarify your command? I might have misunderstood. `!bothelp` lists what I can do."
                ]
                await message.channel.send(random.choice(responses))
                return
        except Exception as e:
            await message.channel.send(f"❌ Prediction error: `{str(e)}`")
            return

        logger.info(f"Handling intent: {intent} with entities: {entities}")
        IS_COMMAND_RUNNING = True

        try:
            # ... (your intent handling logic remains the same)
            if intent == "assign_role":
                await handle_assign_role(message, entities)
            elif intent == "update_team_repo":
                await handle_update_team_repo(message, entities)
            elif intent == "update_team_members":
                await handle_update_team_members(message, entities)
            elif intent == "update_team_status":
                await handle_update_team_status(message, entities)
            elif intent == "update_team_role":
                await handle_update_team_role(message, entities)
            elif intent == "show_team_info":
                await handle_show_team_info(message, entities)
            elif intent == "remove_member":
                await handle_remove_member(message, entities)
            elif intent == "list_teams":
                await handle_list_teams(message)
            elif intent == "create_team":
                logger.info("Calling start_create_team function.")
                await start_create_team(message)
            elif intent == "delete_team":
                await handle_delete_team(message, entities)
            elif intent == "greeting" and confidence == "high":
                greetings = [f"👋 Hello {message.author.display_name}!", f"Hey there, {message.author.display_name}!", f"Greetings, {message.author.display_name}!"]
                await message.channel.send(random.choice(greetings))
        finally:
            IS_COMMAND_RUNNING = False
    else:
        await client.process_commands(message) # Allow regular commands (!ping, !help) to work even without a mention

async def handle_assign_role(message, entities):
    """Handle role assignment intent."""
    name = entities.get("member_name") or entities.get("name")
    role = entities.get("role")
    team = entities.get("team_name") or entities.get("team")

    if not name:
        await message.channel.send("⚠️ Who are you trying to assign a role to?")
        return
    if not team:
        await message.channel.send("⚠️ Which team are you referring to?")
        return

    data = {
        "name": name,
        "role": role,
        "team": team,
        "updated_at": datetime.now()
    }

    try:
        existing_member = collection.find_one({"name": name, "team": team})

        if existing_member:
            collection.update_one(
                {"name": name, "team": team},
                {"$set": data}
            )
            response = f"Updated **{name}'s** role to **{role}** in **{team}**." if role else f"Removed the role for **{name}** in **{team}**."
        else:
            team_doc = collection.find_one({"team_name": team})
            if team_doc:
                collection.update_one(
                    {"team_name": team},
                    {"$addToSet": {"members": name}}
                )
            collection.insert_one(data)
            response = f"Assigned **{role}** to **{name}** in **{team}**." if role else f"Added **{name}** to **{team}**."

        fields = [
            ("Member", name, True),
            ("Role", role if role else "N/A", True),
            ("Team", team, True)
        ]
        embed = await create_success_embed(
            "Role Assignment",
            response,
            fields
        )
        await message.channel.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in handle_assign_role: {e}")
        await message.channel.send(f"❌ Database error: {e}")

from datetime import datetime
import re

async def handle_update_team_repo(message, entities):
    """Handle updating team repo URL."""
    team_name = (entities.get("team_name") or entities.get("team") or "").strip()
    repo = (entities.get("repo") or "").strip()

    if not team_name:
        await message.channel.send("⚠️ Please specify the team name to update the repository for.")
        return
    if not repo:
        await message.channel.send("⚠️ Please provide the new repository URL.")
        return

    try:
        # Case-insensitive search
        result = collection.update_one(
            {
                "$or": [
                    {"team_name": re.compile(f"^{re.escape(team_name)}$", re.IGNORECASE)},
                    {"team": re.compile(f"^{re.escape(team_name)}$", re.IGNORECASE)}
                ]
            },
            {"$set": {"repo": repo, "updated_at": datetime.utcnow()}}
        )

        if result.modified_count > 0:
            fields = [
                ("Team", team_name, True),
                ("Repository", repo, False)
            ]
            embed = await create_success_embed(
                "✅ Repository Updated",
                f"The repository URL for **{team_name}** has been updated.",
                fields
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(f"⚠️ No matching team found with the name **{team_name}**.")
    except Exception as e:
        logger.error(f"Error in handle_update_team_repo: {e}")
        await message.channel.send(f"❌ Database error: {e}")

async def handle_update_team_members(message, entities):
    """Handle updating team members directly."""
    team_name = entities.get("team_name") or entities.get("team")
    members_str = entities.get("members")

    if not team_name:
        await message.channel.send("⚠️ Please specify the team to update members for.")
        return
    if not members_str:
        await message.channel.send("⚠️ Please provide the new list of members.")
        return

    members_list = [m.strip() for m in members_str.split(",")]

    try:
        result = collection.update_one(
            {"$or": [{"team_name": team_name}, {"team": team_name}]},
            {"$set": {"members": members_list, "updated_at": datetime.utcnow()}}
        )
        if result.modified_count > 0:
            fields = [
                ("Team", team_name, True),
                ("Members", "\n• " + "\n• ".join(members_list), False)
            ]
            embed = await create_success_embed(
                "Team Members Updated",
                f"The members for **{team_name}** have been updated.",
                fields
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(f"⚠️ Could not find team **{team_name}**.")
    except Exception as e:
        logger.error(f"Error in handle_update_team_members: {e}")
        await message.channel.send(f"❌ Database error: {e}")

from datetime import datetime
import re

async def handle_update_team_status(message, entities):
    """Handle updating team status directly."""
    team_name = (entities.get("team_name") or entities.get("team") or "").strip()
    status = (entities.get("status") or "").strip()

    if not team_name:
        await message.channel.send("⚠️ Which team's status do you want to update?")
        return
    if not status:
        await message.channel.send("⚠️ What is the new status?")
        return

    try:
        # Case-insensitive search using regex
        result = collection.update_one(
            {
                "$or": [
                    {"team_name": re.compile(f"^{re.escape(team_name)}$", re.IGNORECASE)},
                    {"team": re.compile(f"^{re.escape(team_name)}$", re.IGNORECASE)}
                ]
            },
            {"$set": {"status": status, "updated_at": datetime.utcnow()}}
        )

        if result.modified_count > 0:
            fields = [("Team", team_name, True), ("Status", status, True)]
            embed = await create_success_embed(
                "✅ Team Status Updated",
                f"The status for **{team_name}** has been updated to **{status}**.",
                fields
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(f"⚠️ No matching team found with the name **{team_name}**.")
    except Exception as e:
        logger.error(f"Error in handle_update_team_status: {e}")
        await message.channel.send(f"❌ Database error: {e}")

async def handle_update_team_role(message, entities):
    """Handle updating the overall team role (if your data model supports it)."""
    team_name = entities.get("team_name") or entities.get("team")
    role = entities.get("role")  # Assuming your NLP can differentiate this from member role

    if not team_name:
        await message.channel.send("⚠️ Which team's role do you want to update?")
        return
    if not role:
        await message.channel.send("⚠️ What is the new role for the team?")
        return

    try:
        result = collection.update_one(
            {"$or": [{"team_name": team_name}, {"team": team_name}]},
            {"$set": {"role": role, "updated_at": datetime.utcnow()}}
        )
        if result.modified_count > 0:
            fields = [
                ("Team", team_name, True),
                ("Role", role, True)
            ]
            embed = await create_success_embed(
                "Team Role Updated",
                f"The role for **{team_name}** has been updated to **{role}**.",
                fields
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(f"⚠️ Could not find team **{team_name}**.")
    except Exception as e:
        logger.error(f"Error in handle_update_team_role: {e}")
        await message.channel.send(f"❌ Database error: {e}")

async def handle_show_team_info(message, entities):
    """Handle showing details for a specific team."""
    def extract_team_name(text):
        pattern = r"(?:show\s+(?:team\s+)?)(?:\"([^\"]+)\"|([A-Za-z\s]+))"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) or match.group(2) if match else None

    if not entities.get("team_name") and not entities.get("team"):
        fallback = extract_team_name(message.content)
        if fallback:
            entities["team_name"] = fallback.strip()

    team_name = entities.get("team_name") or entities.get("team")

    if not team_name:
        await message.channel.send("⚠️ Please specify the team name you want to see details for.")
        return

    try:
        doc = collection.find_one({
            "$or": [
                {"team_name": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}},
                {"team": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}}
            ]
        })

        if doc:
            members_list = doc.get("members", [])
            members_str = "\n• " + "\n• ".join(members_list) if members_list else "No members"

            embed = discord.Embed(
                title=f"Team: {doc.get('team_name', doc.get('team', team_name))}",
                color=discord.Color.blue()
            )
            if "role" in doc:
                embed.add_field(name="Role", value=doc["role"], inline=True)
            if "status" in doc:
                embed.add_field(name="Status", value=doc["status"], inline=True)
            if "repo" in doc:
                embed.add_field(name="Repository", value=doc["repo"], inline=False)
            embed.add_field(name="Members", value=members_str, inline=False)
            embed.set_footer(text="Team details fetched from the database")
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(f"⚠️ Team **{team_name}** not found in the database.")
    except Exception as e:
        logger.error(f"Error in handle_show_team_info: {e}")
        await message.channel.send(f"❌ Database error: {e}")

async def handle_remove_member(message, entities):
    """Handle removing a member from a team."""
    def extract_entities(text):
        member_pattern = r"(?:remove|delete)\s+([A-Za-z]+)"
        team_pattern = r"from\s+(?:team\s+)?(?:\"([^\"]+)\"|([A-Za-z\s]+))"

        member_match = re.search(member_pattern, text, re.IGNORECASE)
        team_match = re.search(team_pattern, text, re.IGNORECASE)

        member_name = member_match.group(1) if member_match else None
        team_name = team_match.group(1) or team_match.group(2) if team_match else None

        return {
            "member_name": member_name.strip() if member_name else None,
            "team_name": team_name.strip() if team_name else None
        }

    if not entities.get("team_name") and not entities.get("team"):
        fallback = extract_entities(message.content)
        entities.update({k: v for k, v in fallback.items() if v})

    team_name = entities.get("team_name") or entities.get("team")
    name = entities.get("member_name") or entities.get("name")

    if not name:
        await message.channel.send("⚠️ Please specify the member you want to remove.")
        return
    if not team_name:
        await message.channel.send("⚠️ Please specify the team to remove the member from.")
        return

    try:
        team_doc = collection.find_one({
            "$or": [
                {"team_name": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}},
                {"team": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}}
            ]
        })

        if not team_doc:
            await message.channel.send(f"⚠️ Team **{team_name}** not found.")
            return

        if name not in team_doc.get("members", []):
            await message.channel.send(f"⚠️ **{name}** is not a member of **{team_doc.get('team_name', team_name)}**.")
            return

        result = collection.update_one(
            {"_id": team_doc["_id"]},
            {"$pull": {"members": name}, "$set": {"updated_at": datetime.utcnow()}}
        )

        if result.modified_count > 0:
            fields = [
                ("Member", name, True),
                ("Team", team_doc.get("team_name", team_name), True)
            ]
            embed = await create_success_embed(
                "Member Removed",
                f"**{name}** has been removed from **{team_doc.get('team_name', team_name)}**.",
                fields
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send("⚠️ Could not remove the member. Please try again.")
    except Exception as e:
        logger.error(f"Error in handle_remove_member: {e}")
        await message.channel.send(f"❌ Database error: {e}")

async def handle_list_teams(message):
    """Handle listing all teams in the database."""
    try:
        teams = list(collection.distinct("team_name")) + list(collection.distinct("team"))
        teams = [t for t in teams if t]  # Filter out None values
        unique_teams = sorted(list(set(teams)))

        if unique_teams:
            embed = discord.Embed(
                title="All Teams",
                description="\n• " + "\n• ".join(unique_teams),
                color=discord.Color.blue()
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send("There are currently no teams in the database.")
    except Exception as e:
        logger.error(f"Error in handle_list_teams: {e}")
        await message.channel.send(f"❌ Database error: {e}")

async def handle_delete_team(message, entities):
    """Handle deleting a team from the database."""
    team_name = entities.get("team_name") or entities.get("team")

    if not team_name:
        await message.channel.send("⚠️ Please specify the name of the team you wish to delete.")
        return

    try:
        delete_result = collection.delete_one({
            "$or": [
                {"team_name": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}},
                {"team": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}}
            ]
        })

        if delete_result.deleted_count > 0:
            embed = await create_success_embed(
                "Team Deleted",
                f"Team **{team_name}** has been successfully removed."
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(f"⚠️ No team found with the name **{team_name}** to delete.")
    except Exception as e:
        logger.error(f"Error deleting team {team_name}: {e}")
        await message.channel.send(f"❌ Database error while deleting the team: {e}")

async def create_success_embed(title: str, description: str, fields: list = []) -> discord.Embed:
    """Creates a standard success embed."""
    embed = discord.Embed(title=title, description=description, color=discord.Color.green())
    for name, value, inline in fields:
        embed.add_field(name=name, value=value, inline=inline)
    return embed

async def start_create_team(message: discord.Message):
    """Starts the interactive team creation process."""
    global TEAM_CREATION_USER, TEAM_CREATION_DATA, TEAM_CREATION_INDEX
    if TEAM_CREATION_USER is not None:
        await message.channel.send("⏳ A team creation process is already underway. Please finish that first or type `!exit` to cancel.")
        return
    TEAM_CREATION_USER = message.author
    TEAM_CREATION_DATA = {}
    TEAM_CREATION_INDEX = 0
    await message.channel.send(f"Alright, let's get a new team set up! First, what will be the **{TEAM_CREATION_FIELDS[0].replace('_', ' ')}**?")

async def handle_create_team_interactive(message: discord.Message, team_data: dict):
    """Handles the interactive creation of a new team."""
    global TEAM_CREATION_USER, TEAM_CREATION_DATA, TEAM_CREATION_INDEX
    team_name = team_data.get("team_name")
    role = team_data.get("role")
    members_str = team_data.get("members")
    repo = team_data.get("repo")
    status = team_data.get("status")

    if not team_name:
        await message.channel.send("A team needs a name! Let's try again from the beginning.")
        TEAM_CREATION_USER = message.author
        TEAM_CREATION_DATA = {}
        TEAM_CREATION_INDEX = 0
        await message.channel.send(f"Alright, let's get a new team set up! First, what will be the **{TEAM_CREATION_FIELDS[0].replace('_', ' ')}**?")
        return

    if collection.find_one({"team_name": team_name}):
        await message.channel.send(f"A team with the name **{team_name}** already exists. Please choose a different name.")
        TEAM_CREATION_USER = message.author
        TEAM_CREATION_DATA = {}
        TEAM_CREATION_INDEX = 0
        await message.channel.send(f"Alright, let's get a new team set up! First, what will be the **{TEAM_CREATION_FIELDS[0].replace('_', ' ')}**?")
        return

    members = [member.strip() for member in members_str.split(',')] if members_str and members_str.lower() != "skip" else []

    team_info = {
        "team_name": team_name,
        "role": role if role and role.lower() != "skip" else "",
        "members": members,
        "repo": repo if repo and repo.lower() != "skip" else "",
        "status": status if status and status.lower() != "skip" else "",
        "created_at": datetime.now()
    }

    try:
        collection.insert_one(team_info)

        fields = [
                ("Role", team_info["role"] if team_info["role"] else "N/A", True),
                ("Status", team_info["status"] if team_info["status"] else "N/A", True),
                ("Repository", team_info["repo"] if team_info["repo"] else "N/A", False),
                ("Members", "\n• " + "\n• ".join(members) if members else "No members yet", False)
            ]

        embed = await create_success_embed(
                "Team Created!",
                f"The team **{team_name}** has been successfully created!",
                fields
            )
        await message.channel.send(embed=embed) # <---- THIS IS WHERE THE MESSAGE IS SENT
    except Exception as e:
        logger.error(f"Error creating team {team_name}: {e}")
        await message.channel.send(f"❌ Oops! There was an issue creating the team: {e}")
    finally:
        TEAM_CREATION_USER = None
        TEAM_CREATION_DATA = {}
        TEAM_CREATION_INDEX = 0

async def handle_exit_command(message: discord.Message):
    """Handles cancellation of the team creation process."""
    global TEAM_CREATION_USER, TEAM_CREATION_DATA, TEAM_CREATION_INDEX
    if TEAM_CREATION_USER == message.author:
        TEAM_CREATION_USER = None
        TEAM_CREATION_DATA = {}
        TEAM_CREATION_INDEX = 0
        await message.channel.send("🚪 Team creation process has been cancelled.")
    else:
        await message.channel.send("❌ You are not the one currently creating a team.")

client.run(os.getenv('DISCORD_BOT_TOKEN'))