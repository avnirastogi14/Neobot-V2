import discord
from discord.ext import commands
from pymongo import MongoClient
from fmodel import predict, INTENTS_LIST
import asyncio
import random
import os
from datetime import datetime
from dotenv import load_dotenv
import logging  # Import the logging module
import re

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = commands.Bot(command_prefix="!", intents=intents)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("bot2")

# Global variables for team creation process
TEAM_CREATION_USER = ""
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


def parse_hex(s: str | None):
    if not s:
        return None
    s = s.lower().lstrip("#")
    if len(s) != 6:
        return None
    if s not in "abcdef0123456789":
        return None

    return discord.Color(int(f"0x{s}", 16))


async def resolve_member(token: str, guild: discord.Guild) -> discord.Member | None:
    mention = re.fullmatch(r"<@!?(?P<id>\d+)>", token.strip())
    if mention:
        uid = int(mention.group("id"))
        member = guild.get_member(uid) or await guild.fetch_member(uid)
        return member

    member = guild.get_member_named(token)
    if member:
        return member

    token_lower = token.lower()
    member = discord.utils.find(
        lambda m: m.display_name.lower() == token_lower
        or m.name.lower() == token_lower,
        guild.members,
    )
    return member


async def parse_members(raw: str, guild: discord.Guild):
    names_for_db, matched_members, not_found = [], [], []
    tokens = re.split(r"[,;\n]+", raw)

    for tok in filter(None, (t.strip() for t in tokens)):
        member = await resolve_member(tok, guild)
        names_for_db.append(member.display_name if member else tok)
        if member:
            matched_members.append(member)
        else:
            not_found.append(tok)

    names_for_db = list(dict.fromkeys(names_for_db))
    return names_for_db, matched_members, not_found


@client.event
async def on_ready():
    await client.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening, name="commands | !ping"
        )
    )
    logger.info(
        f"✅ {client.user} is online | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Connected to {len(client.guilds)} server(s)")


@client.command()
async def ping(ctx):
    embed = discord.Embed(
        title="🏓 Pong!",
        description=f"Bot latency: {round(client.latency * 1000)}ms",
        color=discord.Color.green(),
    )
    embed.set_footer(text=f"Requested by {ctx.author.display_name}")
    await ctx.send(embed=embed)


@client.command()
async def bothelp(ctx):
    embed = discord.Embed(
        title="TestBotA Help",
        description="I understand natural language commands for team and role management.",
        color=discord.Color.blue(),
    )
    embed.add_field(
        name="Example Commands",
        value=(
            """ **Available Commands**:
    • `Create a new team`.
    • `Assign role to a member in the team`.
    • `Update Team Details - Members, Repository, Status`.
    • `Remove a Team Member`.
    • `List all teams in the database`.
    • `Show a specific team's details`.
    • `Delete a team`.
    • `!exit`: To exit from current command.
    • `!bothelp`: Show this help message. """
        ),
        inline=False,
    )
    embed.add_field(
        name="Technical Commands",
        value="• !ping - Check if bot is responsive",
        inline=False,
    )
    embed.set_footer(text="I use ML to understand your requests")
    await ctx.send(embed=embed)


@client.event
async def on_message(message):
    global \
        IS_COMMAND_RUNNING, \
        TEAM_CREATION_USER, \
        TEAM_CREATION_DATA, \
        TEAM_CREATION_INDEX

    # ─── Ignore messages sent by ourselves ───────────────────────────────────
    if message.author == client.user:
        return

    # Allow the user to abort any running flow without tagging the bot
    if message.content.lower() == "!exit" and IS_COMMAND_RUNNING:
        IS_COMMAND_RUNNING = False
        await message.channel.send("⌚❌ Exiting current command - Execution Aborted!")
        TEAM_CREATION_USER = None
        TEAM_CREATION_DATA = {}
        TEAM_CREATION_INDEX = 0
        return

    # Prefix commands (e.g. !ping) should always work
    await client.process_commands(message)
    if message.content.startswith(client.command_prefix):
        return

    # ─── NEW MENTION GATE (already in the file) ───────────────────────────────
    bot_mentioned = (message.guild is None) or (client.user in message.mentions)
    in_team_creation = (
        TEAM_CREATION_USER == message.author
        and TEAM_CREATION_INDEX < len(TEAM_CREATION_FIELDS)
    )
    if not bot_mentioned and not in_team_creation:
        return
    # ──────────────────────────────────────────────────────────────────────────

    # ─── HANDLE ON-GOING “CREATE TEAM” WIZARD ────────────────────────────────
    if in_team_creation:
        field = TEAM_CREATION_FIELDS[TEAM_CREATION_INDEX]
        TEAM_CREATION_DATA[field] = message.content
        TEAM_CREATION_INDEX += 1

        if TEAM_CREATION_INDEX < len(TEAM_CREATION_FIELDS):
            await message.channel.send(
                f"Great! Now, what is the **{TEAM_CREATION_FIELDS[TEAM_CREATION_INDEX]}**? "
                "(or type 'skip' to leave empty)"
            )
        else:
            await handle_create_team_interactive(message, TEAM_CREATION_DATA)
            # reset state
            TEAM_CREATION_USER = None
            TEAM_CREATION_DATA = {}
            TEAM_CREATION_INDEX = 0
        return
    # ──────────────────────────────────────────────────────────────────────────

    # Cache to avoid repeat processing
    cache_key = f"{message.channel.id}:{message.id}"
    if not hasattr(client, "processed_messages"):
        client.processed_messages = set()
    if cache_key in client.processed_messages:
        return
    client.processed_messages.add(cache_key)
    if len(client.processed_messages) > 100:
        client.processed_messages = set(list(client.processed_messages)[-80:])

    if message.author == client.user:
        return

    # ML Prediction
    try:
        prediction_result = predict(message.content)
        intent = prediction_result.get("intent")
        entities = prediction_result.get("entities", {})
        confidence = prediction_result.get("confidence", "low")
        logger.info(
            f"Intent predicted: {intent}, Entities: {entities}, Confidence: {confidence}"
        )

        if intent == "help" and confidence == "high":
            await client.get_command("bothelp").invoke(
                await client.get_context(message)
            )
            return
        if intent == "exit" and confidence == "high":
            await message.channel.send("⌚❌ Exiting Command - Command Aborted!")
            return
    except Exception as e:
        await message.channel.send(f"❌ Prediction error: `{str(e)}`")
        return

    logger.info(f"Handling intent: {intent} with entities: {entities}")

    IS_COMMAND_RUNNING = True  # ✅ This is now legal

    try:
        if intent == "assign_role":
            await handle_assign_role(message, entities)
        elif intent == "update_team_repo":
            await handle_update_team_repo(message, entities)
        elif (
            intent == "update_team"
        ):  # This intent might still be triggered by old phrases
            await handle_update_team(message, entities)  # Redirect to the new handler
        elif (
            intent == "show_team_info"
        ):  # Re-using the existing intent for showing team info
            await handle_show_team_info(message, entities)  # New handler
        elif intent == "remove_member":
            await handle_remove_member(message, entities)
        elif intent == "list_teams":
            await handle_list_teams(message)
        elif intent == "get_member_info":
            await handle_member_info(message, entities)
        elif intent == "create_team":
            logger.info("Calling handle_create_team function.")
            await start_create_team(message)
        elif intent == "delete_team":
            await handle_delete_team(message, entities)
        elif intent == "greeting" and confidence == "high":
            await message.channel.send(f"👋 Hello {message.author.display_name}!")
        elif not intent or intent == "unknown" or confidence == "low":
            # Don't respond with the help message for low confidence predictions
            pass
    finally:
        IS_COMMAND_RUNNING = False  # Reset flag after command execution (or error)


async def create_success_embed(title, description, fields=None):
    """Create a consistent success embed that stands out"""
    embed = discord.Embed(
        title=f"✅ {title}", description=description, color=discord.Color.green()
    )
    if fields:
        for name, value, inline in fields:
            if value:
                embed.add_field(name=name, value=value, inline=inline)

    embed.timestamp = datetime.now()  # ✅ Fix here
    return embed


# async def handle_assign_role(message, entities):
#     """Handle role assignment intent"""
#     # FIX: Standardize entity names from model
#     name = entities.get("member_name") or entities.get("name")
#     role = entities.get("role")
#     team = entities.get("team_name") or entities.get("team")

#     if not name:
#         await message.channel.send("❌ Error: Name not provided")
#         return

#     if not team:
#         await message.channel.send("❌ Error: Team name not provided")
#         return

#     data = {
#         "name": name,
#         "role": role,
#         "team": team,
#         "updated_at": datetime.now(),
#     }

#     try:
#         # FIX: Find if user exists in the team and update, otherwise add
#         existing_member = collection.find_one({"name": name, "team": team})

#         if existing_member:
#             # Update existing member
#             collection.update_one({"name": name, "team": team}, {"$set": data})
#         else:
#             # Add to team's members list first
#             team_doc = collection.find_one({"team_name": team})
#             if team_doc:
#                 collection.update_one(
#                     {"team_name": team}, {"$addToSet": {"members": name}}
#                 )

#             # Then add member document
#             collection.insert_one(data)

#         # Use the new success embed format
#         fields = [
#             ("Member", name, True),
#             ("Role", role if role else "N/A", True),
#             ("Team", team, True),
#         ]
#         embed = await create_success_embed(
#             "Role Assignment Successful",
#             f"**{name}** has been added to **{team}**"
#             + (f" as **{role}**" if role else ""),
#             fields,
#         )
#         await message.channel.send(embed=embed)
#     except Exception as e:
#         logger.error(f"Error in handle_assign_role: {e}")
#         await message.channel.send(f"❌ Database error: {e}")


async def handle_update_team_repo(message, entities):
    """Handle updating team repo URL"""
    team_name = entities.get("team_name") or entities.get("team")
    repo = entities.get("repo")

    if not team_name or not repo:
        await message.channel.send("❌ Team name and repository URL are required.")
        return

    try:
        result = collection.update_one(
            {"team_name": team_name},
            {"$set": {"repo": repo, "updated_at": datetime.now()}},
        )

        if result.modified_count == 0:
            # Try with "team" field
            result = collection.update_one(
                {"team": team_name},
                {"$set": {"repo": repo, "updated_at": datetime.now()}},
            )

        if result.modified_count > 0:
            # Get updated team info
            team_doc = collection.find_one(
                {"team_name": team_name}
            ) or collection.find_one({"team": team_name})

            fields = [("Repository", repo, False), ("Team", team_name, True)]
            embed = await create_success_embed(
                "Repository Updated",
                f"Repository URL for **{team_name}** has been updated.",
                fields,
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(f"⚠️ Team **{team_name}** not found.")
    except Exception as e:
        logger.error(f"Error in handle_update_team_repo: {e}")
        await message.channel.send(f"❌ Database error: {e}")


async def handle_update_team(message, entities):
    """Handle updating team details"""
    team_name = entities.get("team_name") or entities.get("team")

    if not team_name:
        await message.channel.send("Please specify the team name to update.")
        return

    await message.channel.send(
        f"Which field of **{team_name}** would you like to update? (role, repo, status, members)"
    )

    def check(m):
        return (
            m.author == message.author
            and m.channel == message.channel
            and m.content.lower() in ["role", "repo", "status", "members"]
        )

    try:
        msg = await client.wait_for("message", check=check, timeout=30.0)
        field_to_update = msg.content.lower()

        if field_to_update == "members":
            await message.channel.send(
                f"Please provide the new list of **members**, separated by commas."
            )
            try:
                members_msg = await client.wait_for(
                    "message",
                    check=lambda m: m.author == message.author
                    and m.channel == message.channel,
                    timeout=30.0,
                )
                members_list = [m.strip() for m in members_msg.content.split(",")]
                update_result = collection.update_one(
                    {"$or": [{"team_name": team_name}, {"team": team_name}]},
                    {
                        "$set": {
                            "members": members_list,
                            "updated_at": datetime.utcnow(),
                        }
                    },
                )
                if update_result.modified_count > 0:
                    # Get updated team info for display
                    team_doc = collection.find_one(
                        {"$or": [{"team_name": team_name}, {"team": team_name}]}
                    )

                    fields = [
                        ("Team", team_name, True),
                        ("Members", "\n• " + "\n• ".join(members_list), False),
                    ]
                    embed = await create_success_embed(
                        "Team Members Updated",
                        f"Members for **{team_name}** have been updated.",
                        fields,
                    )
                    await message.channel.send(embed=embed)
                else:
                    await message.channel.send(
                        f"⚠️ Could not find team **{team_name}** to update members."
                    )
            except asyncio.TimeoutError:
                await message.channel.send("❌ Update members timed out.")
                return

        else:
            await message.channel.send(
                f"What is the new value for **{field_to_update}**?"
            )
            try:
                value_msg = await client.wait_for(
                    "message",
                    check=lambda m: m.author == message.author
                    and m.channel == message.channel,
                    timeout=30.0,
                )
                new_value = value_msg.content

                update_data = {"updated_at": datetime.utcnow()}
                update_data[field_to_update] = new_value

                result = collection.update_one(
                    {"$or": [{"team_name": team_name}, {"team": team_name}]},
                    {"$set": update_data},
                )

                if result.matched_count:
                    # Get updated team info for display
                    team_doc = collection.find_one(
                        {"$or": [{"team_name": team_name}, {"team": team_name}]}
                    )

                    fields = [
                        ("Team", team_name, True),
                        (field_to_update.capitalize(), new_value, True),
                    ]
                    embed = await create_success_embed(
                        "Team Updated",
                        f"Updated **{field_to_update}** for **{team_name}**.",
                        fields,
                    )
                    await message.channel.send(embed=embed)
                else:
                    await message.channel.send(
                        embed=discord.Embed(
                            title="Team Not Found",
                            description=f"No team found with name **{team_name}**.",
                            color=discord.Color.gold(),
                        )
                    )
            except asyncio.TimeoutError:
                await message.channel.send("❌ Update timed out.")
                return

    except asyncio.TimeoutError:
        await message.channel.send("❌ No field specified to update in time.")


async def handle_show_team_info(message, entities):
    # === Fallback extractor if NLP misses team name ===
    def extract_team_name(text):
        pattern = r"(?:show\s+(?:team\s+)?)(?:\"([^\"]+)\"|([A-Za-z\s]+))"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) or match.group(2) if match else None

    # Fallback entity extraction
    if not entities.get("team_name") and not entities.get("team"):
        fallback = extract_team_name(message.content)
        if fallback:
            entities["team_name"] = fallback.strip()

    team_name = entities.get("team_name") or entities.get("team")

    if not team_name:
        await message.channel.send("Please specify the team name.")
        return

    # Case-insensitive search
    doc = collection.find_one(
        {
            "$or": [
                {"team_name": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}},
                {"team": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}},
            ]
        }
    )

    if not doc:
        await message.channel.send(
            embed=discord.Embed(
                title="Team Not Found",
                description=f"No data for team **{team_name}**.",
                color=discord.Color.gold(),
            )
        )
        return

    # Prepare response
    members_list = doc.get("members", [])
    members = "\n• " + "\n• ".join(members_list) if members_list else "No members"

    role = doc.get("role", "N/A")
    repo = doc.get("repo", "N/A")
    status = doc.get("status", "N/A")

    embed = discord.Embed(
        title=f"Team: {doc.get('team_name', team_name)}", color=discord.Color.blue()
    )
    embed.add_field(name="Role", value=role, inline=True)
    embed.add_field(name="Status", value=status, inline=True)
    embed.add_field(name="Repo", value=repo, inline=False)
    embed.add_field(name="Members", value=members, inline=False)

    await message.channel.send(embed=embed)


import re
import discord
from datetime import datetime


async def handle_remove_member(message, entities):
    # === Backup extractor if NLP misses entities ===
    def extract_entities(text):
        member_pattern = r"(?:remove|delete)\s+([A-Za-z]+)"
        team_pattern = r"from\s+(?:team\s+)?(?:\"([^\"]+)\"|([A-Za-z\s]+))"

        member_match = re.search(member_pattern, text, re.IGNORECASE)
        team_match = re.search(team_pattern, text, re.IGNORECASE)

        member_name = member_match.group(1) if member_match else None
        team_name = team_match.group(1) or team_match.group(2) if team_match else None

        return {
            "member_name": member_name.strip() if member_name else None,
            "team_name": team_name.strip() if team_name else None,
        }

    # Update missing entities if needed
    if not entities.get("team_name") and not entities.get("team"):
        fallback = extract_entities(message.content)
        entities.update({k: v for k, v in fallback.items() if v})

    team_name = entities.get("team_name") or entities.get("team")
    name = entities.get("member_name") or entities.get("name")

    if not name:
        await message.channel.send("⚠️ Member name is required.")
        return

    if not team_name:
        await message.channel.send("⚠️ Team name is required to remove a member.")
        return

    try:
        # Case-insensitive team search
        team_doc = collection.find_one(
            {
                "$or": [
                    {
                        "team_name": {
                            "$regex": f"^{re.escape(team_name)}$",
                            "$options": "i",
                        }
                    },
                    {"team": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}},
                ]
            }
        )

        if not team_doc:
            await message.channel.send(f"🚫 Team **{team_name}** not found.")
            return

        if name not in team_doc.get("members", []):
            await message.channel.send(
                embed=discord.Embed(
                    title="Not Found",
                    description=f"No record found for member **{name}** in team **{team_doc.get('team_name', team_name)}**.",
                    color=discord.Color.gold(),
                )
            )
            return

        # Proceed with member removal
        result = collection.update_one(
            {"_id": team_doc["_id"]},
            {"$pull": {"members": name}, "$set": {"updated_at": datetime.utcnow()}},
        )

        if result.modified_count > 0:
            fields = [
                ("Member", name, True),
                ("Team", team_doc.get("team_name", team_name), True),
            ]
            embed = await create_success_embed(
                "Member Removed",
                f"**{name}** has been removed from **{team_doc.get('team_name', team_name)}**.",
                fields,
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(
                "⚠️ Member could not be removed. Please try again."
            )

    except Exception as e:
        logger.error(f"Error in handle_remove_member: {e}")
        await message.channel.send(f"❌ Database error: {e}")


async def handle_list_teams(message):
    """Handle listing teams intent"""
    try:
        # FIX: Try both field names
        teams_from_team_name = list(collection.distinct("team_name"))
        teams_from_team = list(collection.distinct("team"))

        # Combine and filter None values
        team_names = [t for t in teams_from_team_name + teams_from_team if t]

        # Remove duplicates
        team_names = list(set(team_names))

        if team_names:
            embed = discord.Embed(
                title="Teams in Database",
                description="\n• " + "\n• ".join(team_names),
                color=discord.Color.blue(),
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(
                embed=discord.Embed(
                    title="No Teams Found",
                    description="There are no teams in the database.",
                    color=discord.Color.gold(),
                )
            )
    except Exception as e:
        logger.error(f"Database error in handle_list_teams: {e}")
        await message.channel.send(f"❌ Database error: {e}")


async def handle_member_info(message, entities):
    # FIX: Standardize entity names
    name = entities.get("member_name") or entities.get("name")

    if not name:
        await message.channel.send(
            embed=discord.Embed(
                title="Missing Information",
                description="Member name is required.",
                color=discord.Color.red(),
            )
        )
        return

    try:
        # FIX: Get member document
        member_doc = collection.find_one({"name": name})

        if member_doc:
            embed = discord.Embed(
                title=f"Information for {name}", color=discord.Color.blue()
            )
            if "role" in member_doc:
                embed.add_field(name="Role", value=member_doc["role"], inline=True)
            if "team" in member_doc:
                embed.add_field(name="Team", value=member_doc["team"], inline=True)
            embed.set_footer(text="Fetched from database")
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(
                embed=discord.Embed(
                    title="Member Not Found",
                    description=f"No information found for member **{name}**.",
                    color=discord.Color.gold(),
                )
            )
    except Exception as e:
        logger.error(f"Database error in handle_member_info: {e}")
        await message.channel.send(f"❌ Database error: {e}")


async def start_create_team(message):
    global TEAM_CREATION_USER, TEAM_CREATION_INDEX, TEAM_CREATION_DATA
    if TEAM_CREATION_USER:
        await message.channel.send("⚠️ Another user is already in a team-creation flow.")
        return
    TEAM_CREATION_USER, TEAM_CREATION_INDEX, TEAM_CREATION_DATA = message.author, 0, {}
    await message.channel.send(
        f"Let’s create a team! What is the **{TEAM_CREATION_FIELDS[0]}**?"
    )


async def handle_delete_team(message, entities):
    """Handle deleting a team from the database."""
    team_name = entities.get("team_name") or entities.get("team")

    if not team_name:
        await message.channel.send("⚠️ Please specify the name of the team to delete.")
        return

    try:
        # Case-insensitive deletion
        delete_result = collection.delete_one(
            {
                "$or": [
                    {
                        "team_name": {
                            "$regex": f"^{re.escape(team_name)}$",
                            "$options": "i",
                        }
                    },
                    {"team": {"$regex": f"^{re.escape(team_name)}$", "$options": "i"}},
                ]
            }
        )

        if delete_result.deleted_count > 0:
            embed = await create_success_embed(
                "Team Deleted", f"Team **{team_name}** has been successfully deleted."
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(
                embed=discord.Embed(
                    title="Team Not Found",
                    description=f"No team found with the name **{team_name}**.",
                    color=discord.Color.gold(),
                )
            )

    except Exception as e:
        logger.error(f"Error deleting team {team_name}: {e}")
        await message.channel.send(f"❌ Database error while deleting team: {e}")


async def handle_create_team_interactive(message: discord.Message, data: dict):
    guild = message.guild
    if guild is None:
        return await message.channel.send("Must be run in a server.")

    team_name = data.get("team_name")
    if not team_name:
        return await message.channel.send("Team name can’t be empty.")
    if collection.find_one({"team_name": team_name}):
        return await message.channel.send(f"⚠️ Team **{team_name}** already exists.")

    await guild.chunk()

    names_db, matched, not_found = await parse_members(data.get("members", ""), guild)

    role_name = data.get("role")
    role_obj = None
    if role_name and role_name.lower() != "skip":
        role_obj = discord.utils.get(guild.roles, name=role_name)
        if not role_obj:
            try:
                role_obj = await guild.create_role(
                    name=role_name,
                    colour=discord.Color(random.randint(0, 0xFFFFFF)),
                    reason=f"Auto-created for team {team_name}",
                )
                print(f"[TEAM] Created role {role_obj.name}")
            except discord.Forbidden:
                await message.channel.send(
                    "I don’t have permission to create roles – continuing without role."
                )
            except Exception as exc:
                print(f"Role create error: {exc}")

    failed_assign = []
    if role_obj:
        for m in matched:
            try:
                await m.add_roles(role_obj, reason=f"Team {team_name} setup")
            except discord.Forbidden:
                failed_assign.append(m.display_name)
            except Exception as exc:
                failed_assign.append(m.display_name)
                print(f"[TEAM] Add role failed for {m}: {exc}")

    # create channel
    channel_name = re.sub(r"[^a-z0-9-]", "-", f"team-{team_name.lower()}")[:95]

    channel_obj = discord.utils.get(guild.text_channels, name=channel_name)
    if channel_obj is None:
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(view_channel=False),
            guild.me: discord.PermissionOverwrite(view_channel=True),
        }

        if role_obj:
            overwrites[role_obj] = discord.PermissionOverwrite(
                view_channel=True, send_messages=True, read_message_history=True
            )
        else:
            for m in matched:
                overwrites[m] = discord.PermissionOverwrite(
                    view_channel=True, send_messages=True, read_message_history=True
                )

        try:
            channel_obj = await guild.create_text_channel(
                channel_name,
                overwrites=overwrites,
                reason=f"Private channel for team {team_name}",
            )
        except discord.Forbidden:
            channel_obj = None
            logger.warning("⚠️  Missing permission to create channels.")
        except Exception as exc:
            channel_obj = None
            logger.error(f"Channel creation error: {exc}")

    doc = {
        "team_name": team_name,
        "role": role_name or "",
        "members": names_db,
        "repo": data.get("repo", "") if data.get("repo", "").lower() != "skip" else "",
        "status": data.get("status", "")
        if data.get("status", "").lower() != "skip"
        else "",
        "created_at": datetime.utcnow(),
    }
    collection.insert_one(doc)

    fields = [
        ("Role", role_name or "N/A", True),
        (
            "Members saved",
            "\n• " + "\n• ".join(names_db) if names_db else "None",
            False,
        ),
    ]
    if not_found:
        fields.append(("Not found in guild", ", ".join(not_found), False))
    if failed_assign:
        fields.append(("Couldn’t add role to", ", ".join(failed_assign), False))
    embed = await create_success_embed(
        "Team Created", f"Team **{team_name}** successfully set up.", fields
    )
    await message.channel.send(embed=embed)


async def create_server_role(message: discord.Message, entities: dict):
    guild = message.guild
    if guild is None:
        await message.channel.send("⚠️ Cannot create roles in DMs.")
        return

    if (
        isinstance(message.author, discord.Member)
        and not message.author.guild_permissions.manage_roles
    ):
        await message.channel.send("You don’t have permission to manage roles.")
        return

    role_name = (
        entities.get("role_name") or entities.get("name") or entities.get("role")
    )
    colour_raw = entities.get("colour") or entities.get("color")
    colour_obj = parse_hex_colour(colour_raw) or discord.Color.random()

    if not role_name:
        await message.channel.send("Role name missing – please specify a name.")
        return

    existing = discord.utils.find(
        lambda r: r.name.lower() == role_name.lower(), guild.roles
    )
    if existing:
        await message.channel.send(f"⚠️ Role **{existing.name}** already exists.")
        return

    try:
        new_role = await guild.create_role(
            name=role_name, colour=colour_obj, reason=f"Created by {message.author}"
        )
        fields = [
            ("Role", new_role.name, True),
            ("Colour", f"`#{new_role.colour.value:06x}`", True),
        ]
        embed = await create_success_embed(
            "Server Role Created", f"Role **{new_role.name}** has been added.", fields
        )
        await message.channel.send(embed=embed)
    except discord.Forbidden:
        await message.channel.send(
            "I lack the **Manage Roles** permission or my top role is too low."
        )
    except Exception as e:
        print(f"Error creating role: {e}")
        await message.channel.send(f"Unexpected error while creating the role: `{e}`")


async def handle_assign_role(message, entities):
    name = entities.get("member_name") or entities.get("name")
    role = entities.get("role")
    team = entities.get("team_name") or entities.get("team")

    if not name:
        await message.channel.send("Error: Name not provided")
        return
    if not team:
        await message.channel.send("Error: Team name not provided")
        return

    data = {"name": name, "role": role, "team": team, "updated_at": datetime.now()}

    try:
        # ── DB part (unchanged) ───────────────────────────────────────────
        existing_member = collection.find_one({"name": name, "team": team})
        if existing_member:
            collection.update_one({"name": name, "team": team}, {"$set": data})
        else:
            team_doc = collection.find_one({"team_name": team})
            if team_doc:
                collection.update_one(
                    {"team_name": team}, {"$addToSet": {"members": name}}
                )
            collection.insert_one(data)

        # ── NEW: actually give the role in Discord ────────────────────────
        guild = message.guild
        if guild and name:                                   # skip in DMs or if no role supplied
            # 1) resolve the member (by mention, nickname, username, …)
            member_obj = await resolve_member(team, guild)
            if member_obj:
                # 2) find or create the role object
                role_obj = discord.utils.get(guild.roles, name=name)
                if role_obj is None:
                    try:
                        role_obj = await guild.create_role(
                            name=name,
                            colour=discord.Color(random.randint(0, 0xFFFFFF)),
                            reason=f"Auto-created for team {team}",
                        )
                    except discord.Forbidden:
                        logger.warning("⚠️  No permission to create roles.")
                        role_obj = None
                    except Exception as exc:
                        logger.error(f"Role creation error: {exc}")
                        role_obj = None
                # 3) assign it
                if role_obj:
                    try:
                        await member_obj.add_roles(
                            role_obj,
                            reason=f"Assigned via bot (team {team})",
                        )
                    except discord.Forbidden:
                        logger.warning("⚠️  No permission to add roles to member.")
                    except Exception as exc:
                        logger.error(f"Add-roles error: {exc}")
            else:
                logger.info(f"Member '{name}' not found in guild; skipping Discord role assignment.")
        # ───────────────────────────────────────────────────────────────────

        fields = [
            ("Member", name, True),
            ("Role", role if role else "N/A", True),
            ("Team", team, True),
        ]
        embed = await create_success_embed(
            "Role Assignment Successful",
            f"**{name}** has been added to **{team}**"
            + (f" as **{role}**" if role else ""),
            fields,
        )
        await message.channel.send(embed=embed)

    except Exception as e:
        logger.error(f"Error in handle_assign_role: {e}")
        await message.channel.send(f"Database error: {e}")

client.run(os.getenv("DISCORD_BOT_TOKEN"))
