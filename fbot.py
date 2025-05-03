import discord
from discord.ext import commands
from pymongo import MongoClient
from fmodel import predict, INTENTS_LIST
import asyncio
import random
import os
import datetime
from dotenv import load_dotenv
import logging  # Import the logging module

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
    logger.info(f"✅ {client.user} is online | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
async def bothelp(ctx):
    embed = discord.Embed(
        title="TestBotA Help",
        description="I understand natural language commands for team and role management.",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="Example Commands",
        value=( """ **Available Commands**:
    • `Create a new team`.
    • `Assign role to a member in the team`.
    • `Update Team Details - Members, Repository, Status`.
    • `Remove a Team Member`.
    • `List all teams in the database`.
    • `Show a specific team's details`.
    • `!exit`: To exit from current command.
    • `!bothelp`: Show this help message. """),
        inline=False
    )
    embed.add_field(name="Technical Commands", value="• !ping - Check if bot is responsive", inline=False)
    embed.set_footer(text="I use ML to understand your requests")
    await ctx.send(embed=embed)

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    await client.process_commands(message)

    if message.content.startswith(client.command_prefix):
        return

    # Check for ongoing team creation process
    global TEAM_CREATION_USER, TEAM_CREATION_DATA, TEAM_CREATION_INDEX
    if TEAM_CREATION_USER == message.author and TEAM_CREATION_INDEX < len(TEAM_CREATION_FIELDS):
        field = TEAM_CREATION_FIELDS[TEAM_CREATION_INDEX]
        TEAM_CREATION_DATA[field] = message.content
        TEAM_CREATION_INDEX += 1
        
        if TEAM_CREATION_INDEX < len(TEAM_CREATION_FIELDS):
            await message.channel.send(f"Great! Now, what is the **{TEAM_CREATION_FIELDS[TEAM_CREATION_INDEX]}**? (or type 'skip' to leave empty)")
        else:
            await handle_create_team_interactive(message, TEAM_CREATION_DATA)
            TEAM_CREATION_USER = None
            TEAM_CREATION_DATA = {}
            TEAM_CREATION_INDEX = 0
        return

    # Cache to avoid repeat processing
    cache_key = f"{message.channel.id}:{message.id}"
    if not hasattr(client, 'processed_messages'):
        client.processed_messages = set()
    if cache_key in client.processed_messages:
        return
    client.processed_messages.add(cache_key)
    if len(client.processed_messages) > 100:
        client.processed_messages = set(list(client.processed_messages)[-80:])

    # ***CRUCIAL FIX: Check again if the message is from the bot***
    if message.author == client.user:
        return

    # ML Prediction
    try:
        prediction_result = predict(message.content)
        intent = prediction_result.get("intent")
        entities = prediction_result.get("entities", {})
        confidence = prediction_result.get("confidence", "low")
        logger.info(f"Intent predicted: {intent}, Entities: {entities}, Confidence: {confidence}")
        
        # Don't trigger help unless directly asked
        if intent == "help" and confidence == "high":
            await client.get_command('bothelp').invoke(await client.get_context(message))
            return
        if intent == "exit" and confidence == "high":
            await message.channel.send(f"⌚❌ Exiting Command - Command Aborted!")
            return
    except Exception as e:
        await message.channel.send(f"❌ Prediction error: `{str(e)}`")
        return

    # FIX: Log both intent and entities to help with debugging
    logger.info(f"Handling intent: {intent} with entities: {entities}")

    if intent == "assign_role":
        await handle_assign_role(message, entities)
    elif intent == "update_team_repo":
        await handle_update_team_repo(message, entities)
    elif intent == "update_team": # This intent might still be triggered by old phrases
        await handle_update_team(message, entities) # Redirect to the new handler
    elif intent == "show_team_info": # Re-using the existing intent for showing team info
        await handle_show_team_info(message, entities) # New handler
    elif intent == "remove_member":
        await handle_remove_member(message, entities)
    elif intent == "list_teams":
        await handle_list_teams(message)
    elif intent == "get_member_info":
        await handle_member_info(message, entities)
    elif intent == "create_team":
        logger.info("Calling handle_create_team function.")
        await start_create_team(message)
    elif intent == "greeting" and confidence == "high":
        await message.channel.send(f"👋 Hello {message.author.display_name}!")
    elif not intent or intent == "unknown" or confidence == "low":
        # Don't respond with the help message for low confidence predictions
        pass
    
async def create_success_embed(title, description, fields=None):
    """Create a consistent success embed that stands out"""
    embed = discord.Embed(
        title=f"✅ {title}",
        description=description,
        color=discord.Color.green()
    )
    if fields:
        for name, value, inline in fields:
            if value:  # Only add field if value exists
                embed.add_field(name=name, value=value, inline=inline)
    
    # Add timestamp and border to make it stand out more
    embed.timestamp = datetime.datetime.now()
    return embed

async def handle_assign_role(message, entities):
    """Handle role assignment intent"""
    # FIX: Standardize entity names from model
    name = entities.get("member_name") or entities.get("name")
    role = entities.get("role")
    team = entities.get("team_name") or entities.get("team")

    if not name:
        await message.channel.send("❌ Error: Name not provided")
        return

    if not team:
        await message.channel.send("❌ Error: Team name not provided")
        return

    data = {
        "name": name,
        "role": role,
        "team": team,
        "updated_at": datetime.datetime.now()
    }

    try:
        # FIX: Find if user exists in the team and update, otherwise add
        existing_member = collection.find_one({"name": name, "team": team})

        if existing_member:
            # Update existing member
            collection.update_one(
                {"name": name, "team": team},
                {"$set": data}
            )
        else:
            # Add to team's members list first
            team_doc = collection.find_one({"team_name": team})
            if team_doc:
                collection.update_one(
                    {"team_name": team},
                    {"$addToSet": {"members": name}}
                )

            # Then add member document
            collection.insert_one(data)

        # Use the new success embed format
        fields = [
            ("Member", name, True),
            ("Role", role if role else "N/A", True),
            ("Team", team, True)
        ]
        embed = await create_success_embed(
            "Role Assignment Successful", 
            f"**{name}** has been added to **{team}**" + (f" as **{role}**" if role else ""),
            fields
        )
        await message.channel.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in handle_assign_role: {e}")
        await message.channel.send(f"❌ Database error: {e}")

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
            {"$set": {"repo": repo, "updated_at": datetime.datetime.now()}}
        )
        
        if result.modified_count == 0:
            # Try with "team" field
            result = collection.update_one(
                {"team": team_name},
                {"$set": {"repo": repo, "updated_at": datetime.datetime.now()}}
            )
        
        if result.modified_count > 0:
            # Get updated team info
            team_doc = collection.find_one({"team_name": team_name}) or collection.find_one({"team": team_name})
            
            fields = [
                ("Repository", repo, False),
                ("Team", team_name, True)
            ]
            embed = await create_success_embed(
                "Repository Updated", 
                f"Repository URL for **{team_name}** has been updated.",
                fields
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

    await message.channel.send(f"Which field of **{team_name}** would you like to update? (role, repo, status, members)")

    def check(m):
        return m.author == message.author and m.channel == message.channel and m.content.lower() in ["role", "repo", "status", "members"]

    try:
        msg = await client.wait_for('message', check=check, timeout=30.0)
        field_to_update = msg.content.lower()

        if field_to_update == "members":
            await message.channel.send(f"Please provide the new list of **members**, separated by commas.")
            try:
                members_msg = await client.wait_for('message', check=lambda m: m.author == message.author and m.channel == message.channel, timeout=30.0)
                members_list = [m.strip() for m in members_msg.content.split(",")]
                update_result = collection.update_one(
                    {"$or": [{"team_name": team_name}, {"team": team_name}]},
                    {"$set": {"members": members_list, "updated_at": datetime.datetime.utcnow()}}
                )
                if update_result.modified_count > 0:
                    team_doc = collection.find_one({"$or": [{"team_name": team_name}, {"team": team_name}]})
                    
                    fields = [
                        ("Team", team_name, True),
                        ("Members", "\n• " + "\n• ".join(members_list), False)
                    ]
                    embed = await create_success_embed(
                        "Team Members Updated", 
                        f"Members for **{team_name}** have been updated.",
                        fields
                    )
                    await message.channel.send(embed=embed)
                else:
                    await message.channel.send(f"⚠️ Could not find team **{team_name}** to update members.")
            except asyncio.TimeoutError:
                await message.channel.send("❌ Update members timed out.")
                return

        else:
            await message.channel.send(f"What is the new value for **{field_to_update}**?")
            try:
                value_msg = await client.wait_for('message', check=lambda m: m.author == message.author and m.channel == message.channel, timeout=30.0)
                new_value = value_msg.content

                update_data = {"updated_at": datetime.datetime.utcnow()}
                update_data[field_to_update] = new_value

                result = collection.update_one(
                    {"$or": [{"team_name": team_name}, {"team": team_name}]},
                    {"$set": update_data}
                )

                if result.matched_count:
                    team_doc = collection.find_one({"$or": [{"team_name": team_name}, {"team": team_name}]})
                    
                    fields = [
                        ("Team", team_name, True),
                        (field_to_update.capitalize(), new_value, True)
                    ]
                    embed = await create_success_embed(
                        "Team Updated", 
                        f"Updated **{field_to_update}** for **{team_name}**.",
                        fields
                    )
                    await message.channel.send(embed=embed)
                else:
                    await message.channel.send(embed=discord.Embed(
                        title="Team Not Found",
                        description=f"No team found with name **{team_name}**.",
                        color=discord.Color.gold()
                    ))
            except asyncio.TimeoutError:
                await message.channel.send("❌ Update timed out.")
                return

    except asyncio.TimeoutError:
        await message.channel.send("❌ No field specified to update in time.")

async def handle_show_team_info(message, entities):
    team_name = entities.get("team_name") or entities.get("team")

    if not team_name:
        await message.channel.send("Please specify the team name.")
        return

    # FIX: First try with "team_name" field
    doc = collection.find_one({"team_name": team_name})

    if not doc:
        # Try with "team" field if not found
        doc = collection.find_one({"team": team_name})

    if not doc:
        await message.channel.send(embed=discord.Embed(
            title="Team Not Found",
            description=f"No data for team **{team_name}**.",
            color=discord.Color.gold()
        ))
        return

    members_list = doc.get("members", [])
    members = "\n• " + "\n• ".join(members_list) if members_list else "No members"

    role = doc.get("role", "N/A")
    repo = doc.get("repo", "N/A")
    status = doc.get("status", "N/A")

    embed = discord.Embed(
        title=f"Team: {team_name}",
        color=discord.Color.blue()
    )
    embed.add_field(name="Role", value=role, inline=True)
    embed.add_field(name="Status", value=status, inline=True)
    embed.add_field(name="Repo", value=repo, inline=False)
    embed.add_field(name="Members", value=members, inline=False)

    await message.channel.send(embed=embed)

async def handle_remove_member(message, entities):
    # FIX: entity names
    team_name = entities.get("team_name") or entities.get("team")
    name = entities.get("member_name") or entities.get("name")

    if not name:
        await message.channel.send("Member name is required.")
        return

    try:
        if team_name:
            result = collection.update_one(
                {"team_name": team_name},
                {"$pull": {"members": name}}
            )

            if result.modified_count == 0:
                result = collection.update_one(
                    {"team": team_name},
                    {"$pull": {"members": name}}
                )

        result = collection.delete_one({"name": name})

        if result.deleted_count > 0:
            fields = [
                ("Member", name, True),
                ("Team", team_name if team_name else "All teams", True)
            ]
            embed = await create_success_embed(
                "Member Removed", 
                f"**{name}** has been removed" + (f" from **{team_name}**" if team_name else " from all teams"),
                fields
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(embed=discord.Embed(
                title="Not Found",
                description=f"No record found for member **{name}**.",
                color=discord.Color.gold()
            ))

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
                color=discord.Color.blue()
            )
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(embed=discord.Embed(
                title="No Teams Found",
                description="There are no teams in the database.",
                color=discord.Color.gold()
            ))
    except Exception as e:
        logger.error(f"Database error in handle_list_teams: {e}")
        await message.channel.send(f"❌ Database error: {e}")

async def handle_member_info(message, entities):
    # FIX: Standardize entity names
    name = entities.get("member_name") or entities.get("name")

    if not name:
        await message.channel.send(embed=discord.Embed(
            title="Missing Information",
            description="Member name is required.",
            color=discord.Color.red()
        ))
        return

    try:
        # FIX: Get member document
        member_doc = collection.find_one({"name": name})

        if member_doc:
            embed = discord.Embed(
                title=f"Information for {name}",
                color=discord.Color.blue()
            )
            if "role" in member_doc:
                embed.add_field(name="Role", value=member_doc["role"], inline=True)
            if "team" in member_doc:
                embed.add_field(name="Team", value=member_doc["team"], inline=True)
            embed.set_footer(text="Fetched from database")
            await message.channel.send(embed=embed)
        else:
            await message.channel.send(embed=discord.Embed(
                title="Member Not Found",
                description=f"No information found for member **{name}**.",
                color=discord.Color.gold()
            ))
    except Exception as e:
        logger.error(f"Database error in handle_member_info: {e}")
        await message.channel.send(f"❌ Database error: {e}")

async def start_create_team(message):
    global TEAM_CREATION_USER, TEAM_CREATION_DATA, TEAM_CREATION_INDEX
    if TEAM_CREATION_USER is not None:
        await message.channel.send("A team creation process is already in progress. Please complete that first.")
        return
    TEAM_CREATION_USER = message.author
    TEAM_CREATION_DATA = {}
    TEAM_CREATION_INDEX = 0
    await message.channel.send(f"Let's create a new team! What is the **{TEAM_CREATION_FIELDS[0]}**?")

async def handle_create_team_interactive(message, team_data):
    """Handles the interactive creation of a new team."""
    team_name = team_data.get("team_name")
    role = team_data.get("role")
    members_str = team_data.get("members")
    repo = team_data.get("repo")
    status = team_data.get("status")

    if not team_name:
        await message.channel.send("Team name is mandatory.")
        return

    if collection.find_one({"team_name": team_name}):
        await message.channel.send(f"Team **{team_name}** already exists.")
        return

    members = [member.strip() for member in members_str.split(',')] if members_str and members_str.lower() != "skip" else []

    team_info = {
        "team_name": team_name,
        "role": role if role and role.lower() != "skip" else "",
        "members": members,
        "repo": repo if repo and repo.lower() != "skip" else "",
        "status": status if status and status.lower() != "skip" else "",
        "created_at": datetime.datetime.now()
    }

    try:
        collection.insert_one(team_info)
        
        # Use the new success embed format for consistency
        fields = [
            ("Role", team_info["role"] if team_info["role"] else "N/A", True),
            ("Status", team_info["status"] if team_info["status"] else "N/A", True),
            ("Repo", team_info["repo"] if team_info["repo"] else "N/A", False),
            ("Members", "\n• " + "\n• ".join(members) if members else "No members yet", False)
        ]
        
        embed = await create_success_embed(
            "Team Created", 
            f"Team **{team_name}** has been created successfully!",
            fields
        )
        await message.channel.send(embed=embed)
    except Exception as e:
        logger.error(f"Error creating team {team_name}: {e}")
        await message.channel.send(f"❌ Could not create team due to a database error: {e}")

client.run(os.getenv('DISCORD_BOT_TOKEN'))