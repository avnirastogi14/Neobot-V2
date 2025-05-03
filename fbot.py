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
    • `create_team <team_name> [role] [members...] [repo] [status]`: Create a new team (e.g., `create_team Innovation Crew`).
    • `assign_role <member_name> <role> to <team_name>`: Assign a role to a member (e.g., `assign_role Alice Lead Developer to Innovation Crew`).
    • `update_team <team_name>`: Update details for a team (e.g., `update_team Innovation Crew`).
    • `remove_member <member_name> from <team_name>`: Remove a member from a team (e.g., `remove_member Bob from Innovation Crew`).
    • `list_teams`: Show all teams (e.g., `list_teams`).
    • `show_role_info <team_name>`: Show information about a specific team (e.g., `show_role_info Innovation Crew`).
    • `get_member_info <member_name>`: Show information about a specific member (e.g., `get_member_info Alice`).
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
        logger.info(f"Intent predicted: {intent}, Entities: {entities}") # Log prediction result
    except Exception as e:
        await message.channel.send(f"❌ Prediction error: `{str(e)}`")
        return

    # FIX: Log both intent and entities to help with debugging
    logger.info(f"Handling intent: {intent} with entities: {entities}")

    if intent == "assign_role":
        await handle_assign_role(message, entities)
    elif intent == "update_team": # This intent might still be triggered by old phrases
        await handle_update_team(message, entities) # Redirect to the new handler
    elif intent == "show_role_info":
        await handle_show_roles(message, entities)
    elif intent == "remove_member":
        await handle_remove_member(message, entities)
    elif intent == "list_teams":
        await handle_list_teams(message)
    elif intent == "get_member_info":
        await handle_member_info(message, entities)
    elif intent == "create_team":
        logger.info("Calling handle_create_team function.")
        await start_create_team(message)
    elif intent == "help":
        await client.get_command('bothelp').invoke(await client.get_context(message))
    elif intent == "greeting":
        await message.channel.send(f"👋 Hello {message.author.display_name}!")
    else:
        await message.channel.send(f"🤔 Sorry, I didn't quite understand that. Try using `!bothelp` for examples.")

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

        embed = discord.Embed(
            title="✅ Role Assignment Successful",
            color=discord.Color.green()
        )
        embed.add_field(name="Member", value=name, inline=True)
        if role:
            embed.add_field(name="Role", value=role, inline=True)
        if team:
            embed.add_field(name="Team", value=team, inline=True)
        await message.channel.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in handle_assign_role: {e}")
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
                    await message.channel.send(f"✅ Updated members for **{team_name}**.")
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
                    await message.channel.send(embed=discord.Embed(
                        title="✅ Team Updated",
                        description=f"Updated **{field_to_update}** for **{team_name}** to:\n{new_value}",
                        color=discord.Color.green()
                    ))
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

async def handle_show_roles(message, entities):
    # FIX: Standardize entity names
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

    # FIX: Check if members is in the document
    members_list = doc.get("members", [])
    members = "\n".join(members_list) if members_list else "No members"

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
    # FIX: Standardize entity names
    team_name = entities.get("team_name") or entities.get("team")
    name = entities.get("member_name") or entities.get("name")

    if not name:
        await message.channel.send("Member name is required.")
        return

    try:
        # First remove member from the team's member list
        if team_name:
            result = collection.update_one(
                {"team_name": team_name},
                {"$pull": {"members": name}}
            )

            if result.modified_count == 0:
                # Try with "team" field
                result = collection.update_one(
                    {"team": team_name},
                    {"$pull": {"members": name}}
                )

        # Then remove the member document
        result = collection.delete_one({"name": name})

        if result.deleted_count > 0:
            await message.channel.send(embed=discord.Embed(
                title="✅ Member Removed",
                description=f"Removed **{name}** from the database.",
                color=discord.Color.green()
            ))
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
    global TEAM_CREATION_USER, TEAM_CREATION_DATA
    if TEAM_CREATION_USER is not None:
        await message.channel.send("A team creation process is already in progress. Please complete that first.")
        return
    TEAM_CREATION_USER = message.author
    TEAM_CREATION_DATA = {}
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

    members = [member.strip() for member in members_str.split(',')] if members_str else []

    team_info = {
        "team_name": team_name,
        "role": role,
        "members": members,
        "repo": repo,
        "status": status,
        "created_at": datetime.datetime.now()
    }

    try:
        collection.insert_one(team_info)
        embed = discord.Embed(
            title="✅ Team Created",
            description=f"Team **{team_name}** has been created successfully with the following details:",
            color=discord.Color.green()
        )
        embed.add_field(name="Role", value=role if role else "N/A", inline=False)
        embed.add_field(name="Members", value=", ".join(members) if members else "N/A", inline=False)
        embed.add_field(name="Repo", value=repo if repo else "N/A", inline=False)
        embed.add_field(name="Status", value=status if status else "N/A", inline=False)
        await message.channel.send(embed=embed)
    except Exception as e:
        logger.error(f"Error creating team {team_name}: {e}")
        await message.channel.send(f"❌ Could not create team due to a database error: {e}")

client.run(os.getenv('DISCORD_BOT_TOKEN'))