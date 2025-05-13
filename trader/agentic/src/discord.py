import discord
from pydantic import BaseModel, Field
from typing import List, Optional

from trader.config import Config

class ServerInput(BaseModel):
    guild_id: int = Field(..., description="The ID of the server (guild)")
    channels: Optional[List[int]] = Field(None, description="List of channel IDs to fetch messages from. If None, fetch from all accessible channels.")


class MessageOutput(BaseModel):
    author: str = Field(..., description="The author of the message")
    content: str = Field(..., description="The content of the message")
    timestamp: str = Field(..., description="The time the message was sent")
    channel_id: int = Field(..., description="The ID of the channel where the message was sent")
    guild_id: int = Field(..., description="The ID of the server (guild) where the message was sent")


class DiscordClient(discord.Client):
    def __init__(self, intents: discord.Intents, servers_to_scrape: List[ServerInput], *args, **kwargs):
        super().__init__(intents=intents, *args, **kwargs)
        self.token = Config.DISCORD_BOT_TOKEN
        self.servers_to_scrape = servers_to_scrape


    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")

        all_messages = []

        for server in self.servers_to_scrape:
            guild = self.get_guild(server.guild_id)
            if guild is None:
                print(f"Could not access guild with ID {server.guild_id}")
                continue

            print(f"Accessing guild: {guild.name} ({guild.id})")
            channels = guild.text_channels if server.channels is None else [
                guild.get_channel(ch_id) for ch_id in server.channels if guild.get_channel(ch_id)
            ]

            for channel in channels:
                if channel is None:
                    continue

                print(f"Fetching messages from channel: {channel.name} ({channel.id})")
                messages = await self.fetch_messages(channel, guild.id)
                all_messages.extend(messages)

        print("Finished fetching messages.")
        self.pretty_print_messages(all_messages)

        await self.close()


    async def fetch_messages(self, channel: discord.TextChannel, guild_id: int) -> List[MessageOutput]:
        fetched_messages = []
        try:
            async for message in channel.history(limit=100):
                msg_output = MessageOutput(
                    author=str(message.author),
                    content=message.content,
                    timestamp=str(message.created_at),
                    channel_id=channel.id,
                    guild_id=guild_id,
                )
                fetched_messages.append(msg_output)
        except discord.Forbidden:
            print(f"Permission denied to read message history in {channel.name} ({channel.id}).")
        except discord.HTTPException as e:
            print(f"Failed to fetch messages from {channel.name} ({channel.id}): {e}")

        return fetched_messages

    def pretty_print_messages(self, messages: List[MessageOutput]):
        for msg in messages:
            print(f"[{msg.timestamp}] {msg.author} in channel {msg.channel_id} (server {msg.guild_id}): {msg.content}")

if __name__ == "__main__":
    TURBO_SERVER_ID = '1102800896393498685'
    CHANNEL_ID = '1102800896980680816'
    # intents = discord.Intents.default()
    # intents.message_content = True

    # servers = [
    #     ServerInput(guild_id=123456789012345678, channels=[234567890123456789, 345678901234567890]),
    #     ServerInput(guild_id=987654321098765432),
    # ]

    # client = DiscordClient(intents=intents, servers_to_scrape=servers)
    # client.run(client.token)

    import discord
    import asyncio

    # Replace 'YOUR_BOT_TOKEN' with your bot token
    BOT_TOKEN = 'YOUR_BOT_TOKEN'

    # Replace 'CHANNEL_ID' with the ID of the channel you want to scrape
    CHANNEL_ID = 123456789012345678  # Replace with your channel's ID

    class MyClient(discord.Client):
        async def on_ready(self):
            print(f'Logged in as {self.user}')
            channel = self.get_channel(CHANNEL_ID)
            if channel:
                messages = await channel.history(limit=10).flatten()
                for message in messages:
                    print(f'{message.author}: {message.content}')
            else:
                print("Channel not found. Check your CHANNEL_ID.")
            await self.close()

    # Initialize and run the bot
    client = MyClient(intents=discord.Intents.default())
    client.run(BOT_TOKEN)

