from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any
from datetime import datetime


class Tweet(BaseModel):
    text: str
    datetime: datetime


class ConcatTweets(BaseModel):
    ticker: str
    tweets: List[Tweet]


class RelevancyDecision(BaseModel):
    post_idx: int = Field(..., description="Index of the post in the list.")
    is_relevant: bool = Field(
        ..., description="Whether the post is relevant to the cryptocurrency."
    )
    reason: Optional[str] = Field(
        None, description="Short reason explaining the relevance decision."
    )


class Critique(BaseModel):
    decisions: List[RelevancyDecision] = Field(
        ...,
        description="One item for each post (in the same order), containing index, relevancy, and reason.",
    )


class SentimentResult(BaseModel):
    overall: Literal["Negative", "Positive", "Neutral"] = Field(
        ...,
        description="The overall sentiment of the Cryptocurrency based on the received data.",
    )
    score: float = Field(
        ...,
        description="Sentiment score based on your analysis. Between 0 and 1.",
    )
    warnings: str = Field(
        ...,
        description="Any additional considerances to bear in mind, such as potential warnings.",
    )


class Tweet(BaseModel):
    """
    Single Tweet from X/Twitter.
    """

    text: str = Field(..., description="The raw text of the tweet.")
    datetime: Any = Field(..., description="UTC datetime when the tweet was created.")
    username: Optional[str] = Field(
        None, description="Username (handle) of the tweet's author."
    )
    likes: Optional[int] = Field(None, description="Number of likes (if accessible).")
    retweets: Optional[int] = Field(
        None, description="Number of retweets (if accessible)."
    )


class RedditPost(BaseModel):
    """
    Single post from Reddit, e.g. from a search or a specific subreddit.
    """

    title_plus_text: str = Field(
        ..., description="Combination of the post title and body."
    )
    created_at: datetime = Field(
        ..., description="UTC datetime when the post was created."
    )
    upvotes: int = Field(..., description="Number of upvotes at the time of retrieval.")
    views: Optional[int] = Field(None, description="Number of views if available.")
    subreddit: Optional[str] = Field(
        None, description="Name of the subreddit (e.g. r/Crypto)."
    )
    author: Optional[str] = Field(None, description="Reddit username of the author.")


class DiscordMessage(BaseModel):
    """
    Single message from a Discord channel.
    """

    author: str = Field(..., description="Username (or ID) of the message's author.")
    content: str = Field(..., description="The textual content of the Discord message.")
    channel_id: str = Field(
        ..., description="ID or name of the channel where this message was posted."
    )
    timestamp: datetime = Field(
        ..., description="UTC time at which the message was sent."
    )
    mentions: List[str] = Field(
        default_factory=list, description="List of user mentions."
    )
    attachments: List[str] = Field(
        default_factory=list, description="List of attachment URLs, if any."
    )


class ConcatTweets(BaseModel):
    """
    Example aggregator for tweets.
    """

    ticker: str = Field(
        ..., description="Symbol of the cryptocurrency, e.g. BTC or ETH."
    )
    tweets: List[Tweet] = Field(..., description="List of Tweet objects.")


class ConcatRedditPosts(BaseModel):
    """
    Aggregator for Reddit posts.
    """

    ticker: str = Field(
        ..., description="Symbol of the cryptocurrency, e.g. BTC or ETH."
    )
    posts: List[RedditPost] = Field(..., description="List of retrieved Reddit posts.")


class ConcatDiscordMessages(BaseModel):
    """
    Aggregator for Discord messages.
    """

    server_name: Optional[str] = Field(None, description="Name of the Discord server.")
    channel_name: Optional[str] = Field(None, description="Channel name, if known.")
    messages: List[DiscordMessage] = Field(..., description="List of Discord messages.")


class SocialFeed(BaseModel):
    """
    Unified structure combining multiple social sources for the same ticker.
    """

    ticker: str = Field(..., description="Symbol of the cryptocurrency being tracked.")
    tweets: List[Tweet] = Field(
        default_factory=list, description="List of Tweet objects."
    )
    reddit_posts: List[RedditPost] = Field(
        default_factory=list, description="List of Reddit posts."
    )
    discord_messages: List[DiscordMessage] = Field(
        default_factory=list, description="List of Discord messages."
    )


class CultinessResult(BaseModel):
    score: float = Field(
        ..., description="The overall cultiness score, between 0.0 and 1.0."
    )
    analysis: str = Field(..., description="Reason for the provided cultiness score.")
    warnings: str = Field(..., description="Any additional information or warnings.")


class SentimentResult(BaseModel):
    """
    Overall sentiment result for a cryptocurrency, as you had in your example.
    """

    overall: Literal["Negative", "Positive", "Neutral"] = Field(
        ...,
        description="The overall sentiment of the Cryptocurrency based on the received data.",
    )
    score: float = Field(
        ..., description="Sentiment score based on your analysis. Between 0.0 and 1.0"
    )
    confidence: float = Field(
        ..., description="How confident you are in your analysis. Between 0.0 and 1.0"
    )
    warnings: str = Field(
        ...,
        description="Any additional considerances to bear in mind, such as potential warnings.",
    )


class SentimentBreakdown(BaseModel):
    """
    Optional deeper breakdown by source or subcategory.
    For example, you might want a separate sentiment result for tweets, reddit, discord, etc.
    """

    tweets_sentiment: Optional[SentimentResult] = Field(
        None, description="Sentiment analysis result just for Tweets."
    )
    reddit_sentiment: Optional[SentimentResult] = Field(
        None, description="Sentiment analysis result just for Reddit."
    )
    discord_sentiment: Optional[SentimentResult] = Field(
        None, description="Sentiment analysis result just for Discord."
    )
    combined_sentiment: Optional[SentimentResult] = Field(
        None, description="An aggregated or averaged sentiment across all sources."
    )


###############################################################################
# 3) Agent Input/Output
###############################################################################


class SentimentAgentInput(BaseModel):
    """
    Input to your sentiment agent's main method: which ticker, how many items, etc.
    """

    ticker: str = Field(..., description="Crypto ticker symbol, e.g. BTC, ETH, etc.")
    hours: int = Field(24, description="Number of hours to look back for social data.")
    max_tweets: int = Field(10, description="Max tweets to retrieve from X.")
    max_reddit_posts: int = Field(10, description="Max Reddit posts to retrieve.")
    max_discord_messages: int = Field(
        10, description="Max Discord messages to retrieve from relevant channels."
    )


class SentimentAgentOutput(BaseModel):
    """
    Final structured output from your sentiment agent.
    """

    feed: SocialFeed = Field(..., description="Collected data from multiple sources.")
    breakdown: SentimentBreakdown = Field(
        ..., description="A breakdown of sentiment by source, plus combined."
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the sentiment was generated (UTC).",
    )


class TradeSignal(BaseModel):
    """
    A class to capture a trading signal based on sentiment.
    """

    ticker: str = Field(..., description="Crypto ticker symbol.")
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        ..., description="Recommended action."
    )
    confidence: float = Field(..., description="Confidence level between 0.0 and 1.0.")
    rationale: str = Field(
        ..., description="Short explanation of why this action is recommended."
    )
