import json
import logging
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from agentic.opensearch import CritiqueSearch
from agentic.prompts import PromptHandler
from agentic.utils import get_structured_response
from trader.agentic.data_models import SentimentResult, Critique
from trader.agentic.src.praw_sentiment import PrawRedditClient
from trader.agentic.src.xclient import XClient


class Tweet(BaseModel):
    """
    Simple struct for tweets we retrieve from X/Twitter.
    """

    text: str
    created_at: datetime


class RedditPost(BaseModel):
    """
    Simple struct for reddit posts we retrieve.
    """

    concat: str
    created_at: datetime
    upvotes: int
    views: Optional[int] = None
    comments: Optional[List[str]] = None


class ModelInsert(BaseModel):
    """
    Aggregate of retrieved social data for a given ticker.
    We might want to store Twitter and Reddit data in separate sub-lists.
    """

    ticker: str
    token_name: str
    num_holders: int = 0
    holders_percent_increase_24h: float = 0.0
    volume_24h: float = 0.0
    buy_sell_ratio_24h: float = 0.0
    volume_marketcap_ratio: float = 0.0
    market_cap: float = 0.0
    avg_holders_distribution: float = 0.0
    tweets: List[Tweet] = []
    reddit_posts: List[RedditPost] = []


class SentimentAgentInput(BaseModel):
    """
    The input for the SentimentAgent’s sentiment analysis method.
    We specify how many tweets/posts to fetch, plus the ticker.
    """

    ticker: str = Field(
        ..., description="Which cryptocurrency ticker symbol to analyze."
    )
    token_name: Optional[str] = Field(
        None, description="Human-readable token name (e.g., 'Ethereum')."
    )
    max_tweets: int = Field(10, description="Number of tweets to fetch from X.")
    max_reddit_posts: int = Field(
        10, description="Number of posts to fetch from Reddit."
    )
    hours: int = Field(
        24, description="How far back (in hours) we look for X and Reddit data."
    )
    num_holders: int = None
    holders_percent_increase_24h: float = None
    avg_holders_distribution: float = None
    market_cap: float = None
    volume_24h: float = None
    buy_sell_ratio_24h: float = None
    volume_marketcap_ratio: float = None
    found_subreddit: str = None
    links: List[str] = Field(
        None, description="Any auxiliary links to bolster deepsearch."
    )


class SentimentAgentOutput(BaseModel):
    """
    The final structured output from the SentimentAgent, combining
    the social feed plus the final sentiment.
    """

    feed: ModelInsert
    sentiment: SentimentResult


class SentimentAgent:
    """
    Sentiment AI Agent: Twitter, Reddit, Discord, Telegram parsing for sentiment of a given cryptocurrency.
    """

    def __init__(
        self,
        xclient: XClient = None,
        prompt_handler: PromptHandler = None,
        include_x: bool = False,
    ):
        # do not instantiate
        if include_x:
            self.xclient = xclient

        self.include_x = include_x
        self.prompt_handler = prompt_handler or PromptHandler()
        self.opensearch_client = CritiqueSearch()
        self.praw_client = PrawRedditClient()

        self.TABLE_NAME = "sentiment"

    def gather_social_data(
        self, input_payload: SentimentAgentInput, max_retries: int = 2
    ) -> ModelInsert:
        """
        Pull from X (Twitter) and Reddit given the user's request (ticker, hours, max items).
        Return a ModelInsert.
        """
        tweets = []
        relevant_posts = []

        query = f"{input_payload.token_name}"

        if self.include_x:
            tweets = self.xclient.get_recent_tweets(
                query=query,
                hours=input_payload.hours,
                max_results=input_payload.max_tweets,
            )

        token_specific_sub = (
            input_payload.found_subreddit
            if input_payload.found_subreddit
            else input_payload.token_name.lower().replace(" ", "")
        )
        subreddits_to_search = ["CryptoCurrency", "memecoins", token_specific_sub]

        logging.info("Searching relevant media...")
        # maybe have it run on each source separately -- perform bias factor calc manually
        try:
            all_reddit_posts = []
            for sr in subreddits_to_search:
                top_posts = self.praw_client.search_subreddit_posts(
                    subreddit_name=sr,
                    query=input_payload.token_name,
                    sort="top",
                    limit=(
                        input_payload.max_reddit_posts
                        if sr != "CryptoCurrency" or sr != "memecoins"
                        else 3
                    ),
                )

                for p in top_posts:
                    comments = self.praw_client.get_submission_comments(
                        sr, post_id=p["id"], limit=10
                    )
                    concat = f"Subreddit: r/{sr}\nPost Title: {p.get('title', '')}\n\nPost Text: {p.get('selftext', '')}\n\nComments:\n"

                    for c in comments:
                        score = c.score
                        downvotes = c.downs
                        body = c.body
                        concat += f"Score: {score}, Number of Downvotes: {downvotes}  Comment: {body}\n"

                    reddit_post = RedditPost(
                        concat=concat,
                        created_at=datetime.fromtimestamp(p["created_utc"]),
                        upvotes=p.get("score", 0),
                        # comments=comments,
                        # views=None
                    )

                    all_reddit_posts.append(reddit_post)

                # # Critique
                # filtered = self.critique_context(input_payload.ticker, new_posts)
                # relevant_posts.extend(filtered)
                # attempts += 1

        except Exception as e:
            logging.error(f"Error building post data: {e}")

        try:
            logging.info("Critiquing context...")
            relevant_posts = self.critique_context(
                input_payload.ticker, all_reddit_posts
            )

        except Exception as e:
            logging.error(f"Error critiquing context: {e}")
            relevant_posts = all_reddit_posts

        return ModelInsert(
            ticker=input_payload.ticker,
            token_name=input_payload.token_name,
            num_holders=input_payload.num_holders,
            volume_24h=input_payload.volume_24h,
            buy_sell_ratio_24h=input_payload.buy_sell_ratio_24h,
            volume_marketcap_ratio=input_payload.volume_marketcap_ratio,
            holders_percent_increase_24h=input_payload.holders_percent_increase_24h,
            tweets=tweets,
            reddit_posts=relevant_posts,
            market_cap=input_payload.market_cap,
            avg_holders_distribution=input_payload.avg_holders_distribution,
        )

    def critique_context(
        self, ticker: str, posts: List[RedditPost]
    ) -> List[RedditPost]:
        """
        Returns a filtered list of only the relevant posts - decided by critique model. (Examines ticker relevance)
        """
        if not posts:
            return []

        reddit_text_list = [
            f"Post Index: {i}\nTitle Plus Text: {p.concat}" for i, p in enumerate(posts)
        ]
        big_reddit_text = "\n\n".join(reddit_text_list)

        sys_prompt = PromptHandler().get_prompt(
            "critique_instruct", ticker=ticker, reddit_posts=big_reddit_text
        )

        critique_obj = get_structured_response(sys_prompt, Critique, "o3-mini")

        try:
            decisions = critique_obj.decisions

        # fallback 2
        except json.JSONDecodeError as e:
            logging.error(f"Issue extracting relevant posts. Error: {e}")
            return posts

        relevant_posts = []

        for d in decisions:
            idx = d.post_idx
            if d.is_relevant and 0 <= idx < len(posts):
                relevant_posts.append(posts[idx])

        return relevant_posts

    def opensearch(self, ticker: str, token_name: str, links: List[str] = None):
        search_prompt = self.prompt_handler.get_prompt(
            template="opensearch",
            ticker=ticker,
            token_name=token_name,
            current_date=datetime.now(),
            links=links,
        )

        search_results = self.opensearch_client.search(
            prompt=search_prompt, source_blacklist=[], output_format=False
        )

        return search_results

    def run_sentiment_analysis(self, feed: ModelInsert) -> SentimentResult:
        """
        Call 4o -- parse results into a SentimentResult.
        """

        tweets_text = "\n".join([f"[Tweet: {t.text}]" for t in feed.tweets])
        reddit_text = "\n".join([f"[Reddit: {r.concat}]" for r in feed.reddit_posts])

        logging.info("Running opensearch..")
        logging.info("Running opensearch..")
        search_results = self.opensearch(feed.ticker, feed.token_name)

        logging.info("Analyzing sentiment..")
        system_message = self.prompt_handler.get_prompt(
            template="sentiment_instruct",
            ticker=feed.ticker,
            token_name=feed.token_name,
            media=(tweets_text + "\n" + reddit_text),
            opensearch_results=search_results,
            holders=feed.num_holders,
            holders_percentage_increase=feed.holders_percent_increase_24h,
            trading_volume_24h=feed.volume_24h,
            buy_sell_ratio_24h=feed.buy_sell_ratio_24h,
            volume_to_marketcap_ratio=feed.volume_marketcap_ratio,
            market_cap=feed.market_cap,
            avg_holders_distribution=feed.avg_holders_distribution,
        )

        sentiment_result = get_structured_response(
            system_message, SentimentResult, "o3-mini"
        )

        return sentiment_result

    def analyze_ticker_sentiment(
        self, input_payload: SentimentAgentInput
    ) -> SentimentAgentOutput:
        """
        Complete flow:
        1) Retrieve tweets & reddit posts,
        2) Perform sentiment analysis,
        3) Return a structured result (SentimentAgentOutput).
        """

        try:
            feed = self.gather_social_data(input_payload)

        except Exception as e:
            logging.error(f"Error gathering social data: {e}")
            feed = ModelInsert(
                ticker=input_payload.ticker,
                token_name=input_payload.token_name,
                tweets=[],
                reddit_posts=[],
                num_holders=0,
                holders_percent_increase_24h=0.0,
                volume_24h=0.0,
                buy_sell_ratio_24h=0.0,
                volume_marketcap_ratio=0.0,
                avg_holders_distribution=0.0,
                market_cap=0.0,
            )

        sentiment = self.run_sentiment_analysis(feed)

        sentiment_struct = SentimentAgentOutput(feed=feed, sentiment=sentiment)

        # self.log_sentiment(input_payload.ticker, sentiment_struct)

        return sentiment_struct


if __name__ == "__main__":
    # xclient = XClient()

    agent = SentimentAgent(XClient(), include_x=False)

    sentiment_input = SentimentAgentInput(
        ticker="BASED",
        token_name="BASEDAFSOLANA",
        max_tweets=10,
        max_reddit_posts=2,
        hours=24,
        links=["https://x.com/based__solana?lang=en-GB", "https://basedafsolana.com/"],
    )

    result: SentimentAgentOutput = agent.analyze_ticker_sentiment(sentiment_input)

    # for tw in result.feed.tweets:
    logging.info("==== REDDIT ====")
    for rp in result.feed.reddit_posts:
        logging.info(f" - {rp.concat[:100]}... {rp.created_at} {rp.upvotes}")

    logging.info("==== SENTIMENT ANALYSIS ====")
    logging.info(f" Overall: {result.sentiment.overall}")
    logging.info(f" Score: {result.sentiment.score}")
    logging.info(f" Confidence: {result.sentiment.confidence}")
    logging.info(f" Warnings: {result.sentiment.warnings}")
