from typing import Optional, List

from pydantic import BaseModel, Field

from agentic.prompts import PromptHandler
from agentic.utils import get_structured_response
from trader.agentic.src.praw_sentiment import PrawRedditClient


class CommentContributionSettings(BaseModel):
    allowed_media_types: Optional[List[str]]


class SubredditData(BaseModel):
    active_user_count: int
    icon_img: str
    key_color: str
    name: str
    subscriber_count: int
    is_chat_post_feature_enabled: bool
    allow_chat_post_creation: bool
    allow_images: bool


class SubredditsResponse(BaseModel):
    subreddits: List[SubredditData]


class SubredditResult(BaseModel):
    name: str = Field(..., description="The name of the correct subreddit.")


class SubredditScout:
    """
    Attempts to find an appropriate subreddit given a token symbol and/or token name.
    Verifies the subreddit exists and returns the best guess.
    """

    def __init__(self):
        self.prompt_handler = PromptHandler()
        self.praw_client = PrawRedditClient()
        self.TABLE_NAME = "logs"

    # --------------------------------------------------------------------
    # AI SCOUTING
    # --------------------------------------------------------------------

    def run_ai_scout(self, ticker: str, token_name: str, concat: str):
        system_message = self.prompt_handler.get_prompt(
            template="sub_scout", ticker=ticker, token_name=token_name, sub_info=concat
        )

        sub_result = get_structured_response(system_message, SubredditResult, "o3-mini")

        return sub_result

    def find_subreddit_ai(
        self, token_symbol: str, token_name: str, limit: int = 5
    ) -> Optional[str]:
        """
        1) Search subreddits by `token_symbol` or `token_name`.
        2) Fetch their descriptions from PRAW.
        3) Run them through `run_ai_scout` to decide the best match.
        4) Return the chosen subreddit name (if any).
        """

        queries = [token_name, token_symbol, f"{token_symbol}coin"]
        found_subs: List[SubredditData] = []

        for query in queries:
            if not query:
                continue

            raw_subs = self.praw_client.find_subreddits(query=query, limit=limit)
            found_subs.extend(raw_subs)

        if not found_subs:
            logging.warning("No subreddits found.")
            return None

        unique_subs_dict = {}
        for sub in found_subs:
            if sub.name.lower() in unique_subs_dict:
                continue

            unique_subs_dict[sub.name.lower()] = sub

        sub_info_list = []
        for s in unique_subs_dict.values():
            info_block = (
                f"Subreddit: {s.name}\n"
                f"Subscriber count: {s.subscriber_count}\n"
                f"Description: {s.description}\n\n"
            )
            sub_info_list.append(info_block)

        if not sub_info_list:
            logging.warning("No valid subreddits to evaluate.")
            return None

        concatenated_info = "\n".join(sub_info_list)

        logging.debug(f"Concatenated info: {concatenated_info[:400]}")

        chosen_sub = self.run_ai_scout(
            ticker=token_symbol, token_name=token_name, concat=concatenated_info
        )
        return chosen_sub.name


if __name__ == "__main__":
    scout = SubredditScout()
    found_sub = scout.find_subreddit_ai("GIGA", "gigachad", 5)
    if found_sub:
        logging.info(f"Found subreddit: {found_sub}")
    else:
        logging.warning("No matching subreddit found.")
