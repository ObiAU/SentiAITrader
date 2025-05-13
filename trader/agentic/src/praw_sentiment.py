import os, sys, time, praw
from typing import Optional, List, Dict
from praw.models import Subreddit, Submission, Comment
from pydantic import BaseModel, Field

from trader.agentic.utils import get_structured_response
from trader.agentic.prompts import PromptHandler
from trader.config import Config
# ---------------------------------------------------------------------------
# DATA MODELS (Pydantic) to store or serialize as needed
# ---------------------------------------------------------------------------

class CommentData(BaseModel):
    """
    Example model for representing a single comment in a thread.
    """
    comment_id: str
    parent_id: str
    body: str
    score: int
    downs: int
    author: str
    depth: int
    children: List["CommentData"] = Field(default_factory=list, repr=False)

CommentData.model_rebuild()


class ThreadBlock(BaseModel):
    """
    Example for how you might store a formatted text block of comments.
    """
    text_block: str


class SubredditData(BaseModel):
    """
    Minimal representation of a subreddit’s attributes.
    """
    name: str
    subscriber_count: int = 0
    active_user_count: int = 0
    icon_img: str = ""
    key_color: str = ""
    is_chat_post_feature_enabled: bool = False
    allow_chat_post_creation: bool = False
    allow_images: bool = True
    description: str = None


# ---------------------------------------------------------------------------
# PRAW-BASED CLIENT
# ---------------------------------------------------------------------------

class PrawRedditClient:

    def __init__(self,
                 client_id: Optional[str] = Config.REDDIT_CLIENT_ID,
                 client_secret: Optional[str] = Config.REDDIT_CLIENT_SECRET,
                 user_agent: Optional[str] = Config.REDDIT_USER_AGENT):
        _client_id = client_id
        _client_secret = client_secret
        _user_agent = user_agent

        self.reddit = praw.Reddit(
            client_id=_client_id,
            client_secret=_client_secret,
            user_agent=_user_agent,
        )

        self.observatories = ["CryptoCurrency", "memecoins", "Memecoinhub"]


    # -----------------------------------------------------------------------
    # SUBREDDIT SEARCH / LOOKUP
    # -----------------------------------------------------------------------

    def find_subreddits(self, query: str, limit: int = 2) -> List[SubredditData]:
        """
        Use PRAW to search subreddits by name.
        Returns a list of SubredditData objects (partial attributes).
        """
        results = []
        for sub in self.reddit.subreddits.search(query, limit=limit):
            try:
                results.append(SubredditData(
                    name=sub.display_name,
                    subscriber_count=sub.subscribers,
                    active_user_count=sub.active_user_count or 0,
                    icon_img=sub.icon_img or "",
                    key_color=sub.key_color or "",
                    description = getattr(sub, "public_description", ""),
                    is_chat_post_feature_enabled=getattr(sub, "is_chat_post_feature_enabled", False),
                    allow_chat_post_creation=getattr(sub, "allow_chat_post_creation", False),
                    allow_images=sub.allow_images
                ))
            except Exception as e:
                print(f"Skipping a subreddit due to error: {e}")
        return results
    

    def get_subreddit(self, subreddit_name: str) -> Optional[Subreddit]:
        """
        Return a Subreddit object if it exists. If it doesn't exist, this may raise an exception.
        """
        try:
            sub = self.reddit.subreddit(subreddit_name)
            _ = sub.created_utc
            return sub
        
        except Exception as e:
            print(f"Error retrieving subreddit {subreddit_name}: {e}")
            return None
        

    def verify_subreddit_exists(self, subreddit_name: str) -> bool:
        """
        Basic method to check if a subreddit actually exists (not banned or private).
        """
        sub = self.get_subreddit(subreddit_name)
        return sub is not None

    # -----------------------------------------------------------------------
    # (HOT, NEW, TOP)
    # -----------------------------------------------------------------------
    def get_subreddit_hot(self, subreddit_name: str, limit: int = 10):
        posts = []
        sub = self.get_subreddit(subreddit_name)
        if not sub:
            return posts

        for submission in sub.hot(limit=limit):
            posts.append(self._parse_submission(submission))
        return posts

    def get_subreddit_new(self, subreddit_name: str, limit: int = 10):
        posts = []
        sub = self.get_subreddit(subreddit_name)
        if not sub:
            return posts

        for submission in sub.new(limit=limit):
            posts.append(self._parse_submission(submission))
        return posts

    def get_subreddit_top(self, subreddit_name: str, time_filter: str = "day", limit: int = 10):
        posts = []
        sub = self.get_subreddit(subreddit_name)
        if not sub:
            return posts

        for submission in sub.top(time_filter=time_filter, limit=limit):
            posts.append(self._parse_submission(submission))
        return posts

    # -----------------------------------------------------------------------
    # SEARCH FOR POSTS IN A SUBREDDIT
    # -----------------------------------------------------------------------
    def search_subreddit_posts(self, subreddit_name: str, query: str, sort: str = "new", limit: int = 50):
        """
        The 'sort' can be 'new', 'hot', 'top', 'relevance', etc.
        """
        posts = []
        sub = self.get_subreddit(subreddit_name)
        if not sub:
            return posts

        results = sub.search(query=query, sort=sort, limit=limit)
        for submission in results:
            posts.append(self._parse_submission(submission))
        return posts

    # -----------------------------------------------------------------------
    # SUBMISSION & COMMENT PARSING
    # -----------------------------------------------------------------------
    def get_submission_comments(self, subreddit_name: str, post_id: str, limit: Optional[int] = None) -> List[CommentData]:
        """
        Fetch comments from a particular submission (post).
        Replace MoreComments so we can get the full tree if limit=None or a big number.
        Return as a list of parsed CommentData.
        """
        sub = self.get_subreddit(subreddit_name)
        if not sub:
            return []

        submission = self.reddit.submission(id=post_id)  
        submission.comments.replace_more(limit=0)
        comment_forest = submission.comments.list()

        parsed_comments = []
        for c in comment_forest:
            if isinstance(c, Comment):
                parsed_comments.append(CommentData(
                    comment_id=c.id,
                    parent_id=c.parent_id,
                    body=c.body,
                    score=c.score,
                    downs=c.downs,
                    author=str(c.author) if c.author else "[deleted]",
                    depth=c.depth
                ))
        return parsed_comments

    def parse_comment_tree(self, submission: Submission) -> List[CommentData]:
        submission.comments.replace_more(limit=0)
        return self._walk_comment_forest(submission.comments)

    def build_comment_hierarchy(self, parsed_comments: List[CommentData], post_id: str) -> List[CommentData]:
        """
        Convert a flat list of CommentData into a hierarchy.
        """
        comment_map: Dict[str, CommentData] = {}
        for c in parsed_comments:
            c.children = []
            comment_map[c.comment_id] = c

        submission_prefix = f"t3_{post_id}"
        top_level_comments: List[CommentData] = []

        for c in parsed_comments:
            if c.parent_id == submission_prefix:
                top_level_comments.append(c)
            else:
                if c.parent_id.startswith("t1_"):
                    parent_comment_id = c.parent_id.replace("t1_", "")
                    if parent_comment_id in comment_map:
                        parent = comment_map[parent_comment_id]
                        parent.children.append(c)

        return top_level_comments

    def build_thread_block(self, comment: CommentData, indent: int = 0) -> str:
        prefix = "  " * indent
        line = f"{prefix}{{score={comment.score}, body='{comment.body}'}}"

        child_lines = []
        for child in comment.children:
            child_lines.append(self.build_thread_block(child, indent + 1))

        if child_lines:
            return line + "\n" + "\n".join(child_lines)
        else:
            return line

    def create_thread_block_model(self, top_comment: CommentData) -> ThreadBlock:

        block_str = self.build_thread_block(top_comment)
        text_block = f"COMMENT THREAD:\n\n{block_str}\n"
        return ThreadBlock(text_block=text_block)

    # -----------------------------------------------------------------------
    # INTERNALLY USED HELPERS
    # -----------------------------------------------------------------------
    def _parse_submission(self, submission: Submission) -> dict:
        """
        Convert a PRAW Submission object into a dict (or a pydantic model).
        """
        return {
            "id": submission.id,
            "title": submission.title,
            "score": submission.score,
            "created_utc": submission.created_utc,
            "author": str(submission.author) if submission.author else "[deleted]",
            "num_comments": submission.num_comments,
            "selftext": submission.selftext,
            "url": submission.url
        }


# ---------------------------------------------------------------------------
# DEMO USAGE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rclient = PrawRedditClient()
    subs = rclient.find_subreddits(query="turbo toad", limit=1)
    print("Search subreddits result:", subs)

    # if rclient.verify_subreddit_exists("memecoins"):
    #     print("r/memecoins exists!")
    # else:
    #     print("r/memecoins does not exist or is private/banned.")

    # hot_posts = rclient.get_subreddit_hot("memecoins", limit=5)
    # print("HOT posts from /r/memecoins:\n", hot_posts)

    # search_posts = rclient.search_subreddit_posts("memecoins", query="Doge Jones Industrial Average - DJI", limit=5)
    # print("Search posts in /r/memecoins:\n", search_posts)

    # res = rclient.get_submission_comments("turbotoadx", "1hrvyhj", 1)
    # print(res)
