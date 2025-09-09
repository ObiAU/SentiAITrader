import requests, requests.auth, sys, os, logging
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime
from openai import OpenAI

from trader.agentic.src.reddit import RedditClient
from trader.agentic.src.subreddit_scout import SubredditScout
from trader.agentic.data_models import *
from trader.agentic.opensearch import CritiqueSearch
from trader.agentic.prompts import PromptHandler



class CommentData(BaseModel):
    """Represents a single comment parsed from Reddit JSON."""
    comment_id: str
    parent_id: str
    body: str
    score: int
    downs: int
    author: str
    depth: int
    children: List["CommentData"] = Field(default_factory=list)


class ThreadBlock(BaseModel):
    """Represents a single text block combining a top-level comment + all its children."""
    text_block: str


class CultAgent(RedditClient):
    def __init__(self,
                 openai_client: OpenAI = None,
                 prompt_handler: PromptHandler = None):
        super().__init__()
        # rclient = RedditClient()
        self.openai_client = openai_client or OpenAI()
        self.prompt_handler = prompt_handler or PromptHandler()
        self.base_url = "https://oauth.reddit.com"
        self.TABLE_NAME = "cultism"

        self.opensearch_client = CritiqueSearch()

        self.headers = {
                "Authorization": f"bearer {self.token}",
                "User-Agent": "MyCryptoScraperBot/1.0 by /u/Theredeemer08"
            }


    def run_cult_agent(self, ticker: str, token_name: str, full_text: str, subreddit: str, naive: bool = False,
                                average_hold_time: int = "", xholder_distr: int = "", holders: str = "", log: bool = False):

        system_message = self.prompt_handler.get_prompt(
            template="cult_factor",
            ticker=ticker,
            token_name = token_name,
            posts=full_text,
            subreddit = subreddit,
            average_hold_time = average_hold_time,
            num_holders = holders
        )

        if naive:
            system_message = f"Read the provided information and determine what the cultiness score was deemed as:\n {full_text}"

        response = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": (
                        "Please provide a structured JSON for cultiness. "
                        "Must conform to the CultinessResult schema: "
                        "{score, analysis, warnings}."
                    ),
                },
            ],
            response_format=CultinessResult
        )

        if hasattr(response.choices[0].message, 'parsed'):
            cultiness_result = response.choices[0].message.parsed
        else:
            # fallback 
            content = response.choices[0].message.content
            raise ValueError(f"Could not parse cultiness response: {content}")

        if log:
            # self._log_cultiness(ticker, cultiness_result) # freeze logging
            pass
        
        return cultiness_result
    

    def search_subreddit_post(self, subreddit: str, post_id: str, sort: str = "top", limit: int = 10) -> dict:

        params = {
            "sort": sort,
            "limit": limit
        }
        url = f"{self.base_url}/r/{subreddit}/comments/{post_id}"

        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()
        return resp.json()

  
    def parse_comment_tree(self, comment_listing: dict) -> List[CommentData]:
        """
        Recursively parse the comment data
        """
        parsed_comments: List[CommentData] = []

        children = comment_listing.get('data', {}).get('children', [])
        for child in children:
            kind = child.get('kind')
            if kind == 't1':
                data = child['data']
                c = CommentData(
                    comment_id=data.get('id', ''),
                    parent_id=data.get('parent_id', ''),
                    body=data.get('body', ''),
                    score=data.get('score', 0),
                    downs=data.get('downs', 0),
                    author=data.get('author', ''),
                    depth=data.get('depth', 0)
                )
                parsed_comments.append(c)

                # Recurse into replies
                replies = data.get('replies')
                if replies and isinstance(replies, dict):
                    nested_comments = self.parse_comment_tree(replies)
                    parsed_comments.extend(nested_comments)
            elif kind == 'more':
                pass
            else:
                pass

        return parsed_comments

   
    def build_comment_hierarchy(self, parsed_comments: List[CommentData], post_id: str) -> List[CommentData]:
        """
        build comment hierarchy
        """
        comment_map: Dict[str, CommentData] = {}

        for c in parsed_comments:
            c.children = []
            comment_map[c.comment_id] = c

        top_level_comments: List[CommentData] = []

        for c in parsed_comments:
            parent_id = c.parent_id 

            if parent_id == f"t3_{post_id}":
                top_level_comments.append(c)
            else:
                if parent_id.startswith("t1_"):
                    parent_comment_id = parent_id.replace("t1_", "")
                    if parent_comment_id in comment_map:
                        parent = comment_map[parent_comment_id]
                        parent.children.append(c)

        return top_level_comments


    def build_thread_block(self, comment: CommentData, indent: int = 0) -> str:
        """
        build comment tree
        """
        prefix = "  " * indent
        line = f"{prefix}{{score={comment.score}, body={comment.body}}}"

        # Recurse for child comments
        child_lines = []
        for child in comment.children:
            child_str = self.build_thread_block(child, indent=indent + 1)
            child_lines.append(child_str)

        if child_lines:
            return line + "\n" + "\n".join(child_lines)
        else:
            return line


    def create_thread_block_model(self, top_comment: CommentData) -> ThreadBlock:
        """
        Wraps string with 'COMMENT THREAD'
        """
        block_str = self.build_thread_block(top_comment)
        text_block = f"COMMENT THREAD:\n\n{block_str}\n"
        return ThreadBlock(text_block=text_block)

    def analyze_posts(self, 
                      subreddit: str,
                      query: str = "",
                      sort: str = "top",
                      post_limit: int = 3,
                      comment_limit: int = 10,
                      ticker: str = "",
                      token_name: str = "",
                      holders: str = "") -> List[CultinessResult]:
        
        search_json = self.search_subreddit(
            subreddit=subreddit,
            query=query,
            sort=sort,
            limit=post_limit
        )

        posts_data = search_json['data'].get('children', [])
        if not posts_data:
            logging.warning("No posts found.")
            return []

        results: List[CultinessResult] = []

        all_posts = []

        for idx, post in enumerate(posts_data):
            post_info = post.get('data', {})
            post_id = post_info.get('id')
            if not post_id:
                continue

            logging.info(f"[AnalyzePosts] -> Processing post #{idx+1} => ID={post_id}")

            post_title = post_info.get("title", "")
            post_score = post_info.get("score", "")
            post_body = post_info.get("selftext", "")
            truncated_body = (post_body[:100] + "...") if len(post_body) > 100 else post_body

            json_data = self.search_subreddit_post(
                subreddit=subreddit,
                post_id=post_id,
                sort=sort,
                limit=comment_limit
            )

            if len(json_data) < 2:
                logging.warning("No comments found for this post.")
                post_section = (
                    f"--- Post #{idx+1} / ID={post_id} ---\n"
                    f"Title: {post_title}\n"
                    f"Body: {truncated_body}\n\n"
                    "COMMENTS: None\n"
                )
                all_posts.append(post_section)
                continue
            
            try:
                comment_listing = json_data[1]

                parsed_comments = self.parse_comment_tree(comment_listing)

                top_level_comments = self.build_comment_hierarchy(parsed_comments, post_id)
            
            except Exception as e:
                logging.error(f"Failed to extract post comments. Error: {e}")
                return

            comments_text = []
            for top_comment in top_level_comments:
                block_model = self.create_thread_block_model(top_comment)
                comments_text.append(block_model.text_block)

            if not comments_text:
                comments_text_str = "No Comments"
            else:
                comments_text_str = "\n\n".join(comments_text)

            post_section = (
                f"--- Post #{idx+1} / ID={post_id} ---\n"
                f"Title: {post_title}\n"
                f"Body: {truncated_body}\n\n"
                f"COMMENTS:\n{comments_text_str}\n"
            )

            all_posts.append(post_section)

            logging.debug(f"Combined comments found for post {post_id}: {comments_text_str[:200]}...")

        combined_text = "\n\n".join(all_posts)

        logging.debug(f"Full subreddit text for single call: {combined_text[:500]}...")

        cultiness_result = self.run_cult_agent(
            subreddit=subreddit,
            ticker=ticker,
            token_name=token_name,
            full_text=combined_text,
            holders = holders
        )

        return cultiness_result
    
    def cult_search(self, ticker: str, token_name: str, average_hold_time: str = ""):

        search_prompt = self.prompt_handler.get_prompt(
                                                        template="cult_search",
                                                        ticker=ticker,
                                                        token_name=token_name, 
                                                        current_date = datetime.now(),
                                                        average_hold_time = average_hold_time
                                                        )
        
        search_results = self.opensearch_client.search(prompt=search_prompt,
                                                       source_blacklist=[],
                                                       output_format=False)
        
        return search_results
    
    def search_cultism(self, ticker, token_name, num_holders, hold_time = ""):

        try:
            subreddit_name = scout.find_subreddit_ai(ticker, token_name)
            post_limit = 2
            comment_limit = 5

            verdict = agent.analyze_posts(
                        subreddit=subreddit_name,
                        query=ticker,
                        sort="hot",
                        post_limit=post_limit,
                        comment_limit=comment_limit,
                        ticker=ticker,
                        token_name=token_name,
                        holders = num_holders
                    )  
            
        except Exception as e:
            logging.info(f"Resorting to backup deepsearch. Direct reddit parsing failed with error: {e}")
            search_results = self.cult_search(ticker, token_name, hold_time)

            verdict = self.run_cult_agent(ticker, token_name, full_text=search_results, naive=True)

        
        return verdict



if __name__ == "__main__":
    scout = SubredditScout()
    agent = CultAgent()

    # xrp -- unique
    # subreddit_name = "xrp"
    # ticker = token_name = "XRP"
    # num_holders = "5.7 million"

    # moodeng_addr = "ED5nyyWEzpPPiWimP8vYm7sD7TD3LAt3Q3gRTWHzPJBY"

    # addr = "63LfDmNb3MQ8mw9MtZ2To9bEA2M71kZUUGq5tiJxcqj9"
    # meta = fetch_token_overview_and_metadata(addr)

    # ticker = meta.get("ticker", "")
    # token_name = meta.get("name", "")
    # num_holders = meta.get("num_holders", "")

    ticker = "MOODENG"
    token_name = "Moo Deng"
    num_holders = "50000"


    # subreddit_name = "gigachadmemecoin"
    subreddit_name = scout.find_subreddit_ai(ticker, token_name)
    logging.info(f"Found subreddit: {subreddit_name}")

    my_post_limit = 2
    my_comment_limit = 5

    results = agent.analyze_posts(
        subreddit=subreddit_name,
        query=ticker,
        sort="hot",
        post_limit=my_post_limit,
        comment_limit=my_comment_limit,
        ticker=ticker,
        token_name=token_name,
        holders = num_holders
    )
    r = results
    # for i, r in enumerate(results, start=1):
    logging.info(f"Score: {r.score}")
    logging.info(f"Analysis: {r.analysis}")
    logging.info(f"Warnings: {r.warnings}")
