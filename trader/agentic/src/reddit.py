import requests, requests.auth, os, base64, datetime, time, json, logging
from datetime import timedelta, datetime
from pydantic import BaseModel
from typing import List, Optional

from trader.agentic.utils import *
from trader.config import Config


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

class RedditClient:
    def __init__(
        self,
        client_id: str = Config.REDDIT_CLIENT_ID,
        client_secret: str = Config.REDDIT_CLIENT_SECRET,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://oauth.reddit.com"

        self.observatories = ["CryptoCurrency", "memecoins"]
        self.token = self.get_bearer_token()

        self.headers = {
                "Authorization": f"bearer {self.token}",
                "User-Agent": "MyCryptoScraperBot/1.0 by /u/Theredeemer08"
            }

    def get_bearer_token(self) -> str:
        client_auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)

        params = {
            "grant_type": "client_credentials"
        }

        response = request_with_retry(
            "post",
            "https://www.reddit.com/api/v1/access_token",
            auth=client_auth,
            params=params,
        )

        time.sleep(2) # for rate limits 60 reqs p/min

        response.raise_for_status()
        token_json = response.json()

        return token_json["access_token"]

    def search(self, query: str, sort: str = "new", limit: int = 10) -> dict:
        """
        Perform a site-wide search using the given query.
        """

        params = {
            "q": query,
            "sort": sort,
            "limit": limit,
        }
        url = "https://oauth.reddit.com/search"

        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()

        results_json = resp.json()
        outcome = self.prettify_outputs(results_json)        
        return outcome

    def prettify_outputs(self, reddit_json):
        posts_data = []
        
        children = reddit_json['data'].get('children', [])
        
        for item in children:
            post = item['data']
            
            created_utc = post.get('created_utc')
            date_str = None
            if created_utc is not None:
                date_str = datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S')
            
            upvotes = post.get('ups', 0)
            
            views = post.get('view_count', None)
            
            title = post.get('title', '')
            selftext = post.get('selftext', '')
            title_plus_text = f"Title: {title}\n Text: {selftext}".strip()
            
            posts_data.append({
                'date': date_str,
                'upvotes': upvotes,
                'views': views,
                'title_plus_text': title_plus_text
            })
            
        return posts_data

    def get_subreddit_hot(self, subreddit: str, limit: int = 10) -> dict:
        """
        Get the 'hot' posts from a specific subreddit.
        """
        params = {
            "limit": limit
        }
        url = f"https://oauth.reddit.com/r/{subreddit}/hot"

        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()
        results_json = resp.json()

        return self.prettify_outputs(results_json)


    def get_subreddit_top(self, subreddit: str, time_filter: str = "day", limit: int = 10) -> dict:
        """
        Get the 'top' posts from a specific subreddit.
        time_filter can be one of: hour, day, week, month, year, all
        """

        params = {
            "limit": limit,
            "t": time_filter
        }
        url = f"https://oauth.reddit.com/r/{subreddit}/top"

        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()
        results_json = resp.json()

        return self.prettify_outputs(results_json)



    def get_subreddit_new(self, subreddit: str, limit: int = 10) -> dict:
        """
        Get the 'new' posts from a specific subreddit.
        """
        params = {
            "limit": limit
        }
        url = f"https://oauth.reddit.com/r/{subreddit}/new"

        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()
        results_json = resp.json()

        return self.prettify_outputs(results_json)
    
    def subreddit_sentinel(self, query):

        concat = []

        for sub in self.observatories:
            elem = self.search_subreddit(sub, query, "new", 2)

            try:
                lim_content = ""
                children = elem["data"].get("children", [])
                # date = elem["data"].get("approved_at_utc", "")
                for child in children:
                    content = child["data"].get("selftext", "")
                    lim_content += f"\n{content[:100]}"


            except Exception as e:
                logging.error(f"Error observing default subreddits: {e}")
                lim_content = json.dumps(elem, indent=2) 

            concat.append(lim_content)


        return "\n".join(concat)



    def search_subreddit(self, subreddit: str, query: str, sort: str = "hot",
                        #   search_type: str = None,
                        limit: int = 10) -> dict:
        """
        Perform a search specific to a given subreddit using the provided query.
        """

        params = {
            "q": query,
            "sort": sort,
            "limit": limit,
            "restrict_sr": "on",  # Ensures that search is restricted to the subreddit
        }
        
        url = f"https://oauth.reddit.com/r/{subreddit}/search"

        resp = request_with_retry("get", url, headers=self.headers, params=params)
        time.sleep(1)
        resp.raise_for_status()
        results_json = resp.json()

        return results_json
    
    def sentiment_search_subreddits(self, query: str, exact: bool = False, include_over_18: bool = True, include_unadvertisable: bool = True) -> List[SubredditData]:
        """
        Returns a list of SubredditData objects.
        """
        url = f"{self.base_url}/api/search_subreddits"
        data = {
            "query": query,
            "exact": exact,
            "include_over_18": include_over_18,
            "include_unadvertisable": include_unadvertisable,
        }

        resp = request_with_retry(method="post", url=url, headers=self.headers, data=data)

        time.sleep(1)

        resp.raise_for_status()


        search_response = self.get_subreddit_list(resp.json())
        return search_response
    
    def get_subreddit_list(self, data) -> SubredditsResponse:

        sub_list = data.get("subreddits", [])
        results = []

        for sub_info in sub_list:
            results.append(SubredditData(
                name=sub_info.get("name", ""),
                active_user_count=sub_info.get("active_user_count", 0),
                icon_img=sub_info.get("icon_img", ""),
                key_color=sub_info.get("key_color", ""),
                subscriber_count=sub_info.get("subscriber_count", 0),
                is_chat_post_feature_enabled=sub_info.get("is_chat_post_feature_enabled", False),
                allow_chat_post_creation=sub_info.get("allow_chat_post_creation", False),
                allow_images=sub_info.get("allow_images", False),
            ))
            logging.debug(f"Individual results: {sub_info}")

        return results
    

    def capture_subreddit_post_metadata(self, subreddit: str, post_id: str, sort: str = "top", limit: int = 10) -> dict:
   
        params = {
                "sort": sort,
                "limit": limit
        }
        url = f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}"

        resp = requests.get(url, headers=self.headers, params=params)
        time.sleep(1)
        resp.raise_for_status()
        results_json = resp.json()

        return results_json


if __name__ == "__main__":
    rclient = RedditClient()
    ticker = "BTC"
    # reddits = rclient.search("Bitcoin")

    subreddit_search_results = rclient.search_subreddit("memecoins", "Doge Jones Industrial Average - DJI", "new", 2)
    logging.info(f"Subreddit search results (r/CryptoCurrency): {len(subreddit_search_results)}")

    # hot_posts = rclient.get_subreddit_hot("CryptoCurrency")

    # top_posts = rclient.get_subreddit_top("CryptoCurrency", time_filter="week")

    # new_posts = rclient.get_subreddit_new("CryptoCurrency")
