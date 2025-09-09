import requests, requests.auth, base64, datetime, time
from datetime import timedelta, datetime

from trader.config import Config


class XClient:
    def __init__(self,
            api_key = Config.TWITTER_API_KEY,     
            api_secret = Config.TWITTER_API_SECRET,
            access_token = Config.TWITTER_ACCESS_TOKEN,
            access_token_secret = Config.TWITTER_ACCESS_TOKEN_SECRET,
                 ):
        
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = access_token
        self._access_token_secret = access_token_secret
        self.base_url = 'https://api.x.com/'
        self.bearer_token = self.get_bearer_token(self._api_key, self._api_secret)

    def get_bearer_token(self, _api_key, _api_secret) -> str:
        encoded_key = requests.utils.quote(_api_key)
        encoded_secret = requests.utils.quote(_api_secret)

        logging.debug(f"Encoded Key: {encoded_key}")
        
        bearer_token_credentials = f"{encoded_key}:{encoded_secret}"
        
        base64_encoded_credentials = base64.b64encode(
            bearer_token_credentials.encode('utf-8')
        ).decode('utf-8')

        token_url = f"{self.base_url}oauth2/token"
        headers = {
            "Authorization": f"Basic {base64_encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
        }
        data = {
            "grant_type": "client_credentials"
        }
        
        response = requests.post(token_url, headers=headers, data=data)
        
        if response.status_code != 200:
            raise Exception(
                f"Error getting bearer token: "
                f"{response.status_code} {response.text}"
            )
        
        json_resp = response.json()
        return json_resp["access_token"]
    
    
    def get_recent_tweets(self, query, hours=24, max_results=10):
        base_url = f"{self.base_url}2/tweets/search/recent"
        
        # X requires RFC 3339 (ISO 8601) datetime format so add a 'Z' to UTC
        start_time_utc = datetime.now() - timedelta(hours=hours)
        start_time_str = start_time_utc.isoformat(timespec='seconds') + "Z"
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        
        params = {
            "query": query,
            "max_results": max_results,
            "sort_order": "relevancy",
            "start_time": start_time_str
        }
        
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(
                f"Request returned an error: {response.status_code} {response.text}"
            )
        
        data = response.json()
        return data

    def get_recent_tweets_with_backoff(
        self, 
        query, 
        hours=24, 
        max_results=10, 
        max_retries=5, 
        backoff_factor=2
    ):
        """
        exponential back off
        """
        base_url = f"{self.base_url}2/tweets/search/recent"
        
        start_time_utc = datetime.now() - timedelta(hours=hours)
        start_time_str = start_time_utc.isoformat(timespec='seconds') + "Z"
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        
        params = {
            "query": query,
            "max_results": max_results,
            "sort_order": "relevancy",
            "start_time": start_time_str
        }

        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.get(base_url, headers=headers, params=params)
                response.raise_for_status() 
                return response.json()
            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt < max_retries:
                    sleep_time = backoff_factor ** attempt
                    logging.warning(f"[Retry {attempt}/{max_retries}] Error: {e}. "
                          f"Sleeping for {sleep_time} seconds before retrying...")
                    time.sleep(sleep_time)
                else:
                    logging.error("Max retries reached. Reraising exception.")
                    raise e
                

    
    def get_tweet_by_id(self, tweet_id):
        """
        Fetch a single tweet by ID
        """
        endpoint_url = f"{self.base_url}2/tweets/{tweet_id}"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        
        response = requests.get(endpoint_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(
                f"Request returned an error: {response.status_code} {response.text}"
            )
        
        return response.json()
    


    def get_user_profile(self, username):
        """
        Fetch user profile details by username
        """
        endpoint_url = f"{self.base_url}2/users/by/username/{username}"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }

        response = requests.get(endpoint_url, headers=headers)
        if response.status_code != 200:
            raise Exception(
                f"Request returned an error: {response.status_code} {response.text}"
            )
        return response.json()


    def post_tweet(self, text):
        """
        Post a tweet on behalf of the authenticated user. oauth1.0
        """
        endpoint_url = f"{self.base_url}2/tweets"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        response = requests.post(endpoint_url, headers=headers, json=payload)
        if response.status_code not in (200, 201):
            raise Exception(
                f"Request returned an error: {response.status_code} {response.text}"
            )
        return response.json()


if __name__ == "__main__":
    client = XClient()
    ticker = "BTC"
    
    # tweets = client.get_recent_tweets(ticker, max_results=10)

    try:
        tweets_with_backoff = client.get_recent_tweets_with_backoff(ticker, max_results=5)
        logging.info(f"Tweets with backoff: {len(tweets_with_backoff)}")
    except Exception as e:
        logging.error(f"Failed to fetch tweets with backoff: {e}")

    # tweet_data = client.get_tweet_by_id("1234567890123456789")

    # user_profile = client.get_user_profile("elonmusk")

    # Note: You need correct permissions and tokens for this to succeed.
    # response = client.post_tweet("Hello X!")
