import requests
from typing import List
from datetime import datetime

from trader.agentic.prompts import PromptHandler
from trader.config import Config
    
class CritiqueSearch:
    def __init__(self):
        self.api_key = Config.CRITIQUE_SECRET
        self.base_url = "https://api.critiquebrowser.app/v1/search"

        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

        self.default_output_format = {
            "sentiment": {
                "score": "number",
                "analysis": "string",
            },
            "quotes": "array",
            "sources": "array"  # List of source details (e.g., URLs or summaries)
        }

    def search(self, prompt: str, image: str = None, source_blacklist: List[str] = [], output_format: bool = True):
 
        payload = {
            "prompt": prompt,
            "image": image,
            "source_blacklist": source_blacklist,
        }

        if output_format:
            payload["output_format"] = self.default_output_format

        payload = {key: value for key, value in payload.items() if value is not None}

        response = requests.post(self.base_url, headers=self.headers, json=payload)

        response.raise_for_status()

        return response.json()


    def extract_snippets(self, search_results):
        texts = []
        for result in search_results.get("results", []):
            text = result.get("text")
            if text:
                texts.append(text)
        return texts

if __name__ == "__main__":
    # api = BraveSearchAPI()
    # tav_search = TavilySearch()
    # link = LinkupSearch()
    crit = CritiqueSearch()


    query = f"Turbo toad (ticker symbol is TURBO) cryptocurrency discussions on social media within the last 24 hours from today: {datetime.now()}. The search MUST be within the last 1 day"
    simple_query = "Turbo toad (ticker symbol is TURBO) cryptocurrency discussions on social media within the last 24 hours"
    prompt = PromptHandler().get_prompt(template="opensearch", ticker = "DJI", token_name = "Doge Jones Industrial Average")
    # results = api.search("Turbo Toad cryptocurrency")
    # print(results)
    # snippets = api.extract_snippets(results)

    # for snippet in snippets:
    #     print(snippet)

    
    # search_results = tav_search.search(query, max_results=3, include_answer=False)

    # # Extract text snippets from the search results
    # snippets = tav_search.extract_snippets(search_results)
    # for snippet in snippets:
    #     print(snippet)
    
    # print(search_results)
    # qna = tav_search.get_answer("The sentiment of Turbo Toad (ticker symbol is TURBO) cryptocurrency within the past 72 hours. Search forums and discussion areas such as Twitter, Reddit, Discord. Along with your answer, return quotes as validation and return the URLs of your sources.")
    # qna = tav_search.get_answer(query)
    # print(qna)

    # results = link.search(query, depth='standard', output="sourcedAnswer")
    # pp(results)

    results = crit.search(prompt, output_format=False)
    print(results)