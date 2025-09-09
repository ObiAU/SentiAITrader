import base64
import logging
import time
from typing import Literal

import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from trader.agentic.prompts import PromptHandler

class CategorisationResult(BaseModel):
    category: Literal["Animal-Themed", "Political", "Cult", "Utility/Layer1/Layer2"] = Field(..., description="The name of the correct subreddit.")


prompt_handler = PromptHandler()
openai_client = OpenAI()

def request_with_retry(method, url, headers=None, params=None, data=None, auth = None, max_retries=3, backoff_factor=1):
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, headers=headers, params=params, data=data, auth=auth)
            if response.status_code == 429:
                # Rate limit exceeded
                wait = backoff_factor * (2 ** attempt)
                logging.info(f"Rate limit hit. Retrying in {wait} seconds...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            time.sleep(1)  # Pause to respect rate limits
            return response
        
        except RequestException as e:
            logging.error(f"Request failed: {e}")
            if attempt < max_retries - 1:
                wait = backoff_factor * (2 ** attempt)
                logging.info(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise

def get_and_encode_logo(logo_uri):

    response = requests.get(logo_uri)
    if response.status_code == 200:
        base64_image = base64.b64encode(response.content).decode('utf-8')
        logging.info("Retrieved Base64 Encoded Image")
        logging.debug(f"Base64 image: {base64_image[:100]}...")
    else:
        logging.error(f"Failed to fetch the image. Status code: {response.status_code}")
        base64_image = None
    
    return base64_image

def get_structured_response(system_message: str, response_format: BaseModel, model: str = 'o3-mini',
                            # reasoning_effort: Literal['low', 'medium', 'high'] = 'low'
                            ):

    response = openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": "Return only the JSON as specified.",
                },
            ],
            response_format = response_format,
            # reasoning_effort = reasoning_effort

        )

    if hasattr(response.choices[0].message, 'parsed'):
        response_obj = response.choices[0].message.parsed

    else:
        # fallback 
        response_obj = response.choices[0].message.content

    return response_obj



def categorise_token(ticker: str, token_name: str, logo_URI: str, description: str, model: str = 'gpt-4o-2024-08-06'):
        
        # b64_image = get_and_encode_logo(logo_URI)

        # if not b64_image:
        #     return f"Failed to categorise"

        system_message = prompt_handler.get_prompt(
            template="categorise_tokens",
            ticker=ticker,
            token_name = token_name,
            description = description
        )

        if not logo_URI:

            response = openai_client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                ],
                response_format=CategorisationResult
            )
        
        else:

            logo_URI = str(logo_URI)

            if logo_URI.startswith('data:image'):
                image_url = logo_URI
            elif logo_URI.startswith('http'):
                image_url = logo_URI
            else:
                image_url = f"data:image/png;base64,{logo_URI}"

            response = openai_client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    { "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                                    {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                    },
                                },
                            ],
                    },
                ],
                response_format=CategorisationResult
            )

        if hasattr(response.choices[0].message, 'parsed'):
            sub_result = response.choices[0].message.parsed
        else:
            # fallback 
            content = response.choices[0].message.content
            raise ValueError(f"Could not parse cultiness response: {content}")
        
        return sub_result


if __name__ == "__main__":
    class TestStructure(BaseModel):
        name: str
        age: int
    
    response = get_structured_response(
        model = 'o3-mini',
        system_message="You are a helpful assistant. The user's name is Obi and he is 30 years old.",
        response_format=TestStructure
    )

    logging.info(f"Response: {response}")
