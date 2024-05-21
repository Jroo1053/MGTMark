import time

from src.lib.utils import load_auth_creds
import requests
import json

WINSTON_API_URL = "https://api.gowinston.ai/functions/v1/predict"


class APIPipeline():

    def __init__(self, provider: str, auth_creds: str):
        PROVIDER_MAP = {
            "ORIG": self.get_results_originality,
            "WINSTON": self.get_results_winston
        }
        self.provider = provider
        self.provider_func = PROVIDER_MAP[provider]
        self.auth_creds = load_auth_creds(auth_creds, provider)

    def __call__(self, text, max_length):
        if isinstance(text, list):
            text = "".join(text)[:max_length]
        results = self.provider_func(text)
        return [results]

    def get_results_winston(self, text):
        url = "https://api.gowinston.ai/functions/v1/predict"

        payload = {
            "language": "en",
            "sentences": True,
            "text": text,
            "version": "3.0"
        }
        headers = {
            "Authorization": f"Bearer {self.auth_creds}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)

        response_json = json.loads(
            response.text
        )

        if response.status_code == 200:
            if response_json["score"] < 50:
                result = {
                    "label": "Fake",
                    "score": 100 - response_json["score"]
                }
            else:
                result = {
                    "label": "Real",
                    "score": response_json["score"]
                }
            return result
        else:
            print(f"Got Error: {response.text}")
            raise ValueError

    def get_results_originality(self, text):
        url = "https://api.originality.ai/api/v1/scan/ai"

        payload = {
            "content": text,
            "aiModelVersion": "2",
            "storeScan": "false"
        }
        headers = {
            'X-OAI-API-KEY': self.auth_creds,
            'Accept': 'application/json',
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        response_json = json.loads(
            response.text
        )
        if response.status_code == 200:
            if response_json["score"]["ai"] == 0 and\
                    response_json["score"]["original"] == 0:
                print("WARNING RAN OUT OF CREDITS")
            if response_json["score"]["ai"] > 0.5:
                result = {
                    "label": "Fake",
                    "score": response_json['score']['ai']
                }
            else:
                result = {
                    "label": "real",
                    "score": response_json['score']['original']
                }
            return result
        else:
            print(f"Got Error: {response.text}")
            time.sleep(0.25)
            return {}
