import json
import requests

class Wallet:
    def __init__(self, private_key: str):
        self.private_key = private_key
        self.address = self._derive_address()

    def _derive_address(self):
        # stubbed fake address logic
        return "0xFAKEADDRESS"

    def sign_and_send(self, endpoint: str, payload: dict):
        headers = {
            "Content-Type": "application/json",
        }
        # stubbed: signature handling skipped
        response = requests.post(
            f"https://api.hyperliquid.xyz/{endpoint}",
            headers=headers,
            data=json.dumps(payload),
        )
        return response.json()

    @classmethod
    def from_private_key(cls, private_key: str):
        return cls(private_key)
