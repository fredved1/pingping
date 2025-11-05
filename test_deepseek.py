#!/usr/bin/env python3
import requests
import json

api_key = "sk-65da1e70c2f648418878db0db66bdfba"
base_url = "https://api.deepseek.com/v1"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a professional crypto trading analyst."},
        {"role": "user", "content": "Analyze BTC market: Price=$103000, RSI=45. Is it bullish or bearish? Reply in JSON: {\"sentiment\": \"BULLISH/BEARISH\", \"reasoning\": \"why\"}"}
    ],
    "temperature": 0.3,
    "max_tokens": 200
}

print("Testing DeepSeek API...")
print(f"URL: {base_url}/chat/completions")
print(f"Headers: {headers}")

response = requests.post(
    f"{base_url}/chat/completions",
    headers=headers,
    json=payload,
    timeout=30
)

print(f"\nStatus Code: {response.status_code}")
print(f"Response Headers: {dict(response.headers)}")
print(f"Response Text: {response.text[:500]}")

if response.status_code == 200:
    try:
        result = response.json()
        print(f"\nParsed JSON: {json.dumps(result, indent=2)}")
        content = result['choices'][0]['message']['content']
        print(f"\nAI Response Content: {content}")
    except Exception as e:
        print(f"\nError parsing response: {e}")
else:
    print(f"\nAPI Error: {response.status_code}")
    print(f"Error details: {response.text}")
