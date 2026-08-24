import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("JINA_API_KEY")

def test_jina_image():
    # Take the first image
    img_path = "generated_step_images_20260824_0052/01_Starting_Power_Problems_Detailed_step001.png"
    if not os.path.exists(img_path):
        print("Image not found")
        return
        
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    # Try format 1
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Let's try {"image": img_b64}
    payload = {
        "model": "jina-embeddings-v5-omni-small",
        "input": [{"image": img_b64}]
    }
    
    res = requests.post(url, headers=headers, json=payload)
    print(f"Format {{'image': b64}}: {res.status_code}")
    if res.status_code != 200:
        print(res.text)

    # Let's try {"bytes": img_b64}
    if res.status_code != 200:
        payload["input"] = [{"bytes": img_b64}]
        res = requests.post(url, headers=headers, json=payload)
        print(f"Format {{'bytes': b64}}: {res.status_code}")
        if res.status_code != 200:
            print(res.text)
            
    if res.status_code == 200:
        print("Success! Dimensions:", len(res.json()["data"][0]["embedding"]))

test_jina_image()
