import os
import json
import requests
import base64
import time
from datetime import datetime
from pathlib import Path

# Configuration for Freepik AI
FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY")
API_BASE_URL = "https://api.freepik.com/v1/ai/mystic"

OUTPUT_DIR = "./final_cleaned_dataset"
METADATA_FILE = "step_images_metadata.json"

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

class StepImageGenerator:
    def __init__(self, api_key: str, output_dir: str = OUTPUT_DIR):
        self.api_key = api_key
        self.output_dir = output_dir
        self.metadata = self._load_metadata()
        self.step_counter = len(self.metadata) + 1
        
    def _load_metadata(self):
        """Load existing metadata file if it exists"""
        metadata_path = os.path.join(self.output_dir, METADATA_FILE)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        metadata_path = os.path.join(self.output_dir, METADATA_FILE)
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_detailed_prompt(self, step_description: str) -> str:
        """Generate a detailed prompt for image generation with red highlight box"""
        prompt = f"""
Generate a realistic, instructional image showing a step in repairing/maintaining a Samsung WA5471ABP top-load washing machine.

Step: {step_description}

CRITICAL REQUIREMENTS:
1. NO TEXT - The image must NOT contain any text, labels, or written words anywhere
2. RED BOUNDING BOX - Draw a thick red rectangular box around the area where the action is being performed
3. Photography Style:
   - Show a close-up or medium shot of the washing machine or specific component
   - Include a person's hand with Indian skin tone performing the action
   - Show the exact action described in the step
   - Professional photography style, well-lit
   - Include relevant tools if needed
   - Realistic and educational appearance

VISUAL ELEMENTS:
- Red rectangle border (about 3-5 pixels thick) highlighting the action area
- Bright, clear lighting to show details
- Clean background focused on the task
- High quality, 1280x720 resolution

FORBIDDEN:
- No text labels
- No captions
- No numbers
- No arrows with text
- Only visual indicators (the red box)

Make it look like authentic instructional manual photography with a clear red highlight box around the precise action area.
        """
        return prompt.strip()
    
    def _extract_problem_name(self, step_description: str) -> str:
        """Extract a problem/action name from step description"""
        # Take the first meaningful part of the step
        words = step_description.split()[:8]
        return ' '.join(words).rstrip('.,;:')
    
    def _create_metadata_entry(self, filename: str, step_description: str, problem_name: str, detailed_prompt: str) -> dict:
        """Create a metadata entry for the generated image"""
        return {
            "id": filename,
            "file_path": f".\\final_cleaned_dataset\\{filename}",
            "problem_name": problem_name,
            "dense_caption": f"A Samsung WA5471ABP top-load washing machine instructional image showing: {step_description}",
            "detected_objects": [
                "hand",
                "washing machine",
                "Samsung WA5471ABP",
                "top-load washer",
                problem_name.lower()
            ],
            "step_description": step_description,
            "generated_at": datetime.now().isoformat(),
            "model_used": "freepik-mystic"
        }

    def _extract_image_bytes(self, response: requests.Response) -> bytes:
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("image/"):
            return response.content

        data = response.json()

        base64_candidates = [
            data.get("base64"),
            data.get("image"),
            data.get("image_base64"),
        ]

        for candidate in base64_candidates:
            if isinstance(candidate, str) and candidate.strip():
                return base64.b64decode(candidate)

        url_candidates = [
            data.get("url"),
            data.get("image_url"),
            data.get("output_url"),
        ]

        for candidate in url_candidates:
            if isinstance(candidate, str) and candidate.strip():
                image_response = requests.get(candidate, timeout=120)
                image_response.raise_for_status()
                return image_response.content

        if isinstance(data.get("data"), dict):
            nested = data["data"]
            if isinstance(nested.get("generated"), list) and nested["generated"]:
                first_generated = nested["generated"][0]
                if isinstance(first_generated, dict):
                    if isinstance(first_generated.get("base64"), str):
                        return base64.b64decode(first_generated["base64"])
                    if isinstance(first_generated.get("url"), str):
                        image_response = requests.get(first_generated["url"], timeout=120)
                        image_response.raise_for_status()
                        return image_response.content
                if isinstance(first_generated, str):
                    if first_generated.startswith("http"):
                        image_response = requests.get(first_generated, timeout=120)
                        image_response.raise_for_status()
                        return image_response.content
                    return base64.b64decode(first_generated)
            if isinstance(nested.get("base64"), str):
                return base64.b64decode(nested["base64"])
            if isinstance(nested.get("image"), str):
                return base64.b64decode(nested["image"])
            if isinstance(nested.get("url"), str):
                image_response = requests.get(nested["url"], timeout=120)
                image_response.raise_for_status()
                return image_response.content
            if isinstance(nested.get("output_url"), str):
                image_response = requests.get(nested["output_url"], timeout=120)
                image_response.raise_for_status()
                return image_response.content
            if isinstance(nested.get("images"), list) and nested["images"]:
                first_image = nested["images"][0]
                if isinstance(first_image, dict):
                    if isinstance(first_image.get("base64"), str):
                        return base64.b64decode(first_image["base64"])
                    if isinstance(first_image.get("url"), str):
                        image_response = requests.get(first_image["url"], timeout=120)
                        image_response.raise_for_status()
                        return image_response.content
                if isinstance(first_image, str):
                    if first_image.startswith("http"):
                        image_response = requests.get(first_image, timeout=120)
                        image_response.raise_for_status()
                        return image_response.content
                    return base64.b64decode(first_image)

        if isinstance(data.get("data"), list) and data["data"]:
            first = data["data"][0]
            if isinstance(first, dict):
                if isinstance(first.get("base64"), str):
                    return base64.b64decode(first["base64"])
                if isinstance(first.get("image"), str):
                    return base64.b64decode(first["image"])
                if isinstance(first.get("url"), str):
                    image_response = requests.get(first["url"], timeout=120)
                    image_response.raise_for_status()
                    return image_response.content
            if isinstance(first, str):
                if first.startswith("http"):
                    image_response = requests.get(first, timeout=120)
                    image_response.raise_for_status()
                    return image_response.content
                return base64.b64decode(first)

        raise ValueError("Unsupported Freepik response format")


    def _poll_for_image(self, task_id: str, headers: dict) -> bytes:
        endpoints = [
            f"{API_BASE_URL}/{task_id}",
            f"{API_BASE_URL}/{task_id}/result",
        ]

        for attempt in range(1, 16):
            for endpoint in endpoints:
                response = requests.get(endpoint, headers=headers, timeout=60)
                if response.status_code != 200:
                    continue

                try:
                    data = response.json()
                except ValueError:
                    data = None

                if isinstance(data, dict):
                    status = data.get("data", {}).get("status") or data.get("status")
                    if status in {"CREATED", "PENDING", "PROCESSING"}:
                        continue

                try:
                    return self._extract_image_bytes(response)
                except ValueError:
                    continue

            time.sleep(min(2 + attempt, 10))

        raise ValueError("Timed out waiting for Freepik image generation")

    def _log_unexpected_response(self, response: requests.Response) -> None:
        try:
            data = response.json()
            print("⚠️  Unexpected Freepik response JSON:")
            print(json.dumps(data, indent=2)[:2000])
        except ValueError:
            print("⚠️  Unexpected Freepik response text:")
            print((response.text or "").strip()[:2000])
    
    def generate_image(self, step_description: str) -> dict:
        """
        Generate image for a given step using Freepik AI and return metadata
        
        Args:
            step_description: Text description of the step
            
        Returns:
            Dictionary with image metadata
        """
        if not self.api_key:
            raise ValueError("FREEPIK_API_KEY environment variable not set")
        
        # Generate detailed prompt
        detailed_prompt = self._generate_detailed_prompt(step_description)
        problem_name = self._extract_problem_name(step_description)
        
        print(f"\n📸 Generating image for step: {problem_name}")
        print(f"📝 Step description: {step_description}")
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-freepik-api-key": self.api_key
        }

        payload = {
            "prompt": detailed_prompt,
            "aspect_ratio": "widescreen_16_9"
        }
        
        try:
            print("🔄 Calling Freepik AI image generation")

            response = requests.post(
                API_BASE_URL,
                json=payload,
                headers=headers,
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text[:500] if response.text else 'No response body'}...")
                return None
            
            try:
                image_data = self._extract_image_bytes(response)
            except ValueError:
                try:
                    data = response.json()
                except ValueError:
                    data = None

                task_id = None
                if isinstance(data, dict):
                    task_id = data.get("data", {}).get("task_id") or data.get("task_id")

                if task_id:
                    print(f"⏳ Job queued. Polling task: {task_id}")
                    image_data = self._poll_for_image(task_id, headers)
                else:
                    self._log_unexpected_response(response)
                    return None
            
            # Verify we got valid image data
            if not image_data or len(image_data) < 100:
                print("❌ Received invalid or empty image data")
                return None
            
            print(f"✅ Image received ({len(image_data)} bytes)")
            
            # Save image with descriptive filename
            filename = f"Samsung_WA5471ABP_Step{self.step_counter}_{problem_name.replace(' ', '_')}_01.png"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            print(f"✅ Image saved: {filename}")
            
            # Create metadata entry
            metadata_entry = self._create_metadata_entry(filename, step_description, problem_name, detailed_prompt)
            
            self.metadata.append(metadata_entry)
            self._save_metadata()
            self.step_counter += 1
            
            print(f"✅ Metadata saved")
            print(f"\n📋 Generated JSON entry:\n{json.dumps(metadata_entry, indent=2)}\n")
            
            return metadata_entry
            
        except requests.exceptions.Timeout:
            print("❌ Request timed out. Image generation took too long.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return None

def main():
    """Main function to run the image generator"""
    generator = StepImageGenerator(FREEPIK_API_KEY)
    
    print("=" * 70)
    print("✨ Samsung WA5471ABP Step Image Generator (Freepik AI)")
    print("=" * 70)
    print("API Provider: Freepik AI")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()
    if not FREEPIK_API_KEY:
        print("⚠️  WARNING: FREEPIK_API_KEY environment variable not set!")
        print("   Get your API key from: https://www.freepik.com")
        print("=" * 70)
        return
    print("=" * 70)
    print()
    
    # Interactive mode
    while True:
        print("\n" + "=" * 60)
        print("Enter a step description (or 'quit' to exit):")
        print("=" * 60)
        
        step = input("\n> ").strip()
        
        if step.lower() in ['quit', 'exit', 'q']:
            print("\n✅ Exiting. Metadata saved to:", 
                  os.path.join(OUTPUT_DIR, METADATA_FILE))
            break
        
        if not step:
            print("❌ Please enter a valid step description")
            continue
        
        result = generator.generate_image(step)
        if result:
            # Also save just the current entry
            print("\n🔍 Full metadata entry:")
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
