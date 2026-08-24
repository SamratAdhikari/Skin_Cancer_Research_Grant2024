import os
import requests
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
api_url = os.getenv("API_URL")
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

def analyze_image(image):
    """
    Send image to API for analysis
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: API response
    """
    temp_image_path = "temp_upload.jpg"
    image.save(temp_image_path)
    
    try:
        with open(temp_image_path, 'rb') as f:
            files = {"file": f}
            response = requests.post(api_url, headers=headers, files=files)
        
        if response.status_code == 200:
            result = response.json()
            return result, None
        else:
            return None, f"API returned status code {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)