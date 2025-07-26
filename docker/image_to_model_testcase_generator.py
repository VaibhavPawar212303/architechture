import json
import requests
import asyncio
import aiohttp
import base64
import io
import os
from PIL import Image, ImageGrab
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import re
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
import mss
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageAnalysisTestCaseGenerator:
    def __init__(self, api_url: str = "http://localhost:4000/predict", max_workers: int = None):
        self.api_url = api_url
        self.max_workers = max_workers or min(cpu_count(), 4)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        
    def preview_directory_images(self, directory_path: str) -> None:
        """Preview images in directory before processing"""
        try:
            if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
                print(f"❌ Invalid directory: {directory_path}")
                return
            
            print(f"\n📁 Scanning directory: {directory_path}")
            print("=" * 60)
            
            image_files = []
            other_files = []
            
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                
                if not os.path.isfile(file_path):
                    continue
                
                file_ext = Path(file_path).suffix.lower()
                file_size = os.path.getsize(file_path)
                file_size_mb = file_size / (1024 * 1024)
                
                if file_ext in self.supported_formats:
                    try:
                        with Image.open(file_path) as img:
                            dimensions = f"{img.width}x{img.height}"
                    except:
                        dimensions = "unknown"
                    
                    image_files.append({
                        "name": file_name,
                        "size_mb": file_size_mb,
                        "dimensions": dimensions,
                        "format": file_ext[1:].upper()
                    })
                else:
                    other_files.append(file_name)
            
            if image_files:
                print(f"✅ Found {len(image_files)} supported images:")
                print(f"{'#':<3} {'Filename':<25} {'Size':<8} {'Dimensions':<12} {'Format':<6}")
                print("-" * 60)
                for i, img in enumerate(image_files, 1):
                    print(f"{i:<3} {img['name'][:24]:<25} {img['size_mb']:.1f}MB{'':<3} {img['dimensions']:<12} {img['format']:<6}")
            else:
                print("❌ No supported images found")
            
            if other_files:
                print(f"\n⚠️  Found {len(other_files)} unsupported files:")
                for file_name in other_files[:5]:
                    print(f"   - {file_name}")
                if len(other_files) > 5:
                    print(f"   ... and {len(other_files)-5} more")
            
            print(f"\n📊 Supported formats: {', '.join(self.supported_formats)}")
            
        except Exception as e:
            logger.error(f"Error previewing directory: {e}")
            print(f"❌ Error scanning directory: {e}")

    def load_images_from_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Load all supported images from a directory"""
        image_sources = []
        
        try:
            if not os.path.exists(directory_path):
                logger.error(f"Directory not found: {directory_path}")
                return []
            
            if not os.path.isdir(directory_path):
                logger.error(f"Path is not a directory: {directory_path}")
                return []
            
            # Get all files in directory
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                
                # Skip if not a file
                if not os.path.isfile(file_path):
                    continue
                
                # Check if supported format
                file_ext = Path(file_path).suffix.lower()
                if file_ext in self.supported_formats:
                    image_sources.append({
                        "type": "file",
                        "path": file_path,
                        "context": f"image from directory: {file_name}"
                    })
                    logger.info(f"Found image: {file_name}")
            
            logger.info(f"Found {len(image_sources)} supported images in directory")
            return image_sources
            
        except Exception as e:
            logger.error(f"Error scanning directory {directory_path}: {e}")
            return []

    def load_image_from_file(self, file_path: str) -> Optional[str]:
        """Load image from file and convert to base64"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                logger.error(f"Unsupported format: {file_ext}")
                return None
            
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                logger.info(f"Successfully loaded image: {file_path} ({img.size})")
                return img_base64
                
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None

    def capture_screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[str]:
        """Capture screenshot of screen or specific region"""
        try:
            with mss.mss() as sct:
                if region:
                    # Capture specific region (left, top, width, height)
                    monitor = {"top": region[1], "left": region[0], 
                              "width": region[2], "height": region[3]}
                else:
                    # Capture primary monitor
                    monitor = sct.monitors[1]
                
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                logger.info(f"Screenshot captured: {img.size}")
                return img_base64
                
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return None

    def capture_webpage_screenshot(self, url: str, element_selector: Optional[str] = None) -> Optional[str]:
        """Capture screenshot of a webpage or specific element"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # Wait for page to load
            time.sleep(3)
            
            if element_selector:
                # Capture specific element
                element = driver.find_element("css selector", element_selector)
                screenshot = element.screenshot_as_png
            else:
                # Capture full page
                screenshot = driver.get_screenshot_as_png()
            
            driver.quit()
            
            img_base64 = base64.b64encode(screenshot).decode('utf-8')
            logger.info(f"Webpage screenshot captured: {url}")
            return img_base64
            
        except Exception as e:
            logger.error(f"Error capturing webpage screenshot: {e}")
            if 'driver' in locals():
                driver.quit()
            return None

    def analyze_image_with_opencv(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using OpenCV to extract features"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}
            
            # Basic image properties
            height, width, channels = img.shape
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            edge_count = cv2.countNonZero(edges)
            
            # Detect contours (UI elements)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Detect rectangles (buttons, input fields)
            rectangles = []
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    rectangles.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
            
            # Calculate color distribution
            color_hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 50, 50], [0, 180, 0, 256, 0, 256])
            dominant_colors = np.unravel_index(np.argmax(color_hist), color_hist.shape)
            
            # Text detection using template matching (simple approach)
            text_regions = self.detect_text_regions(gray)
            
            analysis = {
                "dimensions": {"width": int(width), "height": int(height), "channels": int(channels)},
                "edge_density": float(edge_count / (width * height)),
                "ui_elements": {
                    "total_contours": len(contours),
                    "rectangles": rectangles[:10],  # Limit to first 10
                    "potential_buttons": len([r for r in rectangles if 50 < r["width"] < 200 and 20 < r["height"] < 60]),
                    "potential_input_fields": len([r for r in rectangles if r["width"] > 100 and 20 < r["height"] < 40])
                },
                "color_analysis": {
                    "dominant_hue": int(dominant_colors[0]),
                    "dominant_saturation": int(dominant_colors[1]),
                    "dominant_value": int(dominant_colors[2])
                },
                "text_regions": text_regions
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}

    def detect_text_regions(self, gray_image) -> List[Dict[str, int]]:
        """Simple text region detection"""
        try:
            # Use morphological operations to find text-like regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
            
            # Find contours that might be text
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter based on aspect ratio and size (typical for text)
                if 10 < w < 500 and 5 < h < 50 and w > h:
                    text_regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
            
            return text_regions[:20]  # Limit to first 20 text regions
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []

    def create_focused_image_analysis_prompt(self, image_analysis: Dict[str, Any], context: str = "login page") -> str:
        """Create a more focused prompt to avoid token limit issues"""
        
        # Extract key information
        dimensions = image_analysis.get('dimensions', {})
        ui_elements = image_analysis.get('ui_elements', {})
        
        prompt = f"""Analyze this {context} screenshot and generate 3 specific test cases.

DETECTED ELEMENTS:
- Image size: {dimensions.get('width', 'unknown')}x{dimensions.get('height', 'unknown')}
- Buttons found: {ui_elements.get('potential_buttons', 0)}
- Input fields found: {ui_elements.get('potential_input_fields', 0)}
- Total UI elements: {ui_elements.get('total_contours', 0)}

Create exactly 3 test cases in this JSON format (keep response under 400 tokens):

{{
  "testCases": [
    {{
      "title": "Test Case 1 Title",
      "description": "Brief description",
      "steps": ["Step 1", "Step 2", "Step 3"],
      "priority": "high"
    }},
    {{
      "title": "Test Case 2 Title", 
      "description": "Brief description",
      "steps": ["Step 1", "Step 2", "Step 3"],
      "priority": "medium"
    }},
    {{
      "title": "Test Case 3 Title",
      "description": "Brief description", 
      "steps": ["Step 1", "Step 2", "Step 3"],
      "priority": "medium"
    }}
  ]
}}"""

        return prompt

    def create_image_analysis_prompt(self, image_analysis: Dict[str, Any], context: str = "login page") -> str:
        """Create a comprehensive prompt based on image analysis"""
        
        prompt = f"""Analyze this {context} image and generate comprehensive test cases based on what you observe.

IMAGE ANALYSIS DATA:
- Dimensions: {image_analysis.get('dimensions', {})}
- UI Elements Found: {image_analysis.get('ui_elements', {})}
- Color Analysis: {image_analysis.get('color_analysis', {})}
- Text Regions: {len(image_analysis.get('text_regions', []))} detected

Based on this analysis, create test cases that cover:

1. **Visual Elements Testing:**
   - Test each UI element identified (buttons, input fields, etc.)
   - Verify proper positioning and alignment
   - Check color contrast and accessibility
   - Validate responsive behavior

2. **Functional Testing:**
   - Test interaction with detected buttons and input fields
   - Verify form validation and error handling
   - Check navigation and workflow completion
   - Test keyboard and mouse interactions

3. **Cross-Browser/Device Testing:**
   - Verify consistent rendering across browsers
   - Test responsive design on different screen sizes
   - Check mobile and tablet compatibility
   - Validate touch and click interactions

4. **Accessibility Testing:**
   - Test screen reader compatibility
   - Verify keyboard navigation
   - Check color contrast ratios
   - Validate ARIA labels and semantic HTML

5. **Performance Testing:**
   - Test page load times and rendering
   - Check image optimization and loading
   - Verify smooth animations and transitions
   - Test under different network conditions

Return ONLY valid JSON in this exact format:
{{
  "testCases": [
    {{
      "title": "Specific test case title based on visual analysis",
      "description": "Clear description referencing specific UI elements observed",
      "steps": [
        "Step 1 with specific element references",
        "Step 2 with expected visual outcome",
        "Step 3 with validation criteria"
      ],
      "visualElements": {{
        "targetElements": ["list of specific UI elements to test"],
        "expectedBehavior": "description of expected visual behavior",
        "accessibilityRequirements": "specific accessibility checks needed"
      }},
      "priority": "high/medium/low",
      "category": "visual/functional/accessibility/performance"
    }}
  ]
}}

Generate 5-10 test cases based on the actual visual elements detected in the image."""

        return prompt

    async def send_image_to_model(self, image_base64: str, context: str = "login page", 
                                 image_analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Send image to model and get test cases"""
        try:
            # Create focused prompt to avoid token limits
            if image_analysis:
                prompt = self.create_focused_image_analysis_prompt(image_analysis, context)
            else:
                prompt = f"""Analyze this {context} image and create exactly 3 test cases.

Focus on what you can see in the image. Generate test cases in this JSON format:

{{
  "testCases": [
    {{
      "title": "Test case title",
      "description": "What this test verifies",
      "steps": ["Step 1", "Step 2", "Step 3"],
      "priority": "high/medium/low"
    }}
  ]
}}

Keep response under 400 tokens. Generate exactly 3 test cases."""

            payload = {
                "prompt": prompt,
                "image": f"data:image/png;base64,{image_base64}",
                "max_tokens": 600,  # Increase token limit
                "temperature": 0.7
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            logger.info("Sending image to model for analysis...")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=90)  # Longer timeout
                ) as response:
                    if response.status == 200:
                        api_response = await response.json()
                        test_cases = self.extract_test_cases_from_response(api_response)
                        logger.info(f"Model generated {len(test_cases)} test cases from image")
                        return test_cases
                    else:
                        logger.error(f"API error: {response.status}")
                        error_text = await response.text()
                        logger.error(f"Error response: {error_text}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error sending image to model: {e}")
            return []

    async def send_image_multiple_focused_calls(self, image_base64: str, context: str = "login page", 
                                              image_analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Send multiple focused requests to get comprehensive test coverage"""
        
        # Define different focus areas for multiple calls
        focus_areas = [
            {
                "focus": "functional testing",
                "prompt_suffix": "Focus on testing button clicks, form submissions, and user interactions."
            },
            {
                "focus": "visual testing", 
                "prompt_suffix": "Focus on layout, alignment, colors, and visual appearance."
            },
            {
                "focus": "accessibility testing",
                "prompt_suffix": "Focus on keyboard navigation, screen readers, and accessibility features."
            }
        ]
        
        all_test_cases = []
        
        for i, focus_area in enumerate(focus_areas):
            try:
                logger.info(f"Making API call {i+1}/3 for {focus_area['focus']}")
                
                # Create focused prompt
                base_info = ""
                if image_analysis:
                    ui_elements = image_analysis.get('ui_elements', {})
                    base_info = f"Detected: {ui_elements.get('potential_buttons', 0)} buttons, {ui_elements.get('potential_input_fields', 0)} input fields. "
                
                prompt = f"""Analyze this {context} image for {focus_area['focus']}.

{base_info}{focus_area['prompt_suffix']}

Generate exactly 2 test cases in JSON format:

{{
  "testCases": [
    {{
      "title": "Test case title for {focus_area['focus']}",
      "description": "Brief description",
      "steps": ["Step 1", "Step 2", "Step 3"],
      "priority": "high/medium/low",
      "category": "{focus_area['focus'].replace(' ', '_')}"
    }},
    {{
      "title": "Second test case title", 
      "description": "Brief description",
      "steps": ["Step 1", "Step 2", "Step 3"],
      "priority": "high/medium/low",
      "category": "{focus_area['focus'].replace(' ', '_')}"
    }}
  ]
}}

Keep response under 300 tokens."""
                
                payload = {
                    "prompt": prompt,
                    "image": f"data:image/png;base64,{image_base64}",
                    "max_tokens": 400,
                    "temperature": 0.7
                }
                
                headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=90)
                    ) as response:
                        if response.status == 200:
                            api_response = await response.json()
                            test_cases = self.extract_test_cases_from_response(api_response)
                            
                            # Add focus metadata
                            for tc in test_cases:
                                tc["metadata"] = tc.get("metadata", {})
                                tc["metadata"]["focus_area"] = focus_area["focus"]
                            
                            all_test_cases.extend(test_cases)
                            logger.info(f"Generated {len(test_cases)} test cases for {focus_area['focus']}")
                        else:
                            logger.error(f"API error for {focus_area['focus']}: {response.status}")
                
                # Small delay between calls to avoid overwhelming the API
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in {focus_area['focus']} call: {e}")
                continue
        
        logger.info(f"Total test cases generated: {len(all_test_cases)}")
        return all_test_cases
        """Extract test cases from model response"""
        try:
            response_text = api_response.get("response", "")
            
            # Try to extract JSON from response
            json_match = re.search(r'```json\s*(.*?)(?:```|$)', response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_content = json_match.group(1) if '```json' in response_text else json_match.group(0)
                json_content = self.clean_json_content(json_content)
                
                parsed = json.loads(json_content)
                test_cases = parsed.get("testCases", [])
                
                # Add metadata
                for tc in test_cases:
                    tc["metadata"] = tc.get("metadata", {})
                    tc["metadata"]["source"] = "image_analysis"
                    tc["metadata"]["generated_at"] = time.time()
                
                return test_cases
            else:
                logger.warning("No JSON found in response, attempting text parsing...")
                return self.parse_text_response(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return self.parse_text_response(api_response.get("response", ""))
        except Exception as e:
            logger.error(f"Error extracting test cases: {e}")
            return []

    def clean_json_content(self, json_str: str) -> str:
        """Clean and fix JSON content"""
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', json_str)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix common issues
        cleaned = re.sub(r'nant\.\s*password', 'invalid password', cleaned)
        
        # Balance brackets and braces
        open_braces = cleaned.count('{')
        close_braces = cleaned.count('}')
        for _ in range(open_braces - close_braces):
            cleaned += '}'
        
        open_brackets = cleaned.count('[')
        close_brackets = cleaned.count(']')
        for _ in range(open_brackets - close_brackets):
            cleaned += ']'
        
        return cleaned.strip()

    def parse_text_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse plain text response to extract test cases"""
        test_cases = []
        
        # Look for test case patterns in text
        patterns = [
            r'(?:Test Case \d+:|^\d+\.)\s*(.+?)(?=Test Case \d+:|^\d+\.|$)',
            r'(?:Title:|**)\s*(.+?)(?=Description:|Steps:)',
        ]
        
        # This is a basic fallback - in practice, encourage JSON responses
        lines = response_text.split('\n')
        current_case = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith(('Test Case', '1.', '2.', '3.')) and ':' in line:
                if current_case:
                    test_cases.append(current_case)
                current_case = {
                    "title": line.split(':', 1)[1].strip(),
                    "description": "Test case extracted from text response",
                    "steps": [],
                    "metadata": {"source": "text_parsing"}
                }
            elif line.startswith(('-', '•', '*')) and current_case:
                current_case["steps"].append(line[1:].strip())
        
        if current_case:
            test_cases.append(current_case)
        
        return test_cases

    def process_multiple_images(self, image_sources: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Process multiple images and generate comprehensive test suite"""
        start_time = time.time()
        logger.info(f"Processing {len(image_sources)} images...")
        
        all_test_cases = []
        processed_images = []
        
        for i, source in enumerate(image_sources):
            logger.info(f"Processing image {i+1}/{len(image_sources)}")
            
            try:
                # Handle different source types
                if isinstance(source, str):
                    if source.startswith('http'):
                        # URL - capture webpage
                        image_base64 = self.capture_webpage_screenshot(source)
                        context = "webpage"
                        image_analysis = None
                    elif os.path.exists(source):
                        if os.path.isdir(source):
                            logger.error(f"Source is a directory, not a file: {source}")
                            continue
                        # File path
                        image_base64 = self.load_image_from_file(source)
                        context = "uploaded image"
                        image_analysis = self.analyze_image_with_opencv(source)
                    else:
                        logger.error(f"Invalid source: {source}")
                        continue
                else:
                    # Dictionary with configuration
                    source_type = source.get("type", "file")
                    source_path = source.get("path", "")
                    context = source.get("context", "login page")
                    
                    if source_type == "file":
                        if os.path.isdir(source_path):
                            logger.error(f"Source path is a directory, not a file: {source_path}")
                            continue
                        image_base64 = self.load_image_from_file(source_path)
                        image_analysis = self.analyze_image_with_opencv(source_path)
                    elif source_type == "url":
                        image_base64 = self.capture_webpage_screenshot(source_path, source.get("selector"))
                        image_analysis = None
                    elif source_type == "screenshot":
                        region = source.get("region")
                        image_base64 = self.capture_screenshot(region)
                        image_analysis = None
                    else:
                        logger.error(f"Unknown source type: {source_type}")
                        continue
                
                if not image_base64:
                    logger.error(f"Failed to load image from source: {source}")
                    continue
                
                # Send to model for analysis - use multiple focused calls for better results
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                test_cases = loop.run_until_complete(
                    self.send_image_multiple_focused_calls(image_base64, context, image_analysis)
                )
                loop.close()
                
                if test_cases:
                    # Add source information to test cases
                    for tc in test_cases:
                        tc["metadata"]["image_source"] = str(source)
                        tc["metadata"]["image_context"] = context
                        if image_analysis:
                            tc["metadata"]["image_analysis"] = image_analysis
                    
                    all_test_cases.extend(test_cases)
                    processed_images.append({
                        "source": str(source),
                        "context": context,
                        "test_cases_generated": len(test_cases),
                        "image_analysis": image_analysis
                    })
                    
                    logger.info(f"Generated {len(test_cases)} test cases from image {i+1}")
                else:
                    logger.warning(f"No test cases generated from image {i+1}")
                    
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                continue
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        result = {
            "metadata": {
                "processing_time_seconds": round(processing_time, 2),
                "total_images_processed": len(processed_images),
                "total_test_cases_generated": len(all_test_cases),
                "successful_images": len(processed_images),
                "failed_images": len(image_sources) - len(processed_images)
            },
            "testCases": all_test_cases,
            "processedImages": processed_images,
            "summary": {
                "images_by_context": {},
                "test_cases_by_priority": {"high": 0, "medium": 0, "low": 0},
                "test_cases_by_category": {}
            }
        }
        
        # Generate summary statistics
        for img in processed_images:
            context = img["context"]
            result["summary"]["images_by_context"][context] = result["summary"]["images_by_context"].get(context, 0) + 1
        
        for tc in all_test_cases:
            priority = tc.get("priority", "medium")
            result["summary"]["test_cases_by_priority"][priority] += 1
            
            category = tc.get("category", "general")
            result["summary"]["test_cases_by_category"][category] = result["summary"]["test_cases_by_category"].get(category, 0) + 1
        
        logger.info(f"Processing completed: {len(all_test_cases)} test cases from {len(processed_images)} images")
        return result

    def save_results(self, results: Dict[str, Any], filename: str = "image_analyzed_test_cases.json"):
        """Save results to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filename}")

def main():
    """Main execution function"""
    print("📸 Image-to-Model Test Case Generator")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = []
    try:
        import mss
    except ImportError:
        missing_deps.append("mss")
    
    try:
        from selenium import webdriver
    except ImportError:
        missing_deps.append("selenium")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    if missing_deps:
        print("⚠️  Missing dependencies. Install with:")
        print(f"   pip install {' '.join(missing_deps)}")
        print("   (Some features may be limited)")
    
    # Initialize generator
    generator = ImageAnalysisTestCaseGenerator(api_url="http://localhost:4000/predict")
    
    # Example usage with different image sources
    image_sources = [
        # File paths
        # "login_page_screenshot.png",
        # "mobile_login.jpg",
        
        # URLs (requires selenium)
        # "https://example.com/login",
        
        # Configured sources
        {
            "type": "screenshot",
            "context": "current screen",
            "region": None  # Full screen, or specify (x, y, width, height)
        },
        
        # Add your own image sources here
        # {
        #     "type": "file",
        #     "path": "path/to/your/login/screenshot.png",
        #     "context": "login page"
        # },
        # {
        #     "type": "url", 
        #     "path": "http://localhost:3000/login",
        #     "context": "local login page",
        #     "selector": ".login-form"  # Optional: capture specific element
        # }
    ]
    
    print("📋 Configuration:")
    print(f"   API URL: {generator.api_url}")
    print(f"   Image sources: {len(image_sources)}")
    print(f"   Supported formats: {', '.join(generator.supported_formats)}")
    
    # Ask user for image sources
    print("\n🖼️  Image Source Options:")
    print("1. Take screenshot of current screen")
    print("2. Load single image from file")
    print("3. Load all images from directory")
    print("4. Capture webpage")
    print("5. Use example configuration")
    
    choice = input("\nEnter choice (1-5) or press Enter for example: ").strip()
    
    if choice == "1":
        image_sources = [{"type": "screenshot", "context": "current screen"}]
    elif choice == "2":
        file_path = input("Enter image file path: ").strip()
        if file_path and os.path.exists(file_path):
            if os.path.isdir(file_path):
                print("❌ Error: You provided a directory path. Use option 3 for directories.")
                return
            image_sources = [{"type": "file", "path": file_path, "context": "uploaded image"}]
        else:
            print("❌ Error: File not found or path not provided")
            return
    elif choice == "3":
        dir_path = input("Enter directory path containing images: ").strip()
        if dir_path and os.path.exists(dir_path):
            if not os.path.isdir(dir_path):
                print("❌ Error: Path is not a directory. Use option 2 for single files.")
                return
            
            # Preview images in directory
            generator.preview_directory_images(dir_path)
            
            # Load all images from directory
            directory_images = generator.load_images_from_directory(dir_path)
            if not directory_images:
                print("❌ Error: No supported images found in directory")
                return
            
            # Ask for confirmation
            proceed = input(f"\nProcess all {len(directory_images)} images? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Operation cancelled")
                return
            
            image_sources = directory_images
        else:
            print("❌ Error: Directory not found or path not provided")
            return
    elif choice == "4":
        url = input("Enter webpage URL: ").strip()
        if url:
            image_sources = [{"type": "url", "path": url, "context": "webpage"}]
        else:
            print("❌ Error: URL not provided")
            return
    # else: use default example configuration
    
    try:
        print(f"\n🚀 Processing {len(image_sources)} image source(s)...")
        results = generator.process_multiple_images(image_sources)
        
        # Save results
        generator.save_results(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("✅ IMAGE ANALYSIS COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Processing time: {results['metadata']['processing_time_seconds']} seconds")
        print(f"🖼️  Images processed: {results['metadata']['successful_images']}/{results['metadata']['total_images_processed']}")
        print(f"🧪 Test cases generated: {results['metadata']['total_test_cases_generated']}")
        
        print(f"\n📊 Test Cases by Priority:")
        for priority, count in results['summary']['test_cases_by_priority'].items():
            print(f"   {priority.title()}: {count}")
        
        print(f"\n📋 Test Cases by Category:")
        for category, count in results['summary']['test_cases_by_category'].items():
            print(f"   {category.title()}: {count}")
        
        print(f"\n📁 Results saved to: image_analyzed_test_cases.json")
        
        if results['metadata']['failed_images'] > 0:
            print(f"⚠️  {results['metadata']['failed_images']} image(s) failed to process")
            
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"❌ Processing failed: {e}")

if __name__ == "__main__":
    main()