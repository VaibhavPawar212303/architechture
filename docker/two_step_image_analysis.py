import json
import requests
import base64
import io
import os
from PIL import Image
import asyncio
import aiohttp
import time
import re
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoStepImageAnalyzer:
    def __init__(self, api_url: str = "http://localhost:4000/predict"):
        self.api_url = api_url
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        
    def image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image file to base64 string"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in self.supported_formats:
                logger.error(f"Unsupported format: {file_ext}")
                return None
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large (to avoid token limits)
                max_size = 1024
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {img.size}")
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                logger.info(f"Successfully converted image to base64: {image_path} ({img.size})")
                return img_base64
                
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return None

    async def step1_ask_what_model_sees(self, image_base64: str, image_path: str) -> Dict[str, Any]:
        """Step 1: Ask the model to describe what it sees in the image"""
        
        prompt = """Look at this image carefully and describe exactly what you see. 

Please provide a detailed description including:

1. **UI Elements**: What buttons, input fields, text, images, icons do you see?
2. **Layout**: How are elements positioned? What's the overall structure?
3. **Text Content**: What text is visible? What do labels, buttons, headings say?
4. **Colors and Design**: What colors, fonts, styling do you observe?
5. **Interactive Elements**: What appears clickable or interactive?
6. **Page Type**: What kind of page/interface does this appear to be?

Be very specific and detailed. Describe the location of elements (top, bottom, left, right, center) and their appearance.

Format your response as plain text description, not JSON."""

        payload = {
            "prompt": prompt,
            "image": f"data:image/png;base64,{image_base64}",
            "max_tokens": 800,
            "temperature": 0.3  # Lower temperature for more consistent descriptions
        }
        
        try:
            logger.info("Step 1: Asking model what it sees in the image...")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        api_response = await response.json()
                        description = api_response.get("response", "").strip()
                        
                        result = {
                            "image_path": image_path,
                            "model_description": description,
                            "api_response_metadata": {
                                "usage": api_response.get("usage", {}),
                                "model": api_response.get("model", "unknown"),
                                "finish_reason": api_response.get("choices", [{}])[0].get("finish_reason", "unknown") if api_response.get("choices") else "unknown"
                            },
                            "step1_success": True
                        }
                        
                        logger.info(f"Step 1 completed. Model provided {len(description)} character description")
                        return result
                    else:
                        logger.error(f"Step 1 API error: {response.status}")
                        error_text = await response.text()
                        return {
                            "image_path": image_path,
                            "model_description": "",
                            "error": f"API error {response.status}: {error_text}",
                            "step1_success": False
                        }
                        
        except Exception as e:
            logger.error(f"Step 1 error: {e}")
            return {
                "image_path": image_path,
                "model_description": "",
                "error": str(e),
                "step1_success": False
            }

    async def step2_generate_test_cases_from_description(self, vision_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Generate test cases based on what the model saw"""
        
        if not vision_result.get("step1_success"):
            logger.error("Step 1 failed, cannot proceed to Step 2")
            return {**vision_result, "test_cases": [], "step2_success": False}
        
        description = vision_result["model_description"]
        
        prompt = f"""Based on this detailed description of a user interface, create comprehensive test cases.

WHAT THE MODEL SAW:
{description}

Now generate test cases that would verify the functionality, usability, and quality of this interface. Create test cases that cover:

1. **Functional Testing** - Testing interactive elements, forms, buttons, navigation
2. **Visual Testing** - Layout, alignment, colors, fonts, responsive design  
3. **Usability Testing** - User experience, accessibility, error handling
4. **Content Testing** - Text accuracy, labels, messages, information display

Return exactly 6 test cases in this JSON format:

{{
  "testCases": [
    {{
      "title": "Specific test case title based on observed elements",
      "description": "Clear description of what this test verifies",
      "steps": [
        "Step 1: Reference specific UI elements from the description",
        "Step 2: Define the action to perform", 
        "Step 3: Expected result/behavior"
      ],
      "category": "functional/visual/usability/content",
      "priority": "high/medium/low",
      "targetElements": ["list of specific UI elements mentioned in description"]
    }}
  ]
}}

Base the test cases directly on the UI elements and content described above. Be specific and reference the actual elements observed."""

        payload = {
            "prompt": prompt,
            "max_tokens": 1200,
            "temperature": 0.5
        }
        
        try:
            logger.info("Step 2: Generating test cases from model's description...")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        api_response = await response.json()
                        response_text = api_response.get("response", "").strip()
                        
                        # Extract test cases from response
                        test_cases = self.extract_test_cases_from_response(response_text)
                        
                        result = {
                            **vision_result,
                            "test_cases": test_cases,
                            "raw_test_response": response_text,
                            "step2_api_metadata": {
                                "usage": api_response.get("usage", {}),
                                "finish_reason": api_response.get("choices", [{}])[0].get("finish_reason", "unknown") if api_response.get("choices") else "unknown"
                            },
                            "step2_success": True,
                            "total_test_cases": len(test_cases)
                        }
                        
                        logger.info(f"Step 2 completed. Generated {len(test_cases)} test cases")
                        return result
                    else:
                        logger.error(f"Step 2 API error: {response.status}")
                        error_text = await response.text()
                        return {
                            **vision_result,
                            "test_cases": [],
                            "error_step2": f"API error {response.status}: {error_text}",
                            "step2_success": False
                        }
                        
        except Exception as e:
            logger.error(f"Step 2 error: {e}")
            return {
                **vision_result,
                "test_cases": [],
                "error_step2": str(e),
                "step2_success": False
            }

    def extract_test_cases_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract test cases from the model's response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(.*?)(?:```|$)', response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_content = json_match.group(1) if '```json' in response_text else json_match.group(0)
                json_content = self.clean_json_content(json_content)
                
                parsed = json.loads(json_content)
                test_cases = parsed.get("testCases", [])
                
                # Add metadata to each test case
                for tc in test_cases:
                    tc["metadata"] = tc.get("metadata", {})
                    tc["metadata"]["source"] = "two_step_analysis"
                    tc["metadata"]["generated_at"] = time.time()
                
                return test_cases
            else:
                logger.warning("No JSON found in response, attempting text parsing...")
                return self.parse_text_response(response_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return self.parse_text_response(response_text)
        except Exception as e:
            logger.error(f"Error extracting test cases: {e}")
            return []

    def clean_json_content(self, json_str: str) -> str:
        """Clean and fix JSON content"""
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', json_str)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
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
        """Fallback: Parse plain text response to extract test cases"""
        test_cases = []
        
        # Look for test case patterns
        lines = response_text.split('\n')
        current_case = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith(('Test Case', '1.', '2.', '3.', '4.', '5.', '6.')) and ':' in line:
                if current_case:
                    test_cases.append(current_case)
                current_case = {
                    "title": line.split(':', 1)[1].strip(),
                    "description": "Test case extracted from text response",
                    "steps": [],
                    "category": "general",
                    "priority": "medium",
                    "metadata": {"source": "text_parsing"}
                }
            elif line.startswith(('-', '•', '*')) and current_case:
                current_case["steps"].append(line[1:].strip())
        
        if current_case:
            test_cases.append(current_case)
        
        return test_cases

    async def analyze_image_two_step(self, image_path: str) -> Dict[str, Any]:
        """Main function: Analyze image using two-step process"""
        start_time = time.time()
        
        logger.info(f"Starting two-step analysis for: {image_path}")
        
        # Convert image to base64
        image_base64 = self.image_to_base64(image_path)
        if not image_base64:
            return {
                "image_path": image_path,
                "error": "Failed to convert image to base64",
                "step1_success": False,
                "step2_success": False,
                "processing_time": time.time() - start_time
            }
        
        # Step 1: Ask what the model sees
        vision_result = await self.step1_ask_what_model_sees(image_base64, image_path)
        
        # Step 2: Generate test cases from description
        final_result = await self.step2_generate_test_cases_from_description(vision_result)
        
        # Add timing information
        final_result["processing_time"] = time.time() - start_time
        
        logger.info(f"Two-step analysis completed in {final_result['processing_time']:.2f} seconds")
        
        return final_result

    def analyze_multiple_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple images using two-step process"""
        start_time = time.time()
        logger.info(f"Starting batch analysis of {len(image_paths)} images")
        
        all_results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.analyze_image_two_step(image_path))
                loop.close()
                
                all_results.append(result)
                
                # Add delay between images to avoid overwhelming the API
                if i < len(image_paths) - 1:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                all_results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "step1_success": False,
                    "step2_success": False
                })
        
        # Compile summary
        total_test_cases = sum(len(result.get("test_cases", [])) for result in all_results)
        successful_images = sum(1 for result in all_results if result.get("step2_success", False))
        
        summary = {
            "batch_metadata": {
                "total_images": len(image_paths),
                "successful_images": successful_images,
                "failed_images": len(image_paths) - successful_images,
                "total_test_cases": total_test_cases,
                "processing_time": time.time() - start_time
            },
            "results": all_results
        }
        
        logger.info(f"Batch processing completed: {successful_images}/{len(image_paths)} images successful")
        logger.info(f"Total test cases generated: {total_test_cases}")
        
        return summary

    def save_results(self, results: Dict[str, Any], filename: str = "two_step_analysis_results.json"):
        """Save results to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filename}")

    def print_vision_results(self, result: Dict[str, Any]):
        """Print what the model saw in a readable format"""
        print("\n" + "="*80)
        print("👁️  WHAT THE MODEL SAW")
        print("="*80)
        print(f"Image: {os.path.basename(result.get('image_path', 'unknown'))}")
        
        if result.get("step1_success"):
            print(f"\nModel Description ({len(result.get('model_description', ''))} characters):")
            print("-" * 40)
            print(result.get('model_description', 'No description'))
            print("-" * 40)
            
            # Print API metadata
            metadata = result.get('api_response_metadata', {})
            usage = metadata.get('usage', {})
            if usage:
                print(f"\nAPI Usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} tokens")
            print(f"Finish Reason: {metadata.get('finish_reason', 'unknown')}")
        else:
            print(f"\n❌ Step 1 failed: {result.get('error', 'Unknown error')}")

    def print_test_cases(self, result: Dict[str, Any]):
        """Print generated test cases in a readable format"""
        print("\n" + "="*80)
        print("🧪 GENERATED TEST CASES")
        print("="*80)
        
        if result.get("step2_success") and result.get("test_cases"):
            test_cases = result["test_cases"]
            print(f"Generated {len(test_cases)} test cases:\n")
            
            for i, tc in enumerate(test_cases, 1):
                print(f"Test Case {i}: {tc.get('title', 'Untitled')}")
                print(f"Category: {tc.get('category', 'unknown')} | Priority: {tc.get('priority', 'medium')}")
                print(f"Description: {tc.get('description', 'No description')}")
                
                steps = tc.get('steps', [])
                if steps:
                    print("Steps:")
                    for j, step in enumerate(steps, 1):
                        print(f"  {j}. {step}")
                
                target_elements = tc.get('targetElements', [])
                if target_elements:
                    print(f"Target Elements: {', '.join(target_elements)}")
                
                print("-" * 60)
        else:
            print(f"❌ No test cases generated")
            if result.get("error_step2"):
                print(f"Error: {result['error_step2']}")

def main():
    """Main execution function"""
    print("🔍 Two-Step Image Analysis and Test Case Generator")
    print("=" * 60)
    print("Step 1: Ask model what it sees in the image")
    print("Step 2: Generate test cases based on model's description")
    print("=" * 60)
    
    analyzer = TwoStepImageAnalyzer(api_url="http://localhost:4000/predict")
    
    # Get image input from user
    print("\n📸 Image Input Options:")
    print("1. Single image file")
    print("2. Multiple image files (comma-separated)")
    print("3. All images in a directory")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    image_paths = []
    
    if choice == "1":
        image_path = input("Enter image file path: ").strip()
        if os.path.exists(image_path):
            image_paths = [image_path]
        else:
            print("❌ Image file not found")
            return
            
    elif choice == "2":
        paths_input = input("Enter image file paths (comma-separated): ").strip()
        paths = [p.strip() for p in paths_input.split(",")]
        for path in paths:
            if os.path.exists(path):
                image_paths.append(path)
            else:
                print(f"⚠️ File not found: {path}")
        
        if not image_paths:
            print("❌ No valid image files found")
            return
            
    elif choice == "3":
        dir_path = input("Enter directory path: ").strip()
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext in analyzer.supported_formats:
                        image_paths.append(file_path)
            
            if not image_paths:
                print("❌ No supported image files found in directory")
                return
            else:
                print(f"📁 Found {len(image_paths)} images in directory")
        else:
            print("❌ Directory not found")
            return
    else:
        print("❌ Invalid choice")
        return
    
    # Show what will be processed
    print(f"\n🚀 Will process {len(image_paths)} image(s)")
    for i, path in enumerate(image_paths, 1):
        print(f"  {i}. {os.path.basename(path)}")
    
    proceed = input(f"\nProceed with analysis? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Operation cancelled")
        return
    
    try:
        if len(image_paths) == 1:
            # Single image analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(analyzer.analyze_image_two_step(image_paths[0]))
            loop.close()
            
            # Print results
            analyzer.print_vision_results(result)
            analyzer.print_test_cases(result)
            
            # Save results
            analyzer.save_results({"single_image_result": result}, "single_image_analysis.json")
            
        else:
            # Multiple images analysis
            results = analyzer.analyze_multiple_images(image_paths)
            
            # Print summary
            print("\n" + "="*80)
            print("📊 BATCH ANALYSIS SUMMARY")
            print("="*80)
            metadata = results["batch_metadata"]
            print(f"Images processed: {metadata['successful_images']}/{metadata['total_images']}")
            print(f"Total test cases: {metadata['total_test_cases']}")
            print(f"Processing time: {metadata['processing_time']:.2f} seconds")
            
            # Print detailed results for each image
            for result in results["results"]:
                if result.get("step1_success"):
                    analyzer.print_vision_results(result)
                if result.get("step2_success"):
                    analyzer.print_test_cases(result)
            
            # Save results
            analyzer.save_results(results, "batch_analysis_results.json")
        
        print(f"\n✅ Analysis completed! Results saved to JSON file.")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()