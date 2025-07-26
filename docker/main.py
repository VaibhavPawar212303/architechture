import json
import requests
import asyncio
import aiohttp
import concurrent.futures
from multiprocessing import Pool, cpu_count
from threading import Thread
import time
import re
from typing import Dict, Any, List, Tuple
from itertools import combinations, product
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standalone functions for multiprocessing (must be at module level)
def generate_steps_for_combination_standalone(combination: Dict[str, str]) -> List[str]:
    """Generate specific test steps based on combination parameters"""
    steps = ["Navigate to login page"]
    
    # Add device-specific setup
    if combination["device"] == "mobile":
        steps.append("Set mobile viewport and orientation")
    elif combination["device"] == "screen_reader":
        steps.append("Enable screen reader mode")
    
    # Add browser-specific setup
    if combination["browser"] != "chrome":
        steps.append(f"Launch {combination['browser']} browser")
    
    # Add input-specific steps
    input_steps = {
        "valid": ["Enter valid username", "Enter valid password"],
        "invalid": ["Enter invalid username", "Enter invalid password"],
        "empty": ["Leave username field empty", "Leave password field empty"],
        "sql_injection": ["Enter SQL injection code in username field", "Enter normal password"],
        "xss": ["Enter XSS script in username field", "Enter normal password"],
        "special_chars": ["Enter username with special characters", "Enter password with special characters"],
        "unicode": ["Enter username with unicode characters", "Enter password with unicode characters"],
        "null": ["Enter null values in both fields"]
    }
    
    steps.extend(input_steps.get(combination["input_type"], ["Enter test data"]))
    
    # Add user state specific steps
    if combination["user_state"] == "locked_account":
        steps.append("Account should be locked from previous attempts")
    elif combination["user_state"] == "disabled_account":
        steps.append("Use credentials for disabled account")
    
    steps.extend([
        "Click login button",
        f"Verify expected behavior for {combination['input_type']} input and {combination['user_state']}"
    ])
    
    return steps

def calculate_priority_standalone(combination: Dict[str, str]) -> str:
    """Calculate test case priority based on combination"""
    high_priority_inputs = ["valid", "invalid", "empty", "sql_injection"]
    critical_user_states = ["existing_user", "locked_account"]
    common_browsers = ["chrome", "firefox", "mobile_chrome"]
    
    score = 0
    if combination["input_type"] in high_priority_inputs:
        score += 3
    if combination["user_state"] in critical_user_states:
        score += 2
    if combination["browser"] in common_browsers:
        score += 1
    if combination["device"] == "desktop":
        score += 1
    
    if score >= 6:
        return "high"
    elif score >= 4:
        return "medium"
    else:
        return "low"

def create_test_case_from_combination_standalone(combination: Dict[str, str]) -> Dict[str, Any]:
    """Create a test case from a specific combination - standalone for multiprocessing"""
    input_type = combination["input_type"]
    user_state = combination["user_state"]
    browser = combination["browser"]
    device = combination["device"]
    
    # Generate test case based on combination
    title = f"Login Test - {input_type} input, {user_state}, {browser} on {device}"
    
    description = f"Verify login functionality with {input_type} input for {user_state} using {browser} browser on {device} device"
    
    steps = generate_steps_for_combination_standalone(combination)
    
    return {
        "title": title,
        "description": description,
        "steps": steps,
        "metadata": {
            "category": "combination_test",
            "input_type": input_type,
            "user_state": user_state,
            "browser": browser,
            "device": device,
            "priority": calculate_priority_standalone(combination)
        }
    }

class OptimizedTestCaseGenerator:
    def __init__(self, api_url: str = "http://localhost:4000/predict", max_workers: int = None):
        self.api_url = api_url
        self.max_workers = max_workers or min(cpu_count(), 10)  # Use all available cores but cap at 10
        
        # Define test case categories for parallel generation
        self.test_categories = {
            "authentication": ["valid_login", "invalid_credentials", "empty_fields", "case_sensitivity"],
            "security": ["sql_injection", "xss_prevention", "csrf_protection", "brute_force"],
            "ui_functionality": ["password_visibility", "remember_me", "forgot_password", "form_validation"],
            "session_management": ["session_timeout", "concurrent_sessions", "logout", "session_hijacking"],
            "accessibility": ["keyboard_navigation", "screen_reader", "high_contrast", "mobile_responsive"],
            "edge_cases": ["special_characters", "unicode", "very_long_inputs", "network_errors"]
        }
        
        # Reduced test data combinations for better performance
        self.test_combinations = {
            "input_types": ["valid", "invalid", "empty", "sql_injection", "xss"],  # Reduced from 8 to 5
            "user_states": ["existing_user", "new_user", "locked_account"],  # Reduced from 5 to 3
            "browsers": ["chrome", "firefox", "mobile_chrome"],  # Reduced from 6 to 3
            "devices": ["desktop", "mobile"]  # Reduced from 4 to 2
        }

    async def generate_test_cases_async(self, categories: List[str]) -> List[Dict[str, Any]]:
        """Generate test cases asynchronously for multiple categories"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for category in categories:
                task = self.generate_category_test_cases_async(session, category)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results and filter out exceptions
            all_test_cases = []
            for result in results:
                if isinstance(result, list):
                    all_test_cases.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error in async generation: {result}")
            
            return all_test_cases

    async def generate_category_test_cases_async(self, session: aiohttp.ClientSession, category: str) -> List[Dict[str, Any]]:
        """Generate test cases for a specific category using async HTTP"""
        prompt = self.create_category_prompt(category)
        
        try:
            async with session.post(
                self.api_url,
                json={"prompt": prompt},
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    api_response = await response.json()
                    return self.extract_test_cases_from_response(api_response, category)
                else:
                    logger.error(f"API error for category {category}: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Exception in category {category}: {e}")
            return []

    def generate_comprehensive_combinations(self) -> List[Dict[str, Any]]:
        """Generate comprehensive test case combinations using multiprocessing"""
        logger.info("Generating comprehensive test case combinations...")
        
        # Create all possible combinations
        combinations_list = []
        for input_type in self.test_combinations["input_types"]:
            for user_state in self.test_combinations["user_states"]:
                for browser in self.test_combinations["browsers"]:
                    for device in self.test_combinations["devices"]:
                        combinations_list.append({
                            "input_type": input_type,
                            "user_state": user_state,
                            "browser": browser,
                            "device": device
                        })
        
        logger.info(f"Generated {len(combinations_list)} test combinations")
        
        # Process combinations in parallel using a standalone function
        with Pool(processes=self.max_workers) as pool:
            chunk_size = max(1, len(combinations_list) // self.max_workers)
            test_cases = pool.map(create_test_case_from_combination_standalone, combinations_list, chunksize=chunk_size)
        
        return [tc for tc in test_cases if tc is not None]

    def create_test_case_from_combination(self, combination: Dict[str, str]) -> Dict[str, Any]:
        """Create a test case from a specific combination - wrapper for backward compatibility"""
        return create_test_case_from_combination_standalone(combination)

    def generate_steps_for_combination(self, combination: Dict[str, str]) -> List[str]:
        """Generate specific test steps based on combination parameters - wrapper"""
        return generate_steps_for_combination_standalone(combination)

    def calculate_priority(self, combination: Dict[str, str]) -> str:
        """Calculate test case priority based on combination - wrapper"""
        return calculate_priority_standalone(combination)

    def create_category_prompt(self, category: str) -> str:
        """Create a focused prompt for a specific test category"""
        category_prompts = {
            "authentication": "Create test cases for login authentication covering valid/invalid credentials, empty fields, and case sensitivity.",
            "security": "Create security-focused test cases for login including SQL injection, XSS prevention, CSRF protection, and brute force attacks.",
            "ui_functionality": "Create UI/UX test cases for login page including password visibility toggle, remember me functionality, forgot password, and form validation.",
            "session_management": "Create test cases for session management including timeout, concurrent sessions, logout functionality, and session security.",
            "accessibility": "Create accessibility test cases for login page including keyboard navigation, screen reader compatibility, high contrast mode, and mobile responsiveness.",
            "edge_cases": "Create edge case test cases for login including special characters, unicode input, very long inputs, and network error handling.",
            "performance": "Create performance test cases for login including load testing, stress testing, memory usage, and response time validation.",
            "integration": "Create integration test cases for login including third-party authentication, SSO, API endpoints, and database connectivity."
        }
        
        base_prompt = category_prompts.get(category, f"Create comprehensive test cases for {category} in login functionality.")
        
        return f"""{base_prompt}

Return ONLY valid JSON in this exact format:
{{
  "testCases": [
    {{
      "title": "Specific test case title",
      "description": "Clear description of what this test verifies",
      "steps": [
        "Step 1 description",
        "Step 2 description",
        "Step 3 with expected result"
      ]
    }}
  ]
}}

Generate 3-5 focused test cases for this category. Ensure JSON is properly formatted."""

    def extract_test_cases_from_response(self, api_response: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
        """Extract test cases from API response with error handling"""
        try:
            response_text = api_response.get("response", "")
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)(?:```|$)', response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_content = json_match.group(1) if '```json' in response_text else json_match.group(0)
                json_content = self.clean_json_content(json_content)
                
                parsed = json.loads(json_content)
                test_cases = parsed.get("testCases", [])
                
                # Add category metadata
                for tc in test_cases:
                    tc["metadata"] = tc.get("metadata", {})
                    tc["metadata"]["category"] = category
                    tc["metadata"]["source"] = "api_generated"
                
                return test_cases
            
        except Exception as e:
            logger.error(f"Error extracting test cases for category {category}: {e}")
        
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

    def generate_optimized_test_suite(self) -> Dict[str, Any]:
        """Generate optimized comprehensive test suite using all CPU cores"""
        start_time = time.time()
        logger.info(f"Starting optimized test generation using {self.max_workers} cores...")
        
        # Phase 1: Generate category-based test cases using async
        logger.info("Phase 1: Generating category-based test cases...")
        categories = list(self.test_categories.keys())
        
        # Split categories for parallel processing
        category_chunks = [categories[i:i+2] for i in range(0, len(categories), 2)]
        category_test_cases = []
        
        for chunk in category_chunks:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                chunk_results = loop.run_until_complete(self.generate_test_cases_async(chunk))
                category_test_cases.extend(chunk_results)
                loop.close()
            except Exception as e:
                logger.error(f"Error in async category generation: {e}")
        
        # Phase 2: Generate combination-based test cases using multiprocessing
        logger.info("Phase 2: Generating combination-based test cases...")
        combination_test_cases = self.generate_comprehensive_combinations()
        
        # Phase 3: Combine and deduplicate
        logger.info("Phase 3: Combining and organizing results...")
        all_test_cases = category_test_cases + combination_test_cases
        
        # Remove duplicates based on title similarity
        deduplicated_cases = self.deduplicate_test_cases(all_test_cases)
        
        # Organize by priority
        organized_cases = self.organize_by_priority(deduplicated_cases)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        result = {
            "metadata": {
                "generation_time_seconds": round(generation_time, 2),
                "total_test_cases": len(deduplicated_cases),
                "categories_generated": len(categories),
                "combinations_generated": len(combination_test_cases),
                "cpu_cores_used": self.max_workers,
                "high_priority_cases": len(organized_cases.get("high", [])),
                "medium_priority_cases": len(organized_cases.get("medium", [])),
                "low_priority_cases": len(organized_cases.get("low", []))
            },
            "testCases": deduplicated_cases,
            "organizedByPriority": organized_cases
        }
        
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        logger.info(f"Generated {len(deduplicated_cases)} unique test cases")
        
        return result

    def deduplicate_test_cases(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate test cases based on title similarity"""
        seen_titles = set()
        unique_cases = []
        
        for case in test_cases:
            title_key = case.get("title", "").lower().replace(" ", "").replace("-", "")
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_cases.append(case)
        
        return unique_cases

    def organize_by_priority(self, test_cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize test cases by priority level"""
        organized = {"high": [], "medium": [], "low": []}
        
        for case in test_cases:
            priority = case.get("metadata", {}).get("priority", "medium")
            organized[priority].append(case)
        
        return organized

    def save_results(self, results: Dict[str, Any], filename: str = "optimized_test_cases.json"):
        """Save results to file with performance metrics"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filename}")
        
        # Also save priority-based files
        for priority, cases in results["organizedByPriority"].items():
            priority_filename = f"{priority}_priority_test_cases.json"
            priority_data = {
                "metadata": results["metadata"],
                "testCases": cases
            }
            with open(priority_filename, 'w', encoding='utf-8') as f:
                json.dump(priority_data, f, indent=2, ensure_ascii=False)

def main():
    """Main execution function"""
    print("🚀 Multi-Core Optimized Login Test Case Generator")
    print("=" * 60)
    
    # Initialize generator with optimal core usage
    generator = OptimizedTestCaseGenerator(
        api_url="http://localhost:4000/predict",
        max_workers=min(cpu_count(), 6)  # Use up to 6 cores to be more conservative
    )
    
    # Show expected combinations count
    total_combinations = (
        len(generator.test_combinations["input_types"]) *
        len(generator.test_combinations["user_states"]) *
        len(generator.test_combinations["browsers"]) *
        len(generator.test_combinations["devices"])
    )
    print(f"📊 Expected combinations: {total_combinations}")
    print(f"💻 Using {generator.max_workers} CPU cores")
    
    try:
        # Generate comprehensive test suite
        results = generator.generate_optimized_test_suite()
        
        # Save results
        generator.save_results(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("✅ GENERATION COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Total time: {results['metadata']['generation_time_seconds']} seconds")
        print(f"🧪 Total test cases: {results['metadata']['total_test_cases']}")
        print(f"🔥 High priority: {results['metadata']['high_priority_cases']}")
        print(f"⚡ Medium priority: {results['metadata']['medium_priority_cases']}")
        print(f"📝 Low priority: {results['metadata']['low_priority_cases']}")
        print(f"💻 CPU cores used: {results['metadata']['cpu_cores_used']}")
        print("\n📁 Files created:")
        print("   - optimized_test_cases.json (complete suite)")
        print("   - high_priority_test_cases.json")
        print("   - medium_priority_test_cases.json") 
        print("   - low_priority_test_cases.json")
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        print(f"❌ Generation failed: {e}")
        
        # Fallback: Generate a smaller set without multiprocessing
        print("\n🔄 Attempting fallback generation...")
        try:
            fallback_generator = OptimizedTestCaseGenerator(api_url="http://localhost:4000/predict", max_workers=1)
            
            # Generate only category-based tests (no combinations)
            categories = list(fallback_generator.test_categories.keys())[:3]  # Only first 3 categories
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            category_test_cases = loop.run_until_complete(fallback_generator.generate_test_cases_async(categories))
            loop.close()
            
            fallback_results = {
                "metadata": {
                    "generation_time_seconds": 0,
                    "total_test_cases": len(category_test_cases),
                    "categories_generated": len(categories),
                    "combinations_generated": 0,
                    "cpu_cores_used": 1,
                    "fallback_mode": True
                },
                "testCases": category_test_cases,
                "organizedByPriority": {"high": category_test_cases, "medium": [], "low": []}
            }
            
            with open("fallback_test_cases.json", 'w', encoding='utf-8') as f:
                json.dump(fallback_results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Fallback completed: {len(category_test_cases)} test cases generated")
            print("📁 File created: fallback_test_cases.json")
            
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            print(f"❌ Fallback failed: {fallback_error}")

if __name__ == "__main__":
    main()