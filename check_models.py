import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
import logging
from models import LLMModel, initialize_models
from model_definitions import MODELS
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv('OPENAI_API_KEY')

async def test_model(model: LLMModel) -> Dict[str, Any]:
    """
    Test a single model by making a simple API call.
    Returns a dictionary with test results.
    """
    test_prompt = "Respond with just the word 'ok' if you can read this message."
    
    try:
        logger.info(f"Testing model: {model.name}")
        response = await model.async_generate(test_prompt)
        
        # Check if response contains 'ok' (case insensitive)
        is_working = 'ok' in response['response'].lower()
        
        return {
            "name": model.name,
            "model_id": model.model_id,
            "provider": model.provider,
            "status": "working" if is_working else "error",
            "response": response['response'],
            "latency_ms": response['latency'],
            "tokens_used": response['total_tokens'],
            "cost_usd": response['cost'],
            "error": None
        }
    except Exception as e:
        logger.error(f"Error testing {model.name}: {str(e)}")
        return {
            "name": model.name,
            "model_id": model.model_id,
            "provider": model.provider,
            "status": "error",
            "response": None,
            "latency_ms": None,
            "tokens_used": None,
            "cost_usd": None,
            "error": str(e)
        }

async def test_all_models() -> List[Dict[str, Any]]:
    """
    Test all models in parallel and return results.
    """
    # Initialize models
    models = initialize_models(MODELS)
    
    # Test all models concurrently
    tasks = [test_model(model) for model in models]
    results = await asyncio.gather(*tasks)
    
    return results

def generate_report(results: List[Dict[str, Any]]) -> str:
    """
    Generate a formatted report from test results.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Count statistics
    total_models = len(results)
    working_models = sum(1 for r in results if r['status'] == 'working')
    error_models = total_models - working_models
    
    # Generate report
    report = [
        f"LLM Model Health Check Report",
        f"Generated at: {timestamp}",
        f"\nSummary:",
        f"Total Models: {total_models}",
        f"Working Models: {working_models}",
        f"Error Models: {error_models}",
        f"\nDetailed Results:",
        "=" * 80
    ]
    
    # Group by provider
    providers = {}
    for result in results:
        provider = result['provider']
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(result)
    
    # Add results by provider
    for provider, provider_results in providers.items():
        report.append(f"\n{provider.upper()} MODELS:")
        report.append("-" * 40)
        
        for result in provider_results:
            status = "✅" if result['status'] == 'working' else "❌"
            report.append(f"{status} {result['name']}")
            report.append(f"   Model ID: {result['model_id']}")
            if result['status'] == 'working':
                report.append(f"   Latency: {result['latency_ms']:.2f}ms")
                report.append(f"   Tokens Used: {result['tokens_used']}")
                report.append(f"   Cost: ${result['cost_usd']:.6f}")
            else:
                report.append(f"   Error: {result['error']}")
            report.append("")
    
    return "\n".join(report)

async def main():
    """
    Main function to run the model health check.
    """
    try:
        # Test all models
        results = await test_all_models()
        
        # Generate and print report
        report = generate_report(results)
        print(report)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_health_check_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 