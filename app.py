from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
import os
import re
import logging
import time
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Verify environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY is not set")
    raise ValueError("GOOGLE_API_KEY is not set")

# Define prompt template with adjusted boosts and clamping
template = """
Context: {context}

Historical Data: {historical_data}

Box Information: {box_info}

You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members in Denmark. Your task is to predict the daily intake (new members per day) for the box at a Customer Acquisition Cost (CAC) of 17.5 EUR. Follow these steps exactly to calculate the daily intake. Return only the final predicted daily intake as a float (e.g., 10.0). Do not return any other numerical value, such as the total retail value of the box.

**Step 1: Determine the Baseline Using Historical Data**
- Parse the Historical Data to find the actual_intake_per_day for the Box SKU specified in the Box Information.
- If the Box SKU is found in the Historical Data, set the baseline to its actual_intake_per_day multiplied by 1.1 to give historical data more weight.
- If the Box SKU is not found, use a default baseline of 10 members/day.
- Example Historical Data format:
  box_sku  actual_intake_per_day
  ALL-2410-WB-6  81.719008
  ALL-2501-WB-2  41.524402

**Step 2: Apply Adjustments Based on Box Information**
- **Retail Value Adjustment**: For every 10 EUR above 50 EUR in total retail value, add a 1% boost to intake, up to a maximum of 15%. For example, a retail value of 150 EUR (100 EUR above 50) adds a 10% boost (100 / 10 * 1%), while a value of 300 EUR adds the maximum 15% boost.
- **Premium Products (>20 EUR)**: Each premium product adds a 1.5% boost to intake, up to a maximum of 7.5%. For example, 3 premium products add a 4.5% boost (3 * 1.5%), 5 or more add a 7.5% boost.
- **Total Weight**: Weight up to 500g adds a 2% boost per 100g; weight above 500g adds a flat 8% boost. For example, 400g adds a 8% boost (4 * 2%), 600g adds a 8% boost.
- **Average Ratings**: For each of product, brand, and category ratings, add a 2% boost for every 0.1 increment above 4.0. For example, a rating of 4.2 adds 4% (0.2 * 2%). Sum the boosts from all three ratings.
- **Niche Products**: Each niche product reduces intake by 1%. For example, 1 niche product reduces intake by 1%, 2 niche products by 2%.
- **Free Gift Value and Rating**: Add 0.3% to intake for every 10 EUR of free gift value (e.g., 50 EUR adds 1.5%). Add an additional 2% if the free gift rating is above 4.0.
- **Seasonality**: If the launch month is early in the month (e.g., January, October), add a 2% boost. Otherwise, no adjustment.

**Step 3: Calculate the Total Adjustment**
- Sum the percentage boosts and reductions to get the total adjustment. For example, if boosts are +10% (retail value), +4.5% (premium products), +8% (weight), +6% (ratings), +3.5% (free gift), +2% (seasonality), and reductions are -1% (niche products), the total adjustment is 10 + 4.5 + 8 + 6 + 3.5 + 2 - 1 = 33%.
- Apply the total adjustment to the baseline: Adjusted Intake = Baseline * (1 + Total Adjustment / 100). For example, if Baseline = 81.719008 * 1.1 = 89.8909088, then 89.8909088 * (1 + 0.33) = 119.554908704.

**Step 4: Clamp the Final Value**
- Ensure the final predicted intake is between 1 and 100 members/day. If the adjusted intake is below 1, set it to 1; if above 100, set it to 100.

**Examples**:
- **Example 1**:
  - Historical Data: box_sku  actual_intake_per_day\nALL-2410-WB-6  81.719008
  - Box Information: Box SKU: ALL-2410-WB-6, Number of products: 7, Number of premium products (>20 EUR): 3, Total weight: 400g, Number of niche products: 1, Average product rating: 4.2, Average category rating: 4.1, Average brand rating: 4.0, Free gift: Value: 50 EUR, Rating: 4.5, Launch month: January, Total retail value (for reference only, do not use as prediction): 150 EUR
  - Step 1: Baseline = 81.719008 * 1.1 = 89.8909088 (from historical data for ALL-2410-WB-6).
  - Step 2:
    - Retail Value: 150 EUR (100 EUR above 50) = 10% boost (100 / 10 * 1%).
    - Premium products: 3 * 1.5% = 4.5% boost.
    - Weight: 400g = 4 * 2% = 8% boost.
    - Ratings: Product 4.2 (4%), Category 4.1 (2%), Brand 4.0 (0%) = 6% boost.
    - Niche products: 1 * 1% = 1% reduction.
    - Free gift: 50 EUR = 1.5% + 2% (rating > 4.0) = 3.5% boost.
    - Seasonality: Early-month = 2% boost.
  - Step 3: Total Adjustment = 10 + 4.5 + 8 + 6 + 3.5 + 2 - 1 = 33%.
  - Adjusted Intake = 89.8909088 * (1 + 0.33) = 119.554908704.
  - Step 4: Clamped Intake = 100 (capped at 100).
  - Output: 100.0

- **Example 2**:
  - Historical Data: box_sku  actual_intake_per_day\nALL-2501-WB-2  41.524402
  - Box Information: Box SKU: ALL-2501-WB-2, Number of products: 8, Number of premium products (>20 EUR): 5, Total weight: 600g, Number of niche products: 2, Average product rating: 4.3, Average category rating: 4.2, Average brand rating: 4.1, Free gift: Value: 65 EUR, Rating: 4.33, Launch month: March, Total retail value (for reference only, do not use as prediction): 250 EUR
  - Step 1: Baseline = 41.524402 * 1.1 = 45.6768422 (from historical data for ALL-2501-WB-2).
  - Step 2:
    - Retail Value: 250 EUR (200 EUR above 50) = 15% boost (capped at 15%).
    - Premium products: 5 * 1.5% = 7.5% boost (capped).
    - Weight: 600g = 8% boost.
    - Ratings: Product 4.3 (6%), Category 4.2 (4%), Brand 4.1 (2%) = 12% boost.
    - Niche products: 2 * 1% = 2% reduction.
    - Free gift: 65 EUR = 1.95% + 2% (rating > 4.0) = 3.95% boost.
    - Seasonality: Not early-month = 0% boost.
  - Step 3: Total Adjustment = 15 + 7.5 + 8 + 12 + 3.95 - 2 = 44.45%.
  - Adjusted Intake = 45.6768422 * (1 + 0.4445) = 65.9766698539.
  - Step 4: Clamped Intake = 65.9766698539 (within 1–100).
  - Output: 65.98

Now, calculate the daily intake for the given box using the same steps. Return only the numerical value of the predicted daily intake as a float (e.g., 10.0). Do not return the total retail value or any other number.
"""

prompt = PromptTemplate(
    input_variables=["context", "historical_data", "box_info"],
    template=template
)

def call_gemini_api(prompt_text, model_name):
    """Make a raw API call to the Gemini API with the v1 endpoint."""
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1000
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # Log the full response text if available
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_detail += f" - Response: {e.response.text}"
        raise Exception(error_detail)

def list_gemini_models():
    """List available models using the Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1/models?key={GOOGLE_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_detail += f" - Response: {e.response.text}"
        raise Exception(error_detail)

def predict_box_intake(context, historical_data, box_info):
    """Predict daily intake for a box using the Gemini API directly."""
    try:
        logger.info(f"Processing prediction request with context: {context[:100]}...")
        logger.info(f"Box info: {box_info[:100]}...")
        predictions = []
        max_retries = 3
        total_retry_time = 0
        # Prioritize gemini-1.5-pro, then fall back to others
        model_names = ["gemini-1.5-pro", "gemini-1.5-pro-001", "gemini-1.5-pro-002", "gemini-2.5-pro-preview-05-06", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-1.5-flash-002"]
        successful_model = None
        for i in range(1):  # Single run to minimize timeout risk
            logger.info(f"Sending request to Gemini API (run {i+1}/1)")
            prompt_text = prompt.format(
                context=context,
                historical_data=historical_data,
                box_info=box_info
            )
            # First pass: try each model once
            for model_name in model_names:
                if total_retry_time >= 7:  # Ensure total retry time stays under 7 seconds
                    logger.error("Total retry time would exceed 7 seconds, aborting retries")
                    raise ValueError("Retry timeout exceeded")
                try:
                    logger.info(f"Attempting to call Gemini API with model: {model_name}")
                    response = call_gemini_api(prompt_text, model_name)
                    result = response["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"Run {i+1} response: {result}")
                    match = re.search(r'\d+\.\d+', result)
                    if match:
                        intake_float = float(match.group())
                        if intake_float < 0:
                            logger.error("Negative intake value received")
                            raise ValueError("Intake cannot be negative")
                        # Enforce clamping between 1 and 100
                        intake_float = max(1.0, min(100.0, intake_float))
                        successful_model = model_name
                        predictions.append(intake_float)
                        break
                    else:
                        logger.warning(f"Invalid intake format in run {i+1}: {result}")
                        raise ValueError("Invalid intake format")
                except Exception as e:
                    logger.warning(f"Initial attempt failed with model {model_name}: {str(e)}")
                    total_retry_time += 0.5  # Approximate 0.5 seconds per attempt
                    continue
            if predictions:  # If successful, break
                break
            # Second pass: retry the first successful model or continue with others
            if not successful_model:
                for model_name in model_names:
                    for attempt in range(max_retries):
                        retry_delay = 0.5 * (2 ** attempt)  # Exponential backoff: 0.5, 1, 2 seconds
                        if total_retry_time + retry_delay > 7:
                            logger.error("Total retry time would exceed 7 seconds, aborting retries")
                            raise ValueError("Retry timeout exceeded")
                        try:
                            logger.info(f"Retrying Gemini API with model: {model_name} (attempt {attempt+1}/{max_retries})")
                            response = call_gemini_api(prompt_text, model_name)
                            result = response["candidates"][0]["content"]["parts"][0]["text"]
                            logger.info(f"Run {i+1} response: {result}")
                            match = re.search(r'\d+\.\d+', result)
                            if match:
                                intake_float = float(match.group())
                                if intake_float < 0:
                                    logger.error("Negative intake value received")
                                    raise ValueError("Intake cannot be negative")
                                # Enforce clamping between 1 and 100
                                intake_float = max(1.0, min(100.0, intake_float))
                                predictions.append(intake_float)
                                break
                            else:
                                logger.warning(f"Invalid intake format in run {i+1}: {result}")
                                raise ValueError("Invalid intake format")
                        except Exception as e:
                            logger.warning(f"Run {i+1} failed with model {model_name} (attempt {attempt+1}/{max_retries}): {str(e)}")
                            if attempt == max_retries - 1 and model_name == model_names[-1]:
                                logger.error(f"All attempts and model names failed for run {i+1}")
                                raise
                            time.sleep(retry_delay)
                            total_retry_time += retry_delay
                            continue
                        break  # Break inner loop if successful
                    if predictions:  # If successful, break
                        break
        if not predictions:
            logger.error("No valid intake values collected")
            raise ValueError("No valid intake values collected")
        avg_intake = sum(predictions) / len(predictions)
        logger.info(f"Averaged intake from 1 run: {avg_intake}")
        return avg_intake
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    """Endpoint for predicting box intake."""
    try:
        data = request.get_json()
        if not data or 'box_info' not in data or 'context' not in data:
            logger.error("Missing box_info or context in request")
            return jsonify({'error': 'Missing box_info or context'}), 400
        historical_data = data.get('historical_data', 'No historical data provided')
        box_info = data['box_info']
        context_text = data['context']
        logger.info("Received request to predict box intake")
        intake = predict_box_intake(context_text, historical_data, box_info)
        logger.info(f"Returning predicted intake: {intake}")
        return jsonify({'predicted_intake': intake})
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'})

@app.route('/list_models', methods=['GET'])
def list_models():
    """Endpoint to list available Gemini models."""
    try:
        logger.info("Received request to list Gemini models")
        models = list_gemini_models()
        logger.info(f"Available models: {models}")
        return jsonify({'status': 'success', 'models': models})
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/test_model', methods=['GET'])
def test_model():
    """Test endpoint to verify Gemini API access."""
    try:
        logger.info("Received request to test Gemini model")
        model_names = ["gemini-1.5-pro", "gemini-1.5-pro-001", "gemini-1.5-pro-002", "gemini-2.5-pro-preview-05-06", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-1.5-flash-002"]
        max_retries = 3
        total_retry_time = 0
        for model_name in model_names:
            if total_retry_time >= 7:  # Ensure total retry time stays under 7 seconds
                logger.error("Total retry time would exceed 7 seconds, aborting retries")
                raise ValueError("Retry timeout exceeded")
            try:
                logger.info(f"Attempting to call Gemini API with model: {model_name}")
                response = call_gemini_api("Test prompt to verify API access", model_name)
                result = response["candidates"][0]["content"]["parts"][0]["text"]
                logger.info(f"Test prompt successful: {result}")
                return jsonify({'status': 'success', 'response': result})
            except Exception as e:
                logger.warning(f"Initial attempt failed with model {model_name}: {str(e)}")
                total_retry_time += 0.5  # Approximate 0.5 seconds per attempt
                continue
        # If initial attempts fail, retry each model
        for model_name in model_names:
            for attempt in range(max_retries):
                retry_delay = 0.5 * (2 ** attempt)  # Exponential backoff: 0.5, 1, 2 seconds
                if total_retry_time + retry_delay > 7:
                    logger.error("Total retry time would exceed 7 seconds, aborting retries")
                    raise ValueError("Retry timeout exceeded")
                try:
                    logger.info(f"Retrying Gemini API with model: {model_name} (attempt {attempt+1}/{max_retries})")
                    response = call_gemini_api("Test prompt to verify API access", model_name)
                    result = response["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"Test prompt successful: {result}")
                    return jsonify({'status': 'success', 'response': result})
                except Exception as e:
                    logger.warning(f"Test prompt failed with model {model_name} (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt == max_retries - 1 and model_name == model_names[-1]:
                        logger.error("All attempts and model names failed")
                        raise
                    time.sleep(retry_delay)
                    total_retry_time += retry_delay
                    continue
                break  # Break inner loop if successful
    except Exception as e:
        logger.error(f"Test prompt failed: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
