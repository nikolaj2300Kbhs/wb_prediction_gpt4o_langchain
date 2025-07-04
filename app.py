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

# Define prompt template for pattern-based approach
template = """
Context: {context}

Box Information: {box_info}

You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members in Denmark. Your task is to predict the daily intake (new members per day) for the box at a Customer Acquisition Cost (CAC) of 17.5 EUR. A regression model has been trained on historical data to predict the daily intake based on the Box Information. The predicted intake from the regression model is {predicted_intake}. Use this predicted intake as the starting point and apply any additional adjustments based on your expertise, then return the final predicted daily intake as a whole number.

**Step 1: Start with the Predicted Intake**
- The regression model predicts a daily intake of {predicted_intake} based on the Box Information.

**Step 2: Apply Additional Adjustments (Optional)**
- Based on your expertise, apply any additional adjustments to the predicted intake if necessary (e.g., market trends, seasonality not captured by the model).
- If no adjustments are needed, use the predicted intake as the final value.

**Step 3: Clamp the Final Value**
- Ensure the final predicted intake is between 1 and 90 members/day. If the adjusted intake is below 1, set it to 1; if above 90, set it to 90.

**Step 4: Round to Whole Numbers**
- Since daily intake represents the number of new members per day, round the clamped intake to the nearest whole number.

Return only the numerical value of the predicted daily intake as a whole number (e.g., 10). Do not return any other number.
"""

prompt = PromptTemplate(
    input_variables=["context", "box_info", "predicted_intake"],
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

def predict_box_intake(context, box_info, predicted_intake):
    """Predict daily intake for a box using the Gemini API directly."""
    try:
        logger.info(f"Processing prediction request with context: {context[:100]}...")
        logger.info(f"Box info: {box_info[:100]}...")
        predictions = []
        max_retries = 3
        total_retry_time = 0
        model_names = ["gemini-1.5-pro", "gemini-1.5-pro-001", "gemini-1.5-pro-002", "gemini-2.5-pro-preview-05-06", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-1.5-flash-002"]
        successful_model = None
        for i in range(1):  # Single run to minimize timeout risk
            logger.info(f"Sending request to Gemini API (run {i+1}/1)")
            prompt_text = prompt.format(
                context=context,
                box_info=box_info,
                predicted_intake=predicted_intake
            )
            for model_name in model_names:
                if total_retry_time >= 7:
                    logger.error("Total retry time would exceed 7 seconds, aborting retries")
                    raise ValueError("Retry timeout exceeded")
                try:
                    logger.info(f"Attempting to call Gemini API with model: {model_name}")
                    response = call_gemini_api(prompt_text, model_name)
                    result = response["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"Run {i+1} response: {result}")
                    match = re.search(r'\d+', result)
                    if match:
                        intake_float = float(match.group())
                        if intake_float < 0:
                            logger.error("Negative intake value received")
                            raise ValueError("Intake cannot be negative")
                        intake_float = max(1.0, min(90.0, intake_float))
                        intake_float = round(intake_float)
                        successful_model = model_name
                        predictions.append(intake_float)
                        break
                    else:
                        logger.warning(f"Invalid intake format in run {i+1}: {result}")
                        raise ValueError("Invalid intake format")
                except Exception as e:
                    logger.warning(f"Initial attempt failed with model {model_name}: {str(e)}")
                    total_retry_time += 0.5
                    continue
            if predictions:
                break
            if not successful_model:
                for model_name in model_names:
                    for attempt in range(max_retries):
                        retry_delay = 0.5 * (2 ** attempt)
                        if total_retry_time + retry_delay > 7:
                            logger.error("Total retry time would exceed 7 seconds, aborting retries")
                            raise ValueError("Retry timeout exceeded")
                        try:
                            logger.info(f"Retrying Gemini API with model: {model_name} (attempt {attempt+1}/{max_retries})")
                            response = call_gemini_api(prompt_text, model_name)
                            result = response["candidates"][0]["content"]["parts"][0]["text"]
                            logger.info(f"Run {i+1} response: {result}")
                            match = re.search(r'\d+', result)
                            if match:
                                intake_float = float(match.group())
                                if intake_float < 0:
                                    logger.error("Negative intake value received")
                                    raise ValueError("Intake cannot be negative")
                                intake_float = max(1.0, min(90.0, intake_float))
                                intake_float = round(intake_float)
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
                        break
                    if predictions:
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
        if not data or 'box_info' not in data or 'context' not in data or 'predicted_intake' not in data:
            logger.error("Missing box_info, context, or predicted_intake in request")
            return jsonify({'error': 'Missing box_info, context, or predicted_intake'}), 400
        box_info = data['box_info']
        context_text = data['context']
        predicted_intake = data['predicted_intake']
        logger.info("Received request to predict box intake")
        intake = predict_box_intake(context_text, box_info, predicted_intake)
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
            if total_retry_time >= 7:
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
                total_retry_time += 0.5
                continue
        for model_name in model_names:
            for attempt in range(max_retries):
                retry_delay = 0.5 * (2 ** attempt)
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
                break
    except Exception as e:
        logger.error(f"Test prompt failed: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
