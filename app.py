from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Verify environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set")
    raise ValueError("OPENAI_API_KEY is not set")

# Set up OpenAI with LangChain
try:
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=1000
    )
    logger.info("OpenAI model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI model: {str(e)}")
    raise

# Define prompt template
template = """
Context: {context}

Historical Data: {historical_data}

Box Information: {box_info}

You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members in Denmark. Based on the context, historical data, and box information, predict the daily intake (new members per day) for the box at a Customer Acquisition Cost (CAC) of 17.5 EUR.

Consider the following factors:
- Number of products, total retail value, and unique categories.
- Number of full-size products and premium products (>20 EUR).
- Total weight (as a proxy for box fullness, higher weight often correlates with higher intake).
- Average product rating, average brand rating, and average category rating (higher ratings generally increase intake).
- Presence of niche products (niche products may reduce intake due to lower relatability, reduce predicted intake by 10-15% per niche product).
- Free gift value and rating (higher value/rating often increases intake, add 5% to intake per 10 EUR of free gift value).
- Seasonality (boxes launched early in the month may have a 20-30% higher intake; adjust downward if not early-month).

Return only the numerical value of the predicted daily intake as a float (e.g., 150.0).
"""

try:
    prompt = PromptTemplate(
        input_variables=["context", "historical_data", "box_info"],
        template=template
    )
    logger.info("Prompt template created successfully")
except Exception as e:
    logger.error(f"Failed to create prompt template: {str(e)}")
    raise

# Create LLM chain
try:
    chain = LLMChain(llm=llm, prompt=prompt)
    logger.info("LLM chain created successfully")
except Exception as e:
    logger.error(f"Failed to create LLM chain: {str(e)}")
    raise

def predict_box_intake(context, historical_data, box_info):
    """Predict daily intake for a box using LangChain."""
    try:
        logger.info(f"Processing prediction request with context: {context[:100]}...")
        logger.info(f"Box info: {box_info[:100]}...")
        predictions = []
        for i in range(5):  # 5 runs for averaging
            logger.info(f"Sending request to LangChain (run {i+1}/5)")
            result = chain.run({
                "context": context,
                "historical_data": historical_data,
                "box_info": box_info
            })
            logger.info(f"Run {i+1} response: {result}")
            match = re.search(r'\d+\.\d+', result)
            if match:
                intake_float = float(match.group())
                if intake_float < 0:
                    logger.error("Negative intake value received")
                    raise ValueError("Intake cannot be negative")
                predictions.append(intake_float)
            else:
                logger.warning(f"Invalid intake format in run {i+1}: {result}")
        if not predictions:
            logger.error("No valid intake values collected")
            raise ValueError("No valid intake values collected")
        avg_intake = sum(predictions) / len(predictions)
        logger.info(f"Averaged intake from 5 runs: {avg_intake}")
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

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
