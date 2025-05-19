from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import re
import logging
import time

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
        model="gpt-4-turbo",  # Switch to GPT-4 Turbo
        temperature=0.1,  # Lower temperature for more deterministic output
        max_tokens=1000,
        timeout=30
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

You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members in Denmark. Your task is to predict the daily intake (new members per day) for the box at a Customer Acquisition Cost (CAC) of 17.5 EUR. Follow these steps exactly to calculate the daily intake. Do not return any other numerical value, such as the total retail value of the box. Only return the final predicted daily intake as a float (e.g., 50.0).

**Step 1: Start with a Baseline**
- Begin with a baseline intake of 35 members/day.

**Step 2: Apply Adjustments Based on Box Information**
- **Retail Value Adjustment**: The total retail value of the box influences intake. For every 10 EUR above 50 EUR, add a 2% boost to intake, up to a maximum of 30%. For example, a retail value of 150 EUR (100 EUR above 50) adds a 20% boost (100 / 10 * 2%), while a value of 250 EUR adds the maximum 30% boost.
- **Premium Products (>20 EUR)**: Each premium product adds a 5% boost to intake, up to a maximum of 25%. For example, 3 premium products add a 15% boost, 5 or more add a 25% boost.
- **Total Weight**: Weight up to 500g adds a 5% boost per 100g; weight above 500g adds a flat 15% boost. For example, 400g adds a 20% boost (4 * 5%), 600g adds a 15% boost.
- **Average Ratings**: For each of product, brand, and category ratings, add a 3% boost for every 0.1 increment above 4.0. For example, a rating of 4.2 adds 6% (0.2 * 3%). Sum the boosts from all three ratings.
- **Niche Products**: Each niche product reduces intake by 20%. For example, 1 niche product reduces intake by 20%, 2 niche products by 40%.
- **Free Gift Value and Rating**: Add 1% to intake for every 10 EUR of free gift value (e.g., 50 EUR adds 5%). Add an additional 5% if the free gift rating is above 4.0.
- **Seasonality**: If the launch month is early in the month (e.g., January, October), add a 5% boost. Otherwise, no adjustment.

**Step 3: Calculate the Total Adjustment**
- Sum the percentage boosts and reductions to get the total adjustment. For example, if boosts are +20% (retail value), +15% (premium products), +20% (weight), +9% (ratings), +10% (free gift), +5% (seasonality), and reductions are -20% (niche products), the total adjustment is 20 + 15 + 20 + 9 + 10 + 5 - 20 = 59%.
- Apply the total adjustment to the baseline: Adjusted Intake = Baseline * (1 + Total Adjustment / 100). For example, 35 * (1 + 0.59) = 55.65.

**Step 4: Clamp the Final Value**
- Ensure the final predicted intake is between 20 and 100 members/day. If the adjusted intake is below 20, set it to 20; if above 100, set it to 100.

**Examples**:
- **Example 1**:
  - Box Information: Number of products: 7, Number of premium products (>20 EUR): 3, Total weight: 400g, Number of niche products: 1, Average product rating: 4.2, Average category rating: 4.1, Average brand rating: 4.0, Free gift: Value: 50 EUR, Rating: 4.5, Launch month: January, Total retail value (for reference only, do not use as prediction): 150 EUR
  - Step 1: Baseline = 35 members/day.
  - Step 2:
    - Retail Value: 150 EUR (100 EUR above 50) = 20% boost (100 / 10 * 2%).
    - Premium products: 3 * 5% = 15% boost.
    - Weight: 400g = 4 * 5% = 20% boost.
    - Ratings: Product 4.2 (6%), Category 4.1 (3%), Brand 4.0 (0%) = 9% boost.
    - Niche products: 1 * 20% = 20% reduction.
    - Free gift: 50 EUR = 5% + 5% (rating > 4.0) = 10% boost.
    - Seasonality: Early-month = 5% boost.
  - Step 3: Total Adjustment = 20 + 15 + 20 + 9 + 10 + 5 - 20 = 59%.
  - Adjusted Intake = 35 * (1 + 0.59) = 55.65.
  - Step 4: Clamped Intake = 55.65 (within 20-100).
  - Output: 55.65

- **Example 2**:
  - Box Information: Number of products: 8, Number of premium products (>20 EUR): 5, Total weight: 600g, Number of niche products: 2, Average product rating: 4.3, Average category rating: 4.2, Average brand rating: 4.1, Free gift: Value: 65 EUR, Rating: 4.33, Launch month: March, Total retail value (for reference only, do not use as prediction): 250 EUR
  - Step 1: Baseline = 35 members/day.
  - Step 2:
    - Retail Value: 250 EUR (200 EUR above 50) = 30% boost (capped at 30%).
    - Premium products: 5 * 5% = 25% boost (capped).
    - Weight: 600g = 15% boost.
    - Ratings: Product 4.3 (9%), Category 4.2 (6%), Brand 4.1 (3%) = 18% boost.
    - Niche products: 2 * 20% = 40% reduction.
    - Free gift: 65 EUR = 6.5% + 5% (rating > 4.0) = 11.5% boost.
    - Seasonality: Not early-month = 0% boost.
  - Step 3: Total Adjustment = 30 + 25 + 15 + 18 + 11.5 - 40 = 59.5%.
  - Adjusted Intake = 35 * (1 + 0.595) = 55.825.
  - Step 4: Clamped Intake = 55.83 (within 20-100, rounded to 2 decimals).
  - Output: 55.83

Now, calculate the daily intake for the given box using the same steps. Return only the numerical value of the predicted daily intake as a float (e.g., 50.0). Do not return the total retail value or any other number.
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
        for i in range(1):  # Single run to minimize timeout risk
            logger.info(f"Sending request to LangChain (run {i+1}/1)")
            for attempt in range(3):
                try:
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
                        break
                    else:
                        logger.warning(f"Invalid intake format in run {i+1}: {result}")
                        raise ValueError("Invalid intake format")
                except Exception as e:
                    logger.warning(f"Run {i+1} failed (attempt {attempt+1}/3): {str(e)}")
                    if attempt == 2:
                        logger.error(f"All attempts failed for run {i+1}")
                        raise
                    time.sleep(2)
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

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
