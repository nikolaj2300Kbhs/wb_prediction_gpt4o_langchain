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

     Future Box Information: {future_box_info}

     You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members in Denmark. Based on the context, historical data, and future box information, predict the daily intake (new members per day) for the future box at a Customer Acquisition Cost (CAC) of 17.5 EUR. Consider factors such as the number of products, total retail value, unique categories, full-size products, premium products (>20 EUR), total weight, average product rating, average brand rating, average category rating, niche products, free gifts, and seasonality. Return only the numerical value of the predicted daily intake as a float (e.g., 150.0).
     """

     try:
         prompt = PromptTemplate(
             input_variables=["context", "historical_data", "future_box_info"],
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

     def predict_box_intake(context, historical_data, future_box_info):
         """Predict daily intake for a future box using LangChain."""
         try:
             predictions = []
             for i in range(5):  # 5 runs for averaging
                 logger.info(f"Sending request to LangChain (run {i+1}/5)")
                 result = chain.run({
                     "context": context,
                     "historical_data": historical_data,
                     "future_box_info": future_box_info
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
         """Endpoint for predicting future box intake."""
         try:
             data = request.get_json()
             if not data or 'future_box_info' not in data or 'context' not in data:
                 logger.error("Missing future_box_info or context in request")
                 return jsonify({'error': 'Missing future_box_info or context'}), 400
             historical_data = data.get('historical_data', 'No historical data provided')
             future_box_info = data['future_box_info']
             context_text = data['context']
             logger.info("Received request to predict box intake")
             intake = predict_box_intake(context_text, historical_data, future_box_info)
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