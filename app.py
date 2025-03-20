# from flask import Flask, jsonify, request
# from Finance_Project_text import predict_next_40_days, generate_investment_advice
# from flask_cors import CORS 

# app = Flask(__name__)

# CORS(app)

# # Basic root route
# @app.route('/')
# def index():
#     return "Welcome to the Finance Prediction API!"
# # API to get stock predictions
# @app.route('/api/predict', methods=['GET'])
# def predict_stock():
#     company = request.args.get('company', 'AAPL')  # Default to Apple stock
#     # days = request.args.get('days', 7)
#     model_path = "apple_stock_model.keras"  # Update with correct model path
#     scaler_path = "apple_scaler.pkl"  # Update with correct scaler path

#     try:
#         dates, prices = predict_next_40_days(company, model_path, scaler_path)
#         return jsonify({"company": company, "dates": dates, "prices": prices.tolist()})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # API to get investment advice
# @app.route('/api/investment_advice', methods=['GET'])
# def investment_advice():
#     company = request.args.get('company', 'AAPL')  # Default to Apple stock
#     model_path = "apple_stock_model.keras"  
#     scaler_path = "apple_scaler.pkl"  

#     try:
#         _, prices = predict_next_40_days(company, model_path, scaler_path)
#         advice = generate_investment_advice(prices, company)
#         return jsonify({"company": company, "investment_advice": advice})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



# ///////////////////////////////////////////////////////////////////












# from flask import Flask, send_from_directory, jsonify, request
# from Finance_Project_text import predict_next_40_days, generate_investment_advice
# from flask_cors import CORS

# app = Flask(__name__)

# CORS(app)

# # Mapping of companies to their model and scaler paths
# company_model_mapping = {
#     'AAPL': {'model': 'apple_stock_model.keras', 'scaler': 'apple_scaler.pkl'},
#     'MSFT': {'model': 'microsoft_stock_model.keras', 'scaler': 'microsoft_scaler.pkl'},
#     'GOOGL': {'model': 'google_stock_model.keras', 'scaler': 'google_scaler.pkl'},
#     # Add more companies here as needed
# }

# # Basic root route
# @app.route('/')
# def index():
#     return "Welcome to the Finance Prediction API!"

# # API to get stock predictions
# @app.route('/api/predict', methods=['GET'])
# def predict_stock():
#     company = request.args.get('company','AAPL').upper()  # Default to Apple stock, and make it uppercase
#     days = request.args.get('days', 4, type=int)  # Default to 40 days
#     # Get model and scaler for the selected company
#     company_data = company_model_mapping.get(company)

#     if not company_data:
#         return jsonify({"error": "Company not found"}), 404

#     model_path = company_data['model']
#     scaler_path = company_data['scaler']

#     try:
#         dates, prices = predict_next_40_days(company, model_path, scaler_path, days)
#         return jsonify({"company": company, "dates": dates, "prices": prices.tolist()})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # API to get investment advice
# @app.route('/api/investment_advice', methods=['GET'])
# def investment_advice():
#     company = request.args.get('company', 'AAPL').upper()  # Default to Apple stock, and make it uppercase
#     days = request.args.get('days', 4, type=int)
#     # Get model and scaler for the selected company
#     company_data = company_model_mapping.get(company)

#     if not company_data:
#         return jsonify({"error": "Company not found"}), 404

#     model_path = company_data['model']
#     scaler_path = company_data['scaler']

#     try:
#         _, prices = predict_next_40_days(company, model_path, scaler_path, days)
#         advice = generate_investment_advice(prices, company)
#         return jsonify({"company": company, "investment_advice": advice})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/images/<image_name>", methods=["GET"])
# def serve_image(image_name):
#     fetch_data_from_yf_for_images(company)
#     image_dir = "static/images"
#     return send_from_directory(image_dir, image_name)

# if __name__ == '__main__':
#     app.run(debug=True)





from flask import Flask, send_from_directory, jsonify, request
from Finance_Project_text import predict_next_40_days, generate_investment_advice
from digrams import fetch_data_from_yf_for_images  # Import the function
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Mapping of companies to their model and scaler paths
company_model_mapping = {
    'AAPL': {'model': 'apple_stock_model.keras', 'scaler': 'apple_scaler.pkl'},
    'MSFT': {'model': 'microsoft_stock_model.keras', 'scaler': 'microsoft_scaler.pkl'},
    'GOOGL': {'model': 'google_stock_model.keras', 'scaler': 'google_scaler.pkl'},
    # Add more companies here as needed
}

# Basic root route
@app.route('/')
def index():
    return "Welcome to the Finance Prediction API!"

# API to get stock predictions
@app.route('/api/predict', methods=['GET'])
def predict_stock():
    company = request.args.get('company', 'AAPL').upper()  # Default to Apple stock, and make it uppercase
    days = request.args.get('days', 4, type=int)  # Default to 4 days
    # Get model and scaler for the selected company
    company_data = company_model_mapping.get(company)

    if not company_data:
        return jsonify({"error": "Company not found"}), 404

    model_path = company_data['model']
    scaler_path = company_data['scaler']

    try:
        dates, prices = predict_next_40_days(company, model_path, scaler_path, days)
        return jsonify({"company": company, "dates": dates, "prices": prices.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to get investment advice
@app.route('/api/investment_advice', methods=['GET'])
def investment_advice():
    company = request.args.get('company', 'AAPL').upper()  # Default to Apple stock, and make it uppercase
    days = request.args.get('days', 4, type=int)
    # Get model and scaler for the selected company
    company_data = company_model_mapping.get(company)

    if not company_data:
        return jsonify({"error": "Company not found"}), 404

    model_path = company_data['model']
    scaler_path = company_data['scaler']

    try:
        _, prices = predict_next_40_days(company, model_path, scaler_path, days)
        advice = generate_investment_advice(prices, company)
        return jsonify({"company": company, "investment_advice": advice})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to generate and serve images for stock data
@app.route("/images/<company>", methods=["GET"])
def serve_images(company):
    try:
        # Fetch data and generate the images
        fetch_data_from_yf_for_images(company, "2024-01-01")  # Replace with the desired start date
        
        # Collect image paths
        image_dir = os.path.join("static", "images", company)
        
        # Get all images generated for the company
        images = []
        for file_name in os.listdir(image_dir):
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                images.append(f"/images/{company}/{file_name}")

        if images:
            return jsonify({"company": company, "images": images})
        else:
            return jsonify({"error": "No images found for this company"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to serve the actual image file
@app.route("/images/<company>/<image_name>", methods=["GET"])
def serve_image(company, image_name):
    image_dir = os.path.join("static", "images", company)
    
    # Ensure the requested image exists
    if os.path.exists(os.path.join(image_dir, image_name)):
        return send_from_directory(image_dir, image_name)
    else:
        return jsonify({"error": "Image not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
