# -*- coding: utf-8 -*-
"""Sustainable Smart City - Flask Version

Flask implementation with IBM Granite model support
"""

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import json
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import threading
import time
from io import StringIO

app = Flask(__name__)
app.secret_key = 'smart_city_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for reports and data (in production, use a database)
citizen_reports = []
kpi_data = {}
chat_history = []

class SmartCityAssistant:
    def __init__(self, hf_token):
        """Initialize the Smart City Assistant with IBM Granite model"""
        self.hf_token = hf_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use a simpler IBM model for better compatibility
        self.model_name = "ibm/granite-3b-code-instruct"  # Smaller, more stable model
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.model_loaded = False

        # Load model in background
        self.load_model()

    def load_model(self):
        """Load IBM Granite model with better error handling"""
        try:
            print("Loading IBM Granite model...")
            
            # Try to load the model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )

            self.model_loaded = True
            print("IBM Granite model loaded successfully!")

        except Exception as e:
            print(f"Error loading IBM Granite model: {e}")
            print("Falling back to DistilGPT2...")
            try:
                self.generator = pipeline(
                    "text-generation", 
                    model="distilgpt2",
                    device=0 if torch.cuda.is_available() else -1,
                    return_full_text=False,
                    pad_token_id=50256
                )
                self.model_loaded = True
                print("Fallback model (DistilGPT2) loaded successfully!")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                self.model_loaded = False

    def generate_response(self, prompt, max_tokens=300):
        """Generate response using loaded model"""
        try:
            if not self.model_loaded or not self.generator:
                return "I apologize, but the AI model is not available at the moment. Please try again later."
            
            # Format prompt for instruction-following
            if self.tokenizer and hasattr(self.tokenizer, 'eos_token_id'):
                # IBM Granite format
                formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            else:
                # Simple format for fallback model
                formatted_prompt = f"Question: {prompt}\nAnswer:"

            response = self.generator(
                formatted_prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else 50256
            )

            # Extract only the response part
            generated_text = response[0]['generated_text']
            
            if "### Response:\n" in generated_text:
                response_part = generated_text.split("### Response:\n")[-1].strip()
            elif "Answer:" in generated_text:
                response_part = generated_text.split("Answer:")[-1].strip()
            else:
                response_part = generated_text.replace(formatted_prompt, "").strip()

            # Ensure response is meaningful
            if len(response_part) < 10:
                response_part = "I understand your question about smart city services. Let me provide you with relevant information and recommendations based on best practices in urban planning and sustainability."

            return response_part

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."

    def policy_summarization(self, policy_text):
        """Summarize complex policy documents"""
        prompt = f"""
        Summarize the following city policy document in citizen-friendly language.
        Make it concise and highlight key points that affect residents:

        {policy_text[:2000]}

        Provide a summary with:
        1. Main objectives
        2. Key changes for citizens
        3. Implementation timeline
        """
        return self.generate_response(prompt, max_tokens=400)

    def process_citizen_feedback(self, report_data):
        """Process and categorize citizen feedback reports"""
        categories = {
            'water': ['water', 'pipe', 'leak', 'drainage', 'sewage'],
            'traffic': ['traffic', 'road', 'signal', 'parking', 'accident'],
            'environment': ['waste', 'pollution', 'noise', 'air', 'garbage'],
            'infrastructure': ['street', 'light', 'sidewalk', 'building', 'construction'],
            'safety': ['crime', 'safety', 'police', 'emergency', 'security']
        }

        description = report_data.get('description', '').lower()
        category = 'general'

        for cat, keywords in categories.items():
            if any(keyword in description for keyword in keywords):
                category = cat
                break

        prompt = f"""
        A citizen reported the following issue: {report_data.get('description', '')}
        Location: {report_data.get('location', 'Not specified')}

        Provide a professional acknowledgment response and suggest immediate actions.
        """

        ai_response = self.generate_response(prompt, max_tokens=200)

        report = {
            'id': len(citizen_reports) + 1,
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'description': report_data.get('description'),
            'location': report_data.get('location'),
            'contact': report_data.get('contact'),
            'priority': self.assess_priority(description),
            'ai_response': ai_response,
            'status': 'pending'
        }

        citizen_reports.append(report)
        return report

    def assess_priority(self, description):
        """Assess priority of citizen reports"""
        high_priority_keywords = ['emergency', 'burst', 'fire', 'accident', 'danger', 'urgent']
        medium_priority_keywords = ['broken', 'damaged', 'blocked', 'overflow']

        description_lower = description.lower()

        if any(keyword in description_lower for keyword in high_priority_keywords):
            return 'high'
        elif any(keyword in description_lower for keyword in medium_priority_keywords):
            return 'medium'
        else:
            return 'low'

    def kpi_forecasting(self, csv_data, kpi_type):
        """Forecast KPI values using machine learning"""
        try:
            df = pd.read_csv(StringIO(csv_data))

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                df['month'] = df['date'].dt.month
                df['year'] = df['date'].dt.year
                df['day_of_year'] = df['date'].dt.dayofyear

                target_col = df.columns[-1]
                feature_cols = ['month', 'year', 'day_of_year']

                X = df[feature_cols].values
                y = df[target_col].values

                model = LinearRegression()
                model.fit(X, y)

                last_date = df['date'].max()
                forecasts = []

                for i in range(1, 13):
                    future_date = last_date + timedelta(days=30*i)
                    features = [future_date.month, future_date.year, future_date.timetuple().tm_yday]
                    prediction = model.predict([features])[0]

                    forecasts.append({
                        'date': future_date.strftime('%Y-%m-%d'),
                        'predicted_value': round(prediction, 2),
                        'kpi_type': kpi_type
                    })

                avg_historical = np.mean(y)
                avg_forecast = np.mean([f['predicted_value'] for f in forecasts])
                trend = "increasing" if avg_forecast > avg_historical else "decreasing"

                prompt = f"""
                Analyze the {kpi_type} KPI forecast results:
                - Historical average: {avg_historical:.2f}
                - Forecasted average: {avg_forecast:.2f}
                - Trend: {trend}

                Provide insights and recommendations for city planning.
                """

                insights = self.generate_response(prompt, max_tokens=300)

                return {
                    'forecasts': forecasts,
                    'insights': insights,
                    'trend': trend,
                    'accuracy_score': 'Based on historical data patterns'
                }

        except Exception as e:
            return {'error': f'Error processing KPI data: {str(e)}'}

    def generate_eco_tips(self, keywords):
        """Generate eco-friendly tips based on keywords"""
        prompt = f"""
        Generate 5 practical and actionable eco-friendly tips related to: {', '.join(keywords)}

        Make the tips specific, easy to implement, and suitable for city residents.
        Include both individual actions and community-level suggestions.
        """
        return self.generate_response(prompt, max_tokens=400)

    def anomaly_detection(self, csv_data):
        """Detect anomalies in KPI data"""
        try:
            df = pd.read_csv(StringIO(csv_data))
            kpi_col = df.columns[-1]
            values = df[kpi_col].values.reshape(-1, 1)

            scaler = StandardScaler()
            values_scaled = scaler.fit_transform(values)

            detector = IsolationForest(contamination=0.1, random_state=42)
            anomalies = detector.fit_predict(values_scaled)

            anomaly_indices = np.where(anomalies == -1)[0]
            anomaly_records = []

            for idx in anomaly_indices:
                record = df.iloc[idx].to_dict()
                record['anomaly_score'] = abs(values_scaled[idx][0])
                anomaly_records.append(record)

            if anomaly_records:
                anomaly_values = [record[kpi_col] for record in anomaly_records]
                prompt = f"""
                Anomalies detected in city KPI data:
                - Anomalous values: {anomaly_values}
                - Normal range average: {np.mean(values):.2f}

                Analyze these anomalies and suggest possible causes and actions for city administrators.
                """
                analysis = self.generate_response(prompt, max_tokens=300)
            else:
                analysis = "No significant anomalies detected in the provided data."

            return {
                'anomalies_found': len(anomaly_records),
                'anomaly_records': anomaly_records,
                'analysis': analysis,
                'total_records': len(df)
            }

        except Exception as e:
            return {'error': f'Error detecting anomalies: {str(e)}'}

    def chat_assistant(self, message):
        """General chat assistant for city-related queries"""
        prompt = f"""
        You are a helpful Smart City Assistant. Answer the following question about urban planning,
        sustainability, city services, or civic matters:

        Question: {message}

        Provide a comprehensive and practical answer.
        """
        return self.generate_response(prompt, max_tokens=400)

    def traffic_route_suggestion(self, origin, destination, city):
        """Generate traffic route suggestions and famous places"""
        prompt = f"""
        A visitor is traveling from {origin} to {destination} in {city}.

        Provide:
        1. Suggested route with less traffic (general directions)
        2. Famous places/attractions near the destination
        3. Best time to travel to avoid traffic
        4. Local transportation options
        """
        return self.generate_response(prompt, max_tokens=400)

# Initialize the assistant
def get_assistant():
    if not hasattr(app, 'assistant'):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not found in environment variables")
            hf_token = "dummy_token"  # Will use fallback model
        app.assistant = SmartCityAssistant(hf_token)
    return app.assistant

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/policy')
def policy():
    return render_template('policy.html')

@app.route('/policy/summarize', methods=['POST'])
def policy_summarize():
    policy_text = request.form.get('policy_text')
    if not policy_text:
        flash('Please enter policy text to summarize', 'error')
        return redirect(url_for('policy'))
    
    assistant = get_assistant()
    summary = assistant.policy_summarization(policy_text)
    
    return render_template('policy.html', summary=summary, policy_text=policy_text)

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/reports/submit', methods=['POST'])
def submit_report():
    report_data = {
        'description': request.form.get('description'),
        'location': request.form.get('location'),
        'contact': request.form.get('contact')
    }
    
    if not report_data['description'] or not report_data['location']:
        flash('Please fill in both description and location', 'error')
        return redirect(url_for('reports'))
    
    assistant = get_assistant()
    report = assistant.process_citizen_feedback(report_data)
    
    flash('Report submitted successfully!', 'success')
    return render_template('reports.html', report=report)

@app.route('/kpi')
def kpi():
    return render_template('kpi.html')

@app.route('/kpi/forecast', methods=['POST'])
def kpi_forecast():
    if 'csv_file' not in request.files:
        flash('Please upload a CSV file', 'error')
        return redirect(url_for('kpi'))
    
    file = request.files['csv_file']
    kpi_type = request.form.get('kpi_type')
    
    if file.filename == '':
        flash('Please select a file', 'error')
        return redirect(url_for('kpi'))
    
    if file and file.filename.endswith('.csv'):
        csv_data = file.read().decode('utf-8')
        assistant = get_assistant()
        result = assistant.kpi_forecasting(csv_data, kpi_type)
        
        return render_template('kpi.html', result=result, kpi_type=kpi_type)
    
    flash('Please upload a valid CSV file', 'error')
    return redirect(url_for('kpi'))

@app.route('/eco')
def eco():
    return render_template('eco.html')

@app.route('/eco/tips', methods=['POST'])
def eco_tips():
    category = request.form.get('category')
    custom_keywords = request.form.get('custom_keywords', '')
    
    eco_categories = {
        "Energy": ["solar", "renewable", "efficiency", "conservation"],
        "Water": ["conservation", "recycling", "rainwater", "usage"],
        "Transportation": ["public transport", "cycling", "electric vehicles", "walking"],
        "Waste": ["recycling", "composting", "reduction", "reuse"],
        "Urban Gardening": ["composting", "gardening", "green spaces", "plants"]
    }
    
    keywords = eco_categories.get(category, [])
    if custom_keywords:
        keywords.extend([kw.strip() for kw in custom_keywords.split(',')])
    
    assistant = get_assistant()
    tips = assistant.generate_eco_tips(keywords)
    
    return render_template('eco.html', tips=tips, category=category)

@app.route('/anomaly')
def anomaly():
    return render_template('anomaly.html')

@app.route('/anomaly/detect', methods=['POST'])
def detect_anomaly():
    if 'csv_file' not in request.files:
        flash('Please upload a CSV file', 'error')
        return redirect(url_for('anomaly'))
    
    file = request.files['csv_file']
    
    if file.filename == '':
        flash('Please select a file', 'error')
        return redirect(url_for('anomaly'))
    
    if file and file.filename.endswith('.csv'):
        csv_data = file.read().decode('utf-8')
        assistant = get_assistant()
        result = assistant.anomaly_detection(csv_data)
        
        return render_template('anomaly.html', result=result)
    
    flash('Please upload a valid CSV file', 'error')
    return redirect(url_for('anomaly'))

@app.route('/chat')
def chat():
    return render_template('chat.html', chat_history=chat_history)

@app.route('/chat/send', methods=['POST'])
def chat_send():
    message = request.form.get('message')
    if not message:
        return redirect(url_for('chat'))
    
    assistant = get_assistant()
    response = assistant.chat_assistant(message)
    
    chat_history.append({'user': message, 'ai': response, 'timestamp': datetime.now()})
    
    return redirect(url_for('chat'))

@app.route('/traffic')
def traffic():
    return render_template('traffic.html')

@app.route('/traffic/route', methods=['POST'])
def traffic_route():
    origin = request.form.get('origin')
    destination = request.form.get('destination')
    city = request.form.get('city')
    
    if not all([origin, destination, city]):
        flash('Please fill in all fields', 'error')
        return redirect(url_for('traffic'))
    
    assistant = get_assistant()
    suggestions = assistant.traffic_route_suggestion(origin, destination, city)
    
    return render_template('traffic.html', suggestions=suggestions, 
                         origin=origin, destination=destination, city=city)

@app.route('/view-reports')
def view_reports():
    return render_template('view_reports.html', reports=citizen_reports)

@app.route('/api/status')
def api_status():
    assistant = get_assistant()
    return jsonify({
        'model_loaded': assistant.model_loaded,
        'model_name': assistant.model_name,
        'device': str(assistant.device),
        'total_reports': len(citizen_reports)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
