# -*- coding: utf-8 -*-
"""Sustainable Smart City - Streamlit Version (Fixed)

Uses Hugging Face Inference API instead of downloading models locally
"""

import os
import pandas as pd
import numpy as np
import requests
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import streamlit as st
from io import StringIO

class SmartCityAssistant:
    def __init__(self, hf_token):
        """Initialize the Smart City Assistant with Hugging Face API"""
        self.hf_token = hf_token
        self.api_url = "https://api-inference.huggingface.co/models/ibm-granite/granite-3.0-8b-instruct"
        self.headers = {"Authorization": f"Bearer {hf_token}"}

        # Storage for reports and data
        if 'citizen_reports' not in st.session_state:
            st.session_state.citizen_reports = []
        if 'kpi_data' not in st.session_state:
            st.session_state.kpi_data = {}

    def query_huggingface_api(self, prompt, max_tokens=300):
        """Query Hugging Face Inference API instead of loading model locally"""
        try:
            # Format prompt for instruction-following
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
                else:
                    return result.get('generated_text', '').strip()
            elif response.status_code == 503:
                # Model is loading, use fallback
                return self.fallback_response(prompt)
            else:
                st.warning(f"API Error {response.status_code}: Using fallback response")
                return self.fallback_response(prompt)
                
        except Exception as e:
            st.warning(f"API request failed: {e}. Using fallback response.")
            return self.fallback_response(prompt)

    def fallback_response(self, prompt):
        """Provide fallback responses when API is unavailable"""
        fallback_responses = {
            "policy": "Thank you for submitting the policy document. Based on standard urban policy analysis, key areas typically include: community impact, implementation timeline, budget considerations, and citizen benefits. Please refer to your local government website for detailed policy information.",
            
            "citizen_report": "Thank you for your report. We have received your submission and it has been categorized appropriately. Your report will be reviewed by the relevant department within 24-48 hours. We appreciate your engagement in making our city better.",
            
            "kpi_forecast": "Based on historical data patterns, the KPI shows measurable trends. Key factors to consider include seasonal variations, urban growth patterns, and infrastructure capacity. Regular monitoring and adaptive planning are recommended for optimal city management.",
            
            "eco_tips": "Here are some practical eco-friendly tips: 1) Reduce energy consumption by using LED lights and energy-efficient appliances. 2) Use public transportation or bike when possible. 3) Implement water conservation practices. 4) Participate in community recycling programs. 5) Support local sustainable businesses and initiatives.",
            
            "anomaly": "Data analysis completed. Any significant deviations from normal patterns should be investigated further. Consider factors such as seasonal changes, special events, or infrastructure modifications that might explain unusual readings.",
            
            "chat": "I'm here to help with smart city questions. While I'm currently operating in basic mode, I can provide general guidance on urban planning, sustainability, citizen services, and city management topics.",
            
            "traffic": "For optimal routing, consider: 1) Avoid peak hours (7-9 AM, 5-7 PM). 2) Use main arterial roads when possible. 3) Check local traffic apps for real-time conditions. 4) Explore public transportation options. 5) Plan for extra travel time during events or construction."
        }
        
        # Determine response type based on prompt content
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['policy', 'document', 'summarize']):
            return fallback_responses["policy"]
        elif any(word in prompt_lower for word in ['report', 'issue', 'problem']):
            return fallback_responses["citizen_report"]
        elif any(word in prompt_lower for word in ['forecast', 'kpi', 'predict']):
            return fallback_responses["kpi_forecast"]
        elif any(word in prompt_lower for word in ['eco', 'green', 'sustainable', 'tips']):
            return fallback_responses["eco_tips"]
        elif any(word in prompt_lower for word in ['anomaly', 'detect', 'unusual']):
            return fallback_responses["anomaly"]
        elif any(word in prompt_lower for word in ['traffic', 'route', 'travel']):
            return fallback_responses["traffic"]
        else:
            return fallback_responses["chat"]

    def generate_response(self, prompt, max_tokens=300):
        """Generate response using Hugging Face API or fallback"""
        return self.query_huggingface_api(prompt, max_tokens)

    def policy_summarization(self, policy_text):
        """Summarize complex policy documents"""
        prompt = f"""
        Summarize the following city policy document in citizen-friendly language.
        Make it concise and highlight key points that affect residents:

        {policy_text[:1500]}  # Limit input length

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

        # Auto-categorize based on keywords
        description = report_data.get('description', '').lower()
        category = 'general'

        for cat, keywords in categories.items():
            if any(keyword in description for keyword in keywords):
                category = cat
                break

        # Generate automated response
        prompt = f"""
        A citizen reported the following issue: {report_data.get('description', '')}
        Location: {report_data.get('location', 'Not specified')}

        Provide a professional acknowledgment response and suggest immediate actions.
        """

        ai_response = self.generate_response(prompt, max_tokens=200)

        # Store report
        report = {
            'id': len(st.session_state.citizen_reports) + 1,
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'description': report_data.get('description'),
            'location': report_data.get('location'),
            'contact': report_data.get('contact'),
            'priority': self.assess_priority(description),
            'ai_response': ai_response,
            'status': 'pending'
        }

        st.session_state.citizen_reports.append(report)
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
            # Parse CSV data
            df = pd.read_csv(StringIO(csv_data))

            # Prepare data for forecasting
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                # Create features
                df['month'] = df['date'].dt.month
                df['year'] = df['date'].dt.year
                df['day_of_year'] = df['date'].dt.dayofyear

                # Assume the last column is the target KPI
                target_col = df.columns[-1]
                feature_cols = ['month', 'year', 'day_of_year']

                X = df[feature_cols].values
                y = df[target_col].values

                # Train simple linear regression
                model = LinearRegression()
                model.fit(X, y)

                # Forecast next 12 months
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

                # Generate insights using AI
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

            # Assume last column is the KPI value
            kpi_col = df.columns[-1]
            values = df[kpi_col].values.reshape(-1, 1)

            # Standardize data
            scaler = StandardScaler()
            values_scaled = scaler.fit_transform(values)

            # Detect anomalies using Isolation Forest
            detector = IsolationForest(contamination=0.1, random_state=42)
            anomalies = detector.fit_predict(values_scaled)

            # Identify anomalous records
            anomaly_indices = np.where(anomalies == -1)[0]
            anomaly_records = []

            for idx in anomaly_indices:
                record = df.iloc[idx].to_dict()
                record['anomaly_score'] = abs(values_scaled[idx][0])
                anomaly_records.append(record)

            # Generate AI analysis
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

# Initialize Streamlit app
def main():
    st.set_page_config(
        page_title="Sustainable Smart City Assistant",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏙️ Sustainable Smart City Assistant")
    st.markdown("*AI-powered urban management and citizen services platform*")
    
    # Get HF Token
    hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    
    if not hf_token:
        st.error("❌ HF_TOKEN is missing. Please set it in Streamlit secrets or environment variables.")
        st.info("Add your Hugging Face token to use the AI features.")
        st.stop()

    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = SmartCityAssistant(hf_token)

    assistant = st.session_state.assistant

    # Display system info
    with st.sidebar:
        st.header("System Information")
        st.write("**Mode:** Cloud API (No local downloads)")
        st.write("**Model:** IBM Granite via HF API")
        st.success("✅ Ready to use!")

    # Navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏛️ Policy Summary", 
        "📝 Citizen Reports", 
        "📊 KPI Forecasting", 
        "🌱 Eco Tips", 
        "🔍 Anomaly Detection", 
        "💬 Chat Assistant", 
        "🚗 Traffic Routes",
        "📋 View Reports"
    ])

    # Policy Summarization Tab
    with tab1:
        st.header("📄 Policy Document Summarization")
        st.write("Upload or paste policy documents to get citizen-friendly summaries")
        
        policy_text = st.text_area(
            "Enter policy document text:",
            height=200,
            placeholder="Paste your policy document here..."
        )
        
        if st.button("Generate Summary", key="policy_summary"):
            if policy_text:
                with st.spinner("Generating summary..."):
                    summary = assistant.policy_summarization(policy_text)
                    st.success("Summary generated!")
                    st.markdown("### Summary")
                    st.write(summary)
            else:
                st.error("Please enter policy text to summarize")

    # Citizen Reports Tab
    with tab2:
        st.header("📝 Citizen Feedback Reports")
        st.write("Submit issues and get automated responses")
        
        col1, col2 = st.columns(2)
        
        with col1:
            description = st.text_area(
                "Issue Description:",
                height=100,
                placeholder="Describe the issue you want to report..."
            )
            
        with col2:
            location = st.text_input(
                "Location:",
                placeholder="Enter the location of the issue"
            )
            contact = st.text_input(
                "Contact Information:",
                placeholder="Your email or phone number"
            )
        
        if st.button("Submit Report", key="submit_report"):
            if description and location:
                report_data = {
                    'description': description,
                    'location': location,
                    'contact': contact
                }
                
                with st.spinner("Processing report..."):
                    report = assistant.process_citizen_feedback(report_data)
                    
                st.success("Report submitted successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Report ID", report['id'])
                with col2:
                    st.metric("Category", report['category'].title())
                with col3:
                    priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    st.metric("Priority", f"{priority_color[report['priority']]} {report['priority'].title()}")
                
                st.markdown("### AI Response")
                st.info(report['ai_response'])
            else:
                st.error("Please fill in both description and location")

    # KPI Forecasting Tab
    with tab3:
        st.header("📊 KPI Forecasting")
        st.write("Upload CSV data to forecast city KPIs")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
        with col2:
            kpi_type = st.selectbox(
                "KPI Type:",
                ["Energy Consumption", "Water Usage", "Traffic Volume", "Waste Generation", "Air Quality", "Other"]
            )
        
        if uploaded_file is not None:
            # Read and display CSV
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            if st.button("Generate Forecast", key="kpi_forecast"):
                csv_string = uploaded_file.getvalue().decode('utf-8')
                
                with st.spinner("Generating forecast..."):
                    result = assistant.kpi_forecasting(csv_string, kpi_type)
                    
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success("Forecast generated!")
                    
                    # Display forecasts
                    forecast_df = pd.DataFrame(result['forecasts'])
                    st.line_chart(forecast_df.set_index('date')['predicted_value'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Forecast Data")
                        st.dataframe(forecast_df)
                    
                    with col2:
                        st.markdown("### AI Insights")
                        st.write(result['insights'])
                        st.metric("Trend", result['trend'].title())

    # Eco Tips Tab
    with tab4:
        st.header("🌱 Eco-Friendly Tips Generator")
        st.write("Get personalized sustainability recommendations")
        
        # Predefined keywords
        eco_categories = {
            "Energy": ["solar", "renewable", "efficiency", "conservation"],
            "Water": ["conservation", "recycling", "rainwater", "usage"],
            "Transportation": ["public transport", "cycling", "electric vehicles", "walking"],
            "Waste": ["recycling", "composting", "reduction", "reuse"],
            "Urban Gardening": ["composting", "gardening", "green spaces", "plants"]
        }
        
        selected_category = st.selectbox("Select Category:", list(eco_categories.keys()))
        custom_keywords = st.text_input("Additional Keywords (comma-separated):", "")
        
        if st.button("Generate Eco Tips", key="eco_tips"):
            keywords = eco_categories[selected_category]
            if custom_keywords:
                keywords.extend([kw.strip() for kw in custom_keywords.split(',')])
            
            with st.spinner("Generating eco-friendly tips..."):
                tips = assistant.generate_eco_tips(keywords)
                st.success("Tips generated!")
                st.markdown("### 🌿 Your Personalized Eco Tips")
                st.write(tips)

    # Anomaly Detection Tab
    with tab5:
        st.header("🔍 Anomaly Detection in City Data")
        st.write("Upload KPI data to detect unusual patterns")
        
        uploaded_file = st.file_uploader("Upload CSV file for anomaly detection", type=['csv'], key="anomaly_csv")
        
        if uploaded_file is not None:
            # Read and display CSV
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            if st.button("Detect Anomalies", key="detect_anomalies"):
                csv_string = uploaded_file.getvalue().decode('utf-8')
                
                with st.spinner("Detecting anomalies..."):
                    result = assistant.anomaly_detection(csv_string)
                    
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success("Anomaly detection completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", result['total_records'])
                    with col2:
                        st.metric("Anomalies Found", result['anomalies_found'])
                    with col3:
                        anomaly_rate = (result['anomalies_found'] / result['total_records']) * 100
                        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
                    
                    if result['anomaly_records']:
                        st.markdown("### Anomalous Records")
                        anomaly_df = pd.DataFrame(result['anomaly_records'])
                        st.dataframe(anomaly_df)
                    
                    st.markdown("### AI Analysis")
                    st.write(result['analysis'])

    # Chat Assistant Tab
    with tab6:
        st.header("💬 Smart City Chat Assistant")
        st.write("Ask questions about urban planning, sustainability, and city services")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(ai_msg)
        
        # Chat input
        user_message = st.chat_input("Ask me anything about smart cities...")
        
        if user_message:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(user_message)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ai_response = assistant.chat_assistant(user_message)
                    st.write(ai_response)
            
            # Save to chat history
            st.session_state.chat_history.append((user_message, ai_response))

    # Traffic Routes Tab
    with tab7:
        st.header("🚗 Traffic Route Suggestions")
        st.write("Get optimal routes and nearby attractions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            origin = st.text_input("From:", placeholder="Enter starting location")
        with col2:
            destination = st.text_input("To:", placeholder="Enter destination")
        with col3:
            city = st.text_input("City:", placeholder="Enter city name")
        
        if st.button("Get Route Suggestions", key="traffic_route"):
            if origin and destination and city:
                with st.spinner("Generating route suggestions..."):
                    suggestions = assistant.traffic_route_suggestion(origin, destination, city)
                    st.success("Route suggestions generated!")
                    st.markdown("### 🗺️ Route Information")
                    st.write(suggestions)
            else:
                st.error("Please fill in all fields (From, To, City)")

    # View Reports Tab
    with tab8:
        st.header("📋 All Citizen Reports")
        st.write("View and manage submitted reports")
        
        if st.session_state.citizen_reports:
            # Display summary statistics
            total_reports = len(st.session_state.citizen_reports)
            priority_counts = {}
            category_counts = {}
            
            for report in st.session_state.citizen_reports:
                priority = report['priority']
                category = report['category']
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reports", total_reports)
            with col2:
                st.metric("High Priority", priority_counts.get('high', 0))
            with col3:
                st.metric("Medium Priority", priority_counts.get('medium', 0))
            with col4:
                st.metric("Low Priority", priority_counts.get('low', 0))
            
            # Display reports table
            reports_df = pd.DataFrame(st.session_state.citizen_reports)
            st.dataframe(
                reports_df[['id', 'timestamp', 'category', 'priority', 'location', 'status']],
                use_container_width=True
            )
            
            # Detailed view
            selected_report_id = st.selectbox("Select Report for Details:", 
                                            [f"Report #{r['id']}" for r in st.session_state.citizen_reports])
            
            if selected_report_id:
                report_id = int(selected_report_id.split('#')[1])
                selected_report = next(r for r in st.session_state.citizen_reports if r['id'] == report_id)
                
                st.markdown("### Report Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**ID:** {selected_report['id']}")
                    st.write(f"**Category:** {selected_report['category'].title()}")
                    st.write(f"**Priority:** {selected_report['priority'].title()}")
                    st.write(f"**Status:** {selected_report['status'].title()}")
                
                with col2:
                    st.write(f"**Location:** {selected_report['location']}")
                    st.write(f"**Contact:** {selected_report['contact']}")
                    st.write(f"**Timestamp:** {selected_report['timestamp']}")
                
                st.markdown("**Description:**")
                st.write(selected_report['description'])
                
                st.markdown("**AI Response:**")
                st.info(selected_report['ai_response'])
        else:
            st.info("No reports submitted yet. Go to the Citizen Reports tab to submit your first report!")

    # Footer
    st.markdown("---")
    st.markdown("**🏙️ Sustainable Smart City Assistant** - Powered by AI for better urban living")

if __name__ == "__main__":
    main()
