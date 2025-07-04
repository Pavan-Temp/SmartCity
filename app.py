import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import io
import os
import tempfile 
import base64
import json
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any
import hashlib
import random
from dotenv import load_dotenv
import os
load_dotenv()

# Trcy to import fitz (PyMuPDF) - make it optional
try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("PyMuPDF not installed. PDF support disabled. Install with: pip install PyMuPDF")

# Set page configuration
st.set_page_config(
    page_title="Smart City Assistant",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern interactive styling
st.markdown("""
<style>
    /* Main App Styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
    }
    
    /* Card Styles */
    .card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        color: #333;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .green-card {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #2d5016;
        border: 1px solid rgba(255, 255, 255, 0.3);
        font-weight: 500;
    }
    
    .alert-card {
        background: linear-gradient(135deg, #ffd3a5 0%, #fd9853 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #ff6b35;
        color: #8b4513;
        font-weight: 500;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #dc3545;
        color: #721c24;
        font-weight: 500;
    }
    
    /* Chat Message Styles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        font-weight: 500;
        animation: slideInRight 0.5s ease;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        font-weight: 500;
        animation: slideInLeft 0.5s ease;
    }
    
    /* Header Styles */
    .header {
        color: #ffffff;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 2.5rem;
        margin-bottom: 20px;
    }
    
    .subheader {
        color: #f0f0f0;
        font-size: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }
    
    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        border: 2px solid transparent;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Feedback Item Styles */
    .feedback-item {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        color: #333;
        transition: transform 0.3s ease;
    }
    
    .feedback-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Input Styles */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        color: #333;
        font-weight: 500;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        color: #333;
        font-weight: 500;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Selectbox Styles */
    .stSelectbox>div>div>select {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        color: #333;
        font-weight: 500;
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
        border-radius: 10px;
        border: none;
        color: #2d5016;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        border-radius: 10px;
        border: none;
        color: white;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffd3a5 0%, #fd9853 100%);
        border-radius: 10px;
        border: none;
        color: #8b4513;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .card, .green-card, .alert-card, .danger-card {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Sample data generators
def generate_sample_csv():
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    usage = [150, 140, 160, 180, 220, 280, 320, 310, 240, 190, 170, 155]
    df = pd.DataFrame({'Month': months, 'Usage': usage})
    return df

def generate_sample_city_data():
    """Generate sample city KPI data"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    
    # Simulate different city metrics
    energy_consumption = np.random.normal(1000, 100, len(dates))
    water_usage = np.random.normal(800, 80, len(dates))
    waste_generation = np.random.normal(500, 50, len(dates))
    air_quality_index = np.random.normal(85, 15, len(dates))
    traffic_density = np.random.normal(70, 10, len(dates))
    
    # Add some anomalies
    energy_consumption[6] = 1500  # July spike
    water_usage[3] = 1200  # April spike
    
    return pd.DataFrame({
        'Date': dates,
        'Energy_Consumption': energy_consumption,
        'Water_Usage': water_usage,
        'Waste_Generation': waste_generation,
        'Air_Quality_Index': air_quality_index,
        'Traffic_Density': traffic_density
    })

def load_sample_policies():
    """Load sample policy documents"""
    return [
        {
            'title': 'Smart City Energy Policy 2024',
            'content': 'This policy outlines the city\'s commitment to renewable energy sources, smart grid implementation, and energy efficiency measures. Key provisions include 30% reduction in energy consumption by 2030, mandatory solar installations for new buildings, and electric vehicle charging infrastructure development.',
            'category': 'Energy',
            'date': '2024-01-15'
        },
        {
            'title': 'Waste Management Guidelines',
            'content': 'Comprehensive waste segregation and recycling program for the city. Citizens must separate organic, recyclable, and hazardous waste. Collection schedule: organic waste daily, recyclables twice weekly, hazardous materials monthly.',
            'category': 'Environment',
            'date': '2024-02-20'
        },
        {
            'title': 'Public Transportation Enhancement Plan',
            'content': 'Investment in electric buses, metro expansion, and bike-sharing programs. The plan includes 200 new electric buses, 50km metro extension, and 1000 bike-sharing stations across the city by 2025.',
            'category': 'Transportation',
            'date': '2024-03-10'
        }
    ]

# Initialize session state
def initialize_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'citizen_feedback' not in st.session_state:
        st.session_state.citizen_feedback = []
    if 'city_kpis' not in st.session_state:
        st.session_state.city_kpis = generate_sample_city_data()
    if 'policies' not in st.session_state:
        st.session_state.policies = load_sample_policies()

initialize_session_state()

# Model loading with proper authentication
@st.cache_resource
def load_model():
    """Load IBM Granite model with HuggingFace token"""
    try:
        # You can set your HuggingFace token here or use environment variable
        
        hf_token = os.getenv("HF_TOKEN") # Your token
        
        model_name = "ibm-granite/granite-3.3-2b-instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=hf_token
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto"
        )
        
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Using fallback response generation...")
        return None, None

def generate_text(prompt, max_new_tokens=500):
    """Generate text using IBM Granite model or fallback"""
    try:
        if st.session_state.model is None or st.session_state.tokenizer is None:
            st.session_state.model, st.session_state.tokenizer = load_model()
        
        if st.session_state.model is None:
            # Fallback response for demo purposes
            return generate_fallback_response(prompt)
        
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.eos_token_id
            )
        
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return generate_fallback_response(prompt)

def generate_fallback_response(prompt):
    """Generate fallback responses when model is not available"""
    if "electricity" in prompt.lower() or "energy" in prompt.lower():
        return """Here are some practical energy-saving tips:
• Use LED bulbs instead of incandescent bulbs
• Set AC temperature to 24°C or higher
• Unplug electronics when not in use
• Use natural light during the day
• Consider solar panels for your home"""
    
    elif "transportation" in prompt.lower() or "vehicle" in prompt.lower():
        return """Sustainable transportation options:
• Use public transportation when possible
• Consider carpooling or ride-sharing
• Walk or cycle for short distances
• Choose electric or hybrid vehicles
• Work from home when feasible"""
    
    elif "waste" in prompt.lower():
        return """Waste reduction strategies:
• Practice the 3 R's: Reduce, Reuse, Recycle
• Segregate waste properly
• Compost organic waste
• Avoid single-use plastics
• Buy products with minimal packaging"""
    
    else:
        return """Thank you for your question about sustainability. Here are some general tips:
• Conserve water and energy
• Use eco-friendly products
• Support local and sustainable businesses
• Reduce, reuse, and recycle
• Choose sustainable transportation options"""

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    if not PDF_SUPPORT:
        return None
    
    try:
        pdf_bytes = pdf_file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def detect_anomalies(df, column, threshold=2):
    """Simple anomaly detection using z-score"""
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores > threshold]

def search_policies(query, policies):
    """Simple policy search based on keywords"""
    results = []
    query_lower = query.lower()
    
    for policy in policies:
        title_match = query_lower in policy['title'].lower()
        content_match = query_lower in policy['content'].lower()
        category_match = query_lower in policy['category'].lower()
        
        if title_match or content_match or category_match:
            results.append(policy)
    
    return results

def get_csv_download_link(df):
    """Create download link for CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_usage_data.csv" style="color: #388e3c; text-decoration: underline;">Download Sample CSV</a>'
    return href

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🏙️ Smart City Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your AI companion for sustainable urban living</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation buttons
    pages = {
        "🏠 City Dashboard": "Dashboard",
        "💬 Chat Assistant": "Chatbot",
        "📊 KPI Forecasting": "Predictor",
        "📄 Document Summarizer": "Summarizer",
        "🔍 Policy Search": "PolicySearch",
        "📝 Citizen Feedback": "Feedback",
        "⚠️ Anomaly Detection": "Anomaly",
        "🌱 Eco Tips": "EcoTips",
        "🧍 Personal Recommendations": "Recommendations"
    }
    
    for page_name, page_key in pages.items():
        if st.button(page_name, key=f"nav_{page_key}"):
            st.session_state.current_page = page_key
    
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Created for IBM Hackathon</div>", unsafe_allow_html=True)

# Main Content Area
# City Dashboard
if st.session_state.current_page == "Dashboard":
    st.markdown("<h1 class='header'>🏠 Smart City Health Dashboard</h1>", unsafe_allow_html=True)
    
    # Add refresh button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("<p class='subheader'>Real-time city metrics and performance indicators</p>", unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Refresh Data", key="refresh_dashboard"):
            st.session_state.city_kpis = generate_sample_city_data()
            st.success("✅ Data refreshed!")
            st.rerun()
    
    # Key Metrics Row with enhanced styling
    st.markdown("<h3>📊 Key Performance Indicators</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    current_energy = st.session_state.city_kpis['Energy_Consumption'].iloc[-1]
    current_water = st.session_state.city_kpis['Water_Usage'].iloc[-1]
    current_air = st.session_state.city_kpis['Air_Quality_Index'].iloc[-1]
    current_waste = st.session_state.city_kpis['Waste_Generation'].iloc[-1]
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("⚡ Energy Consumption", f"{current_energy:.0f} MWh", f"{np.random.uniform(-10, 5):.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("💧 Water Usage", f"{current_water:.0f} ML", f"{np.random.uniform(-5, 8):.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("🌬️ Air Quality Index", f"{current_air:.0f}", f"{np.random.uniform(-15, 3):.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("🗑️ Waste Generated", f"{current_waste:.0f} Tons", f"{np.random.uniform(-12, 2):.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Interactive Charts with selection
    st.markdown("<h3>📈 Trend Analysis</h3>", unsafe_allow_html=True)
    
    # Chart selection
    chart_type = st.selectbox("📊 Select visualization:", 
                             ["Line Charts", "Bar Charts", "Area Charts"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>⚡ Energy Consumption Trend</h4>", unsafe_allow_html=True)
        if chart_type == "Line Charts":
            fig_energy = px.line(st.session_state.city_kpis, x='Date', y='Energy_Consumption',
                                title="Monthly Energy Usage")
        elif chart_type == "Bar Charts":
            fig_energy = px.bar(st.session_state.city_kpis, x='Date', y='Energy_Consumption',
                               title="Monthly Energy Usage")
        else:
            fig_energy = px.area(st.session_state.city_kpis, x='Date', y='Energy_Consumption',
                               title="Monthly Energy Usage")
        
        fig_energy.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_energy, use_container_width=True)
    
    with col2:
        st.markdown("<h4>💧 Water Usage Pattern</h4>", unsafe_allow_html=True)
        if chart_type == "Line Charts":
            fig_water = px.line(st.session_state.city_kpis, x='Date', y='Water_Usage',
                               title="Monthly Water Consumption")
        elif chart_type == "Bar Charts":
            fig_water = px.bar(st.session_state.city_kpis, x='Date', y='Water_Usage',
                              title="Monthly Water Consumption")
        else:
            fig_water = px.area(st.session_state.city_kpis, x='Date', y='Water_Usage',
                              title="Monthly Water Consumption")
        
        fig_water.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_water, use_container_width=True)
    
    # Environmental Quality Overview
    st.markdown("<h3>🌍 Environmental Quality</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_air = px.line(st.session_state.city_kpis, x='Date', y='Air_Quality_Index',
                         title="🌬️ Air Quality Index")
        fig_air.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Unhealthy Level")
        fig_air.update_layout(height=300)
        st.plotly_chart(fig_air, use_container_width=True)
        
    with col2:
        fig_waste = px.line(st.session_state.city_kpis, x='Date', y='Waste_Generation',
                           title="🗑️ Waste Generation")
        fig_waste.update_layout(height=300)
        st.plotly_chart(fig_waste, use_container_width=True)
    
    # Recent Alerts with improved styling
    st.markdown("<h3>⚠️ System Alerts</h3>", unsafe_allow_html=True)
    
    # Simulate some alerts based on data
    alerts = []
    high_energy_months = st.session_state.city_kpis[st.session_state.city_kpis['Energy_Consumption'] > st.session_state.city_kpis['Energy_Consumption'].mean() + st.session_state.city_kpis['Energy_Consumption'].std()]
    if not high_energy_months.empty:
        for _, month_data in high_energy_months.iterrows():
            alerts.append({
                'type': 'warning',
                'message': f"⚡ Energy consumption spike detected in {month_data['Date'].strftime('%B %Y')} - {month_data['Energy_Consumption']:.0f} MWh",
                'icon': '⚠️'
            })
    
    high_water_months = st.session_state.city_kpis[st.session_state.city_kpis['Water_Usage'] > st.session_state.city_kpis['Water_Usage'].mean() + st.session_state.city_kpis['Water_Usage'].std()]
    if not high_water_months.empty:
        for _, month_data in high_water_months.iterrows():
            alerts.append({
                'type': 'info',
                'message': f"💧 Water usage above normal in {month_data['Date'].strftime('%B %Y')} - {month_data['Water_Usage']:.0f} ML",
                'icon': '💧'
            })
    
    if alerts:
        for alert in alerts[:3]:  # Show only first 3 alerts
            if alert['type'] == 'warning':
                st.markdown("<div class='alert-card'>", unsafe_allow_html=True)
                st.warning(f"{alert['icon']} {alert['message']}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.info(f"{alert['icon']} {alert['message']}")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='green-card'>", unsafe_allow_html=True)
        st.success("✅ All systems operating normally - No alerts at this time")
        st.markdown("</div>", unsafe_allow_html=True)

# Chat Assistant
elif st.session_state.current_page == "Chatbot":
    st.markdown("<h1 class='header'>💬 Smart City Chat Assistant</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Ask anything about sustainability, energy-saving, or smart city policies.</p>", unsafe_allow_html=True)
    
    # Example questions with better interactivity
    st.markdown("<h4>🎯 Try these popular questions:</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💡 How to save electricity at home?", use_container_width=True):
            st.session_state.chat_history.append(("You", "How to save electricity at home?"))
            with st.spinner("💡 Generating energy-saving tips..."):
                response = generate_text("Suggest practical tips for saving electricity at home:")
                st.session_state.chat_history.append(("Assistant", response))
            st.success("✅ Energy tips generated!")
            st.rerun()
    
    with col2:
        if st.button("🚗 Electric vehicle policies?", use_container_width=True):
            st.session_state.chat_history.append(("You", "What are the latest electric vehicle policies?"))
            with st.spinner("🚗 Researching EV policies..."):
                response = generate_text("Explain electric vehicle policies and incentives:")
                st.session_state.chat_history.append(("Assistant", response))
            st.success("✅ Policy information retrieved!")
            st.rerun()
    
    with col3:
        if st.button("🌱 Sustainable living tips", use_container_width=True):
            st.session_state.chat_history.append(("You", "Give me sustainable living tips"))
            with st.spinner("🌱 Compiling sustainability tips..."):
                response = generate_text("Provide sustainable living tips for urban residents:")
                st.session_state.chat_history.append(("Assistant", response))
            st.success("✅ Sustainability guide ready!")
            st.rerun()
    
    # Input for custom questions with better UX
    user_question = st.text_input("💭 Ask me anything about sustainability, smart cities, or green living...", 
                                  key="chat_input", 
                                  placeholder="e.g., How can I reduce my carbon footprint?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_clicked = st.button("🚀 Ask", key="ask_button", use_container_width=True)
    
    if ask_clicked and user_question:
        st.session_state.chat_history.append(("You", user_question))
        with st.spinner("🤔 Thinking..."):
            response = generate_text(f"Answer this question about sustainability or smart cities: {user_question}")
            st.session_state.chat_history.append(("Assistant", response))
        st.success("✅ Response generated!")
        st.rerun()
    elif ask_clicked and not user_question:
        st.warning("⚠️ Please enter a question first!")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chat history with styled message bubbles
    if st.session_state.chat_history:
        st.markdown("<h3 class='subheader'>💬 Conversation</h3>", unsafe_allow_html=True)
        for speaker, message in st.session_state.chat_history[-10:]:  # Show last 10 messages
            if speaker == "You":
                st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'>{message}</div>", unsafe_allow_html=True)
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()

# KPI Forecasting
elif st.session_state.current_page == "Predictor":
    st.markdown("<h1 class='header'>📊 KPI Forecasting</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Upload your KPI data to predict future trends and plan accordingly.</p>", unsafe_allow_html=True)
    
    # Sample CSV download with better presentation
    st.markdown("<h4>📊 Sample Data Template</h4>", unsafe_allow_html=True)
    sample_df = generate_sample_csv()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(sample_df, use_container_width=True)
    with col2:
        st.markdown(get_csv_download_link(sample_df), unsafe_allow_html=True)
        st.info("💡 Download this template to see the expected format")
    
    st.markdown("<h4>📤 Upload Your Data</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file with columns: Month, Usage", type=["csv"])
    
    if uploaded_file is not None:
        try:
            with st.spinner("📊 Processing your data..."):
                df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['Month', 'Usage']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("💡 Your CSV must have columns named 'Month' and 'Usage'")
                st.info("📋 Available columns in your file: " + ", ".join(df.columns.tolist()))
            else:
                st.success("✅ Data loaded successfully!")
                
                # Display data with insights
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h4>📈 Your Data</h4>", unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                with col2:
                    st.markdown("<h4>📊 Quick Stats</h4>", unsafe_allow_html=True)
                    st.metric("Total Records", len(df))
                    st.metric("Average Usage", f"{df['Usage'].mean():.1f}")
                    st.metric("Peak Usage", f"{df['Usage'].max():.1f}")
                
                # Simple linear regression forecast
                if len(df) >= 3:  # Need at least 3 data points for meaningful forecast
                    # Convert months to numbers for regression
                    month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                   'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                    
                    # Handle month mapping
                    df['MonthNum'] = df['Month'].map(month_mapping)
                    if df['MonthNum'].isna().any():
                        st.warning("⚠️ Some months couldn't be mapped. Using sequential numbering.")
                        df['MonthNum'] = range(1, len(df) + 1)
                    
                    # Fit model
                    X = df[['MonthNum']].values
                    y = df['Usage'].values
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Calculate R-squared
                    r_squared = model.score(X, y)
                    
                    # Predict next 6 months
                    future_months = np.array([[df['MonthNum'].max() + i] for i in range(1, 7)])
                    predictions = model.predict(future_months)
                    
                    # Create forecast dataframe
                    future_month_names = ['Month +1', 'Month +2', 'Month +3', 'Month +4', 'Month +5', 'Month +6']
                    forecast_df = pd.DataFrame({
                        'Period': future_month_names,
                        'Predicted Usage': predictions.round(1)
                    })
                    
                    st.markdown("<h4>🔮 6-Month Forecast</h4>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.dataframe(forecast_df, use_container_width=True)
                    with col2:
                        st.metric("Model Accuracy", f"{r_squared:.2%}")
                        if r_squared > 0.7:
                            st.success("✅ High accuracy")
                        elif r_squared > 0.4:
                            st.warning("⚠️ Moderate accuracy")
                        else:
                            st.error("❌ Low accuracy")
                    
                    # Enhanced plot
                    fig = px.line(df, x='Month', y='Usage', title='📈 Historical Data vs Forecast')
                    fig.add_scatter(x=future_month_names, y=predictions, 
                                  mode='lines+markers', name='Forecast', 
                                  line=dict(dash='dash', color='red'))
                    fig.update_layout(
                        xaxis_title="Time Period",
                        yaxis_title="Usage",
                        hovermode='closest'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend analysis
                    trend = "increasing" if model.coef_[0] > 0 else "decreasing"
                    st.info(f"📊 Trend Analysis: Your usage is {trend} by {abs(model.coef_[0]):.1f} units per month on average.")
                    
                else:
                    st.error("❌ Need at least 3 data points for forecasting")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please ensure your CSV file is properly formatted")
    else:
        st.info("📤 Upload a CSV file to get started with forecasting")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Document Summarizer
elif st.session_state.current_page == "Summarizer":
    st.markdown("<h1 class='header'>📄 Document Summarizer</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Upload policy documents or paste text to get AI-powered summaries.</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])
    
    document_text = None
    
    with tab1:
        uploaded_pdf = st.file_uploader("📄 Upload a PDF document", type=["pdf"])
        if uploaded_pdf is not None:
            with st.spinner("📖 Extracting text from PDF..."):
                document_text = extract_text_from_pdf(uploaded_pdf)
            if document_text:
                st.success("✅ PDF text extracted successfully!")
                st.balloons()
                with st.expander("📋 View extracted text (preview)"):
                    st.text_area("Extracted Text:", document_text[:1000] + "..." if len(document_text) > 1000 else document_text, height=200)
            else:
                st.error("❌ Failed to extract text from PDF. Please try a different file.")
    
    with tab2:
        document_text = st.text_area("📝 Paste your text here:", 
                                   height=300,
                                   placeholder="Paste your document text, policy, or any content you want summarized...")
        if document_text:
            st.info(f"📊 Document length: {len(document_text)} characters")
    
    if document_text:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("✨ Generate Summary", key="summarize_button", use_container_width=True):
                with st.spinner("🧠 Analyzing document and generating summary..."):
                    summary_prompt = f"Summarize the following document in 3-5 key points:\n\n{document_text[:2000]}"
                    summary = generate_text(summary_prompt)
                    st.markdown("<h3 class='subheader'>📋 Document Summary</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='green-card'>{summary}</div>", unsafe_allow_html=True)
                    st.success("✅ Summary generated successfully!")
                    st.balloons()
    else:
        st.info("📝 Please upload a PDF or paste some text to get started with summarization.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Policy Search
elif st.session_state.current_page == "PolicySearch":
    st.markdown("<h1 class='header'>🔍 Policy Search</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Search through city policies and regulations using keywords.</p>", unsafe_allow_html=True)
    
    search_query = st.text_input("🔍 Enter search keywords:", 
                                placeholder="e.g., 'energy', 'transportation', 'waste', 'environment'")
    
    if search_query:
        with st.spinner("🔍 Searching policies..."):
            results = search_policies(search_query, st.session_state.policies)
        
        if results:
            st.success(f"✅ Found {len(results)} matching policy(ies)")
            st.markdown(f"<h3 class='subheader'>🎯 Search Results for '{search_query}'</h3>", unsafe_allow_html=True)
            for i, policy in enumerate(results):
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<h4>📋 {policy['title']}</h4>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**📂 Category:** {policy['category']}")
                    st.markdown(f"**📅 Date:** {policy['date']}")
                with col2:
                    if st.button(f"🤖 Get AI Summary", key=f"summary_{hash(policy['title'])}", use_container_width=True):
                        with st.spinner("🧠 Generating AI summary..."):
                            summary = generate_text(f"Summarize this policy in 2-3 sentences: {policy['content']}")
                            st.markdown(f"<div class='alert-card'><b>🤖 AI Summary:</b><br>{summary}</div>", unsafe_allow_html=True)
                
                with st.expander("📖 View Full Policy Content"):
                    st.markdown(policy['content'])
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("🚫 No policies found matching your search query. Try different keywords!")
            st.info("💡 Try searching for: 'energy', 'transportation', 'waste', 'environment', 'smart', 'city'")
    else:
        st.info("💡 Enter keywords above to search through city policies and regulations.")
    
    # Show all policies
    if st.checkbox("Show all available policies"):
        st.markdown("<h3 class='subheader'>All Policies</h3>", unsafe_allow_html=True)
        for policy in st.session_state.policies:
            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h4>{policy['title']}</h4>", unsafe_allow_html=True)
            st.markdown(f"<b>Category:</b> {policy['category']}<br>", unsafe_allow_html=True)
            st.markdown(f"<b>Date:</b> {policy['date']}<br><br>", unsafe_allow_html=True)
            st.markdown(f"{policy['content']}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Citizen Feedback
elif st.session_state.current_page == "Feedback":
    st.markdown("<h1 class='header'>📝 Citizen Feedback System</h1>", unsafe_allow_html=True)
    
    # Feedback submission form
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Report an Issue</h3>", unsafe_allow_html=True)
    
    with st.form("feedback_form"):
        col1, col2 = st.columns(2)
        with col1:
            issue_type = st.selectbox("🏷️ Issue Type", ["Infrastructure", "Environment", "Transportation", "Utilities", "Other"])
            priority = st.selectbox("⚡ Priority Level", ["Low", "Medium", "High", "Critical"])
        with col2:
            location = st.text_input("📍 Location/Area", placeholder="e.g., Main Street, Downtown, Sector 12")
            contact_info = st.text_input("📞 Contact (optional)", placeholder="Your email or phone number")
        
        description = st.text_area("📝 Issue Description", 
                                 height=150,
                                 placeholder="Please describe the issue in detail. Include any relevant information that would help us address it.")
        
        submit_feedback = st.form_submit_button("🚀 Submit Feedback", use_container_width=True)
        
        if submit_feedback:
            if description and location:
                feedback_id = len(st.session_state.citizen_feedback) + 1
                new_feedback = {
                    'id': feedback_id,
                    'type': issue_type,
                    'location': location,
                    'priority': priority,
                    'description': description,
                    'contact': contact_info,
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'status': 'Submitted'
                }
                st.session_state.citizen_feedback.append(new_feedback)
                st.success("✅ Your feedback has been submitted successfully!")
                st.balloons()
                st.info(f"🎫 Your ticket number is: #{feedback_id}")
            else:
                st.error("❌ Please provide both location and description.")
                if not location:
                    st.warning("⚠️ Location is required")
                if not description:
                    st.warning("⚠️ Description is required")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display existing feedback with improved UI
    if st.session_state.citizen_feedback:
        st.markdown(f"<h3 class='subheader'>📋 Submitted Feedback ({len(st.session_state.citizen_feedback)} total)</h3>", unsafe_allow_html=True)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Filter by Status:", ["All", "Submitted", "In Progress", "Resolved", "Closed"])
        with col2:
            priority_filter = st.selectbox("Filter by Priority:", ["All", "Low", "Medium", "High", "Critical"])
        
        # Apply filters
        filtered_feedback = st.session_state.citizen_feedback.copy()
        if status_filter != "All":
            filtered_feedback = [f for f in filtered_feedback if f['status'] == status_filter]
        if priority_filter != "All":
            filtered_feedback = [f for f in filtered_feedback if f['priority'] == priority_filter]
        
        if filtered_feedback:
            for feedback in filtered_feedback:
                priority_color = {"Low": "#28a745", "Medium": "#ffc107", "High": "#fd7e14", "Critical": "#dc3545"}
                status_emoji = {"Submitted": "📋", "In Progress": "⏳", "Resolved": "✅", "Closed": "🔒"}
                
                st.markdown(f"<div class='feedback-item'>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"<h4>🎫 #{feedback['id']} - {feedback['type']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"**📍 Location:** {feedback['location']}")
                    st.markdown(f"**⚡ Priority:** <span style='color: {priority_color[feedback['priority']]}'>{feedback['priority']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**📊 Status:** {status_emoji.get(feedback['status'], '📋')} {feedback['status']}")
                    st.markdown(f"**📅 Date:** {feedback['date']}")
                    
                with col2:
                    new_status = st.selectbox(f"Update Status", 
                                            ["Submitted", "In Progress", "Resolved", "Closed"],
                                            index=["Submitted", "In Progress", "Resolved", "Closed"].index(feedback['status']),
                                            key=f"status_{feedback['id']}")
                    if st.button(f"💾 Update", key=f"update_{feedback['id']}", use_container_width=True):
                        feedback['status'] = new_status
                        st.success(f"✅ Status updated to {new_status}")
                        st.rerun()
                
                with st.expander("📖 View Full Description"):
                    st.markdown(f"**Issue Description:**\n{feedback['description']}")
                    if feedback['contact']:
                        st.markdown(f"**Contact Info:** {feedback['contact']}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("🔍 No feedback matches the selected filters.")
    else:
        st.info("📝 No feedback has been submitted yet. Be the first to report an issue!")

# Anomaly Detection
elif st.session_state.current_page == "Anomaly":
    st.markdown("<h1 class='header'>⚠️ Anomaly Detection</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Monitor city KPIs for unusual patterns and anomalies.</p>", unsafe_allow_html=True)
    
    # Select KPI to analyze
    kpi_options = ['Energy_Consumption', 'Water_Usage', 'Waste_Generation', 'Air_Quality_Index', 'Traffic_Density']
    selected_kpi = st.selectbox("Select KPI to analyze:", kpi_options)
    
    # Detect anomalies
    anomalies = detect_anomalies(st.session_state.city_kpis, selected_kpi, threshold=1.5)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.line(st.session_state.city_kpis, x='Date', y=selected_kpi, 
                     title=f'{selected_kpi} Over Time')
        
        # Highlight anomalies
        if not anomalies.empty:
            fig.add_scatter(x=anomalies['Date'], y=anomalies[selected_kpi], 
                           mode='markers', marker=dict(color='red', size=10),
                           name='Anomalies')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"<h4>Anomaly Summary</h4>", unsafe_allow_html=True)
        if not anomalies.empty:
            st.markdown(f"<div class='danger-card'>", unsafe_allow_html=True)
            st.markdown(f"<b>Anomalies Found:</b> {len(anomalies)}<br>", unsafe_allow_html=True)
            for _, anomaly in anomalies.iterrows():
                st.markdown(f"• {anomaly['Date'].strftime('%B %Y')}: {anomaly[selected_kpi]:.1f}<br>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("Get AI Analysis", key="anomaly_analysis"):
                with st.spinner("Analyzing anomalies..."):
                    analysis_prompt = f"Analyze these anomalies in {selected_kpi}: {anomalies[selected_kpi].tolist()}"
                    analysis = generate_text(analysis_prompt)
                    st.markdown(f"<div class='alert-card'><b>AI Analysis:</b><br>{analysis}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='green-card'>", unsafe_allow_html=True)
            st.markdown("<b>No anomalies detected</b><br>All values are within normal range.", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Eco Tips Generator
elif st.session_state.current_page == "EcoTips":
    st.markdown("<h1 class='header'>🌱 Eco Tips Generator</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Get personalized eco-friendly tips based on keywords and topics.</p>", unsafe_allow_html=True)
    
    # Quick tip buttons
    st.markdown("<h3 class='subheader'>Quick Tips</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔌 Energy Saving", key="eco_energy"):
            with st.spinner("Generating energy tips..."):
                tips = generate_text("Give 5 practical energy-saving tips for home:")
                st.markdown(f"<div class='green-card'><b>Energy Saving Tips:</b><br>{tips}</div>", unsafe_allow_html=True)
    
    with col2:
        if st.button("💧 Water Conservation", key="eco_water"):
            with st.spinner("Generating water tips..."):
                tips = generate_text("Give 5 practical water conservation tips for daily life:")
                st.markdown(f"<div class='green-card'><b>Water Conservation Tips:</b><br>{tips}</div>", unsafe_allow_html=True)
    
    with col3:
        if st.button("♻️ Waste Reduction", key="eco_waste"):
            with st.spinner("Generating waste tips..."):
                tips = generate_text("Give 5 practical waste reduction tips for households:")
                st.markdown(f"<div class='green-card'><b>Waste Reduction Tips:</b><br>{tips}</div>", unsafe_allow_html=True)
    
    with col4:
        if st.button("🚗 Transportation", key="eco_transport"):
            with st.spinner("Generating transport tips..."):
                tips = generate_text("Give 5 sustainable transportation tips for city dwellers:")
                st.markdown(f"<div class='green-card'><b>Transportation Tips:</b><br>{tips}</div>", unsafe_allow_html=True)
    
    # Custom keyword search
    st.markdown("<h3 class='subheader'>Custom Eco Tips</h3>", unsafe_allow_html=True)
    
    keywords = st.text_input("Enter keywords (e.g., 'plastic', 'solar', 'garden', 'office'):")
    
    if keywords:
        if st.button("Generate Custom Tips", key="custom_eco_tips"):
            with st.spinner("Generating custom tips..."):
                custom_tips = generate_text(f"Give eco-friendly tips related to: {keywords}")
                st.markdown(f"<div class='green-card'><b>Custom Tips for '{keywords}':</b><br>{custom_tips}</div>", unsafe_allow_html=True)
    
    # Daily eco challenge
    st.markdown("<h3 class='subheader'>Daily Eco Challenge</h3>", unsafe_allow_html=True)
    
    challenges = [
        "🌱 Use a reusable water bottle for the entire day",
        "🚶 Walk or bike for at least one trip instead of driving",
        "💡 Unplug all electronics when not in use",
        "🥗 Have one plant-based meal today",
        "📱 Use digital receipts instead of paper ones",
        "🌡️ Set your thermostat 2 degrees higher/lower to save energy",
        "♻️ Find 3 items to recycle that you might normally throw away"
    ]
    daily_challenge = random.choice(challenges)
    
    st.markdown("<div class='alert-card'>", unsafe_allow_html=True)
    st.markdown("**Today's Eco Challenge:**")
    st.markdown(daily_challenge)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Personal Recommendations
elif st.session_state.current_page == "Recommendations":
    st.markdown("<h1 class='header'>🧍 Personal Sustainability Recommendations</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p>Get personalized sustainability recommendations based on your lifestyle.</p>", unsafe_allow_html=True)
    
    with st.form("recommendation_form"):
        st.markdown("<h3 class='subheader'>Tell us about your lifestyle</h3>", unsafe_allow_html=True)
        
        # Lifestyle questions
        home_type = st.selectbox("Type of residence:", ["Apartment", "House", "Condo", "Other"])
        family_size = st.selectbox("Family size:", ["1 person", "2 people", "3-4 people", "5+ people"])
        commute_method = st.selectbox("Primary commute method:", ["Car", "Public Transport", "Bike", "Walk", "Work from Home"])
        energy_usage = st.selectbox("Monthly energy bill:", ["Under $50", "$50-100", "$100-200", "Over $200"])
        eco_interest = st.selectbox("Interest in eco-friendly practices:", ["Very interested", "Somewhat interested", "Not very interested"])
        
        current_practices = st.multiselect(
            "Current sustainable practices:",
            ["Recycling", "Composting", "Using LED bulbs", "Solar panels", "Electric vehicle", 
             "Water-saving fixtures", "Organic food", "Public transport", "Bike commuting"]
        )
        
        improvement_areas = st.multiselect(
            "Areas you'd like to improve:",
            ["Energy consumption", "Water usage", "Waste reduction", "Transportation", 
             "Food choices", "Shopping habits", "Home efficiency"]
        )
        
        submit_recommendations = st.form_submit_button("Get My Personalized Plan")
    
    if submit_recommendations:
        with st.spinner("Creating your personalized sustainability plan..."):
            # Create a comprehensive prompt based on user inputs
            profile_prompt = f"""
            Create a personalized sustainability plan for someone with this profile:
            - Lives in: {home_type}
            - Family size: {family_size}
            - Commutes by: {commute_method}
            - Energy bill: {energy_usage}
            - Eco interest level: {eco_interest}
            - Current practices: {', '.join(current_practices)}
            - Wants to improve: {', '.join(improvement_areas)}
            
            Provide specific, actionable recommendations organized by category.
            """
            
            recommendations = generate_text(profile_prompt, max_new_tokens=800)
            
            st.markdown("<h3 class='subheader'>Your Personalized Sustainability Plan</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='green-card'>{recommendations}</div>", unsafe_allow_html=True)
            
            # Additional recommendations based on improvement areas
            if improvement_areas:
                st.markdown("<h4>Specific Action Items</h4>", unsafe_allow_html=True)
                for area in improvement_areas:
                    area_tips = generate_text(f"Give 3 specific tips for improving {area.lower()} at home:")
                    st.markdown(f"<div class='card'><b>{area}:</b><br>{area_tips}</div>", unsafe_allow_html=True)
            
            if st.button("Save My Plan", key="save_plan"):
                st.success("✅ Your sustainability plan has been saved!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 0.8em;'>🏙️ Smart City Assistant | Built with IBM Granite LLM & Streamlit | Sustainable Urban Living Platform</p>", unsafe_allow_html=True)
