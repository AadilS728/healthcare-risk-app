import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Page configuration
st.set_page_config(
    page_title="Healthcare Risk Stratification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .risk-high {
        color: #ef4444;
        font-weight: bold;
        font-size: 2rem;
    }
    .risk-medium {
        color: #f59e0b;
        font-weight: bold;
        font-size: 2rem;
    }
    .risk-low {
        color: #10b981;
        font-weight: bold;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Risk Prediction Model
class HealthcareRiskModel:
    def __init__(self):
        # Logistic regression coefficients
        self.intercept = -2.5
        self.age_coef = 1.8
        self.stay_coef = 2.2
        self.cost_coef = 1.5
    
    def normalize_features(self, age, stay, cost):
        """Normalize input features"""
        age_norm = (age - 40) / 40
        stay_norm = (stay - 5) / 15
        cost_norm = (cost - 2000) / 8000
        return age_norm, stay_norm, cost_norm
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def predict(self, age, stay, cost):
        """Predict risk probability"""
        age_norm, stay_norm, cost_norm = self.normalize_features(age, stay, cost)
        
        # Calculate log odds
        log_odds = (self.intercept + 
                   self.age_coef * age_norm + 
                   self.stay_coef * stay_norm + 
                   self.cost_coef * cost_norm)
        
        # Convert to probability
        probability = self.sigmoid(log_odds)
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High Risk"
            risk_color = "#ef4444"
            risk_class = "risk-high"
        elif probability > 0.4:
            risk_level = "Medium Risk"
            risk_color = "#f59e0b"
            risk_class = "risk-medium"
        else:
            risk_level = "Low Risk"
            risk_color = "#10b981"
            risk_class = "risk-low"
        
        # Calculate risk factors contribution
        factors = {
            'age': ((age - 40) / 60 * 100),
            'stay': ((stay - 1) / 30 * 100),
            'cost': ((cost - 1000) / 9000 * 100)
        }
        
        return {
            'level': risk_level,
            'probability': probability,
            'color': risk_color,
            'class': risk_class,
            'factors': factors
        }

# Generate synthetic historical data
@st.cache_data
def generate_historical_data(n_samples=100):
    """Generate synthetic patient data for analysis"""
    np.random.seed(42)
    model = HealthcareRiskModel()
    
    data = []
    for _ in range(n_samples):
        age = random.randint(30, 90)
        stay = random.randint(1, 25)
        cost = random.randint(1000, 10000)
        
        prediction = model.predict(age, stay, cost)
        
        data.append({
            'Age': age,
            'Length_of_Stay': stay,
            'Treatment_Cost': cost,
            'Risk_Probability': prediction['probability'] * 100,
            'Risk_Category': prediction['level']
        })
    
    return pd.DataFrame(data)

# Get clinical recommendations
def get_recommendations(risk_level):
    """Return clinical recommendations based on risk level"""
    recommendations = {
        "High Risk": [
            "🚨 Immediate medical attention required",
            "📊 Daily monitoring and assessment",
            "🏥 Consider specialized care unit transfer",
            "⚕️ Implement preventive intervention protocols"
        ],
        "Medium Risk": [
            "👁️ Regular monitoring recommended",
            "📅 Schedule follow-up assessments",
            "📋 Review treatment plan effectiveness",
            "⚠️ Monitor for symptom escalation"
        ],
        "Low Risk": [
            "✅ Standard care protocols apply",
            "🔄 Routine monitoring sufficient",
            "📝 Continue current treatment plan",
            "🗓️ Schedule regular check-ups"
        ]
    }
    return recommendations.get(risk_level, [])

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🏥 Healthcare Risk Stratification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Patient Risk Analysis & Prediction</div>', unsafe_allow_html=True)
    
    # Initialize model
    model = HealthcareRiskModel()
    
    # Sidebar - Input Parameters
    st.sidebar.header("📋 Patient Data Input")
    
    age = st.sidebar.slider("Age", min_value=0, max_value=100, value=65, step=1)
    length_of_stay = st.sidebar.slider("Length of Stay (days)", min_value=1, max_value=30, value=7, step=1)
    treatment_cost = st.sidebar.slider("Treatment Cost (Rs.)", min_value=100, max_value=100000, value=3000, step=100)
    
    predict_button = st.sidebar.button("🔍 Predict Risk", type="primary", use_container_width=True)
    show_analytics = st.sidebar.checkbox("📊 Show Analytics Dashboard", value=False)
    
    # Prediction Section
    if predict_button or 'prediction' not in st.session_state:
        prediction = model.predict(age, length_of_stay, treatment_cost)
        st.session_state['prediction'] = prediction
    
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        
        # Display Key Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👤 Patient Age", f"{age} years")
        with col2:
            st.metric("📅 Length of Stay", f"{length_of_stay} days")
        with col3:
            st.metric("💰 Treatment Cost", f"Rs.{treatment_cost:,}")
        
        st.markdown("---")
        
        # Risk Prediction Results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Risk Assessment")
            st.markdown(f'<div class="{prediction["class"]}">{prediction["level"]}</div>', unsafe_allow_html=True)
            st.markdown(f"**Risk Probability:** {prediction['probability']*100:.1f}%")
            
            # Risk probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction['probability']*100,
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': prediction['color']},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.subheader("📊 Risk Factor Contribution")
            
            # Risk factors breakdown
            factors_df = pd.DataFrame({
                'Factor': ['Age Factor', 'Stay Duration', 'Treatment Cost'],
                'Contribution': [
                    max(0, min(100, prediction['factors']['age'])),
                    max(0, min(100, prediction['factors']['stay'])),
                    max(0, min(100, prediction['factors']['cost']))
                ]
            })
            
            fig_factors = px.bar(
                factors_df,
                x='Contribution',
                y='Factor',
                orientation='h',
                color='Factor',
                color_discrete_sequence=['#3b82f6', '#8b5cf6', '#10b981']
            )
            fig_factors.update_layout(
                showlegend=False,
                height=300,
                xaxis_title="Contribution (%)",
                yaxis_title=""
            )
            st.plotly_chart(fig_factors, use_container_width=True)
        
        # Clinical Recommendations
        st.markdown("---")
        st.subheader("💊 Clinical Recommendations")
        recommendations = get_recommendations(prediction['level'])
        
        cols = st.columns(2)
        for idx, rec in enumerate(recommendations):
            with cols[idx % 2]:
                st.info(rec)
    
    # Analytics Dashboard
    if show_analytics:
        st.markdown("---")
        st.header("📈 Analytics Dashboard")
        
        # Generate historical data
        df = generate_historical_data(100)
        
        # Create tabs for different analytics
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Distributions", "🔗 Correlations", "📋 Data Table"])
        
        with tab1:
            # Summary Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Patients", len(df))
            with col2:
                high_risk = len(df[df['Risk_Category'] == 'High Risk'])
                st.metric("High Risk", high_risk, delta=f"{high_risk/len(df)*100:.1f}%")
            with col3:
                avg_age = df['Age'].mean()
                st.metric("Avg Age", f"{avg_age:.1f} yrs")
            with col4:
                avg_cost = df['Treatment_Cost'].mean()
                st.metric("Avg Cost", f"Rs.{avg_cost:,.0f}")
            
            # Risk distribution pie chart
            st.subheader("Risk Category Distribution")
            risk_counts = df['Risk_Category'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={
                    'Low Risk': '#10b981',
                    'Medium Risk': '#f59e0b',
                    'High Risk': '#ef4444'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                st.subheader("Age Distribution")
                fig_age = px.histogram(
                    df,
                    x='Age',
                    nbins=0,
                    color_discrete_sequence=['#6366f1']
                )
                fig_age.update_layout(
                    xaxis_title="Age (years)",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                # Length of stay distribution
                st.subheader("Length of Stay Distribution")
                fig_stay = px.histogram(
                    df,
                    x='Length_of_Stay',
                    nbins=0,
                    color_discrete_sequence=['#8b5cf6']
                )
                fig_stay.update_layout(
                    xaxis_title="Days",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_stay, use_container_width=True)
            
            # Treatment cost distribution
            st.subheader("Treatment Cost Distribution")
            # load your CSV File

            df = pd.read_csv(r"C:\Users\91855\Desktop\DATA ANALYSTICS\Healthcare Risk Stratification App PROJECT\Healthcare_risk_data.csv")

            # Remove Unwanted Unnamed Column if exists
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # DIsplay cleaned data 
            st.dataframe(df) 
               
            fig_cost = px.histogram(
                df,
                x='Treatment_Cost',
                nbins=20,
                color_discrete_sequence=['#10b981']
            )
            fig_cost.update_layout(
                xaxis_title="Cost (Rs.)",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with tab3:
            # Correlation heatmap
            st.subheader("Feature Correlations")
            corr_matrix = df[['Age', 'Length_of_Stay', 'Treatment_Cost', 'Risk_Probability']].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Scatter plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Age vs Risk Probability")
                fig_scatter1 = px.scatter(
                    df,
                    x='Age',
                    y='Risk_Probability',
                    color='Risk_Category',
                    color_discrete_map={
                        'Low Risk': '#10b981',
                        'Medium Risk': '#f59e0b',
                        'High Risk': '#ef4444'
                    }
                )
                st.plotly_chart(fig_scatter1, use_container_width=True)
            
            with col2:
                st.subheader("Cost vs Risk Probability")
                fig_scatter2 = px.scatter(
                    df,
                    x='Treatment_Cost',
                    y='Risk_Probability',
                    color='Risk_Category',
                    color_discrete_map={
                        'Low Risk': '#10b981',
                        'Medium Risk': '#f59e0b',
                        'High Risk': '#ef4444'
                    }
                )
                st.plotly_chart(fig_scatter2, use_container_width=True)
        
        with tab4:
            st.subheader("Patient Data Table")
            st.dataframe(
                df.style.background_gradient(subset=['Risk_Probability'], cmap='RdYlGn_r'),
                use_container_width=True,
                height=400
            )
            
            # Download button
            file_path = r"C:\Users\91855\Desktop\DATA ANALYSTICS\Healthcare Risk Stratification App PROJECT\Healthcare_risk_data.csv"
            with open (file_path, "r",encoding="utf-8") as f:
                csv_data = f.read()
    
        
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv_data,
                file_name="Healthcare_risk_data.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 2rem;'>
        <p><strong>Model Information</strong></p>
        <p>Algorithm: Logistic Regression | Features: Age, Length of Stay, Treatment Cost | Accuracy: 87.3% (Simulated)</p>
        <p>Developed for Healthcare Risk Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

