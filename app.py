import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend for Matplotlib

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


app = Flask(__name__)

# Load model, selected features, and scaler
model = joblib.load('alz_reduced_model.pkl')  # Adjust path if needed
feature_names = joblib.load('alz_selected_features.pkl')
scaler = joblib.load('alz_scaler.pkl')

# Load the dataset
df = pd.read_csv('alzheimers_disease_data.csv')  # relative path now

# Split the dataset into disease and non-disease patients
df_disease = df[df['Diagnosis'] == 1]  # Alzheimer's disease patients
df_no_disease = df[df['Diagnosis'] == 0]  # Healthy patients

# Visualizations function
def create_visualizations(df_subset, group_type):
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    df_subset = df_subset.fillna(df_subset.median())

    def style_plot(fig):
        fig.update_traces(marker=dict(color='#FFA500', line=dict(width=1.2, color='black')))
        fig.update_layout(title=None)
        return fig
    
    def uniform_pie_style(fig):
        fig.update_layout(
            height=400,
            width=400,
            margin=dict(t=0, b=0, l=0, r=0),
            plot_bgcolor='#001F3F',   # Navy Blue plot area
            paper_bgcolor='#001F3F',  # Navy Blue full chart background
            font_color='white',
            showlegend=True  # Optional: set to False if you want minimal look
        )
        return fig
    
    def apply_global_chart_style(fig, is_pie=False):
        layout_config = {
            "height": 400,
            "width": 400 if is_pie else None,
            "margin": dict(t=0, b=0, l=0, r=0),
            "plot_bgcolor": "#001F3F",
            "paper_bgcolor": "#001F3F",
            "font_color": "white",
            "showlegend": True if is_pie else None
        }
        fig.update_layout(**{k: v for k, v in layout_config.items() if v is not None})
        return fig



    # Gender Distribution (Pie)
    gender_distribution = df_subset['Gender'].value_counts(normalize=True) * 100
    gender_data = [gender_distribution.get(1, 0), gender_distribution.get(0, 0)]
    gender_labels = ['Male', 'Female']
    gender_fig = px.pie(names=gender_labels, values=gender_data, color_discrete_sequence=["#9B59B6", "#F1C40F"])
    gender_fig = uniform_pie_style(gender_fig)
    gender_chart = pio.to_html(gender_fig, full_html=False)

    # MMSE Distribution
    mmse_fig = px.histogram(df_subset, x="MMSE", nbins=20, opacity=0.9, color_discrete_sequence=["#FFA500"])
    mmse_fig.update_traces(marker_line_color='black', marker_line_width=1.2)
    mmse_fig = apply_global_chart_style(mmse_fig)

    mmse_chart = pio.to_html(style_plot(mmse_fig), full_html=False)

    # Memory Complaints
    memory_complaints_distribution = df_subset['MemoryComplaints'].value_counts(normalize=True) * 100
    memory_complaints_data = [memory_complaints_distribution.get(1, 0), memory_complaints_distribution.get(0, 0)]
    memory_complaints_labels = ['Yes', 'No']
    memory_complaints_fig = px.bar(x=memory_complaints_labels, y=memory_complaints_data,
                                   labels={"x": "Complaints", "y": "Percentage"})
    memory_complaints_fig = apply_global_chart_style(memory_complaints_fig)
    memory_complaints_chart = pio.to_html(style_plot(memory_complaints_fig), full_html=False)

    # Behavioral Problems
    behavioral_problems_distribution = df_subset['BehavioralProblems'].value_counts(normalize=True) * 100
    behavioral_problems_data = [behavioral_problems_distribution.get(1, 0), behavioral_problems_distribution.get(0, 0)]
    behavioral_problems_labels = ['Yes', 'No']
    behavioral_problems_fig = px.bar(x=behavioral_problems_labels, y=behavioral_problems_data,
                                     labels={"x": "Problems", "y": "Percentage"})
    behavioral_problems_fig = apply_global_chart_style(behavioral_problems_fig)
    behavioral_problems_chart = pio.to_html(style_plot(behavioral_problems_fig), full_html=False)

    # HDL Cholesterol
    hdl_fig = px.histogram(df_subset, x="CholesterolHDL", nbins=20, opacity=0.9, color_discrete_sequence=["#3b82f6"])
    hdl_fig = apply_global_chart_style(hdl_fig)
    hdl_chart = pio.to_html(style_plot(hdl_fig), full_html=False)

    # Physical Activity
    physical_activity_fig = px.histogram(df_subset, x="PhysicalActivity", nbins=10,
                                         labels={"PhysicalActivity": "Activity Score", "count": "Patient Count"},
                                         opacity=0.9, color_discrete_sequence=["#3b82f6"])
    physical_activity_fig = apply_global_chart_style(physical_activity_fig)
    physical_activity_chart = pio.to_html(style_plot(physical_activity_fig), full_html=False)

    # Smoking (Pie)
    smoking_distribution = df_subset['Smoking'].value_counts(normalize=True) * 100
    smoking_data = [smoking_distribution.get(1, 0), smoking_distribution.get(0, 0)]
    smoking_labels = ['Smoker', 'Non-smoker']
    smoking_fig = px.pie(names=smoking_labels, values=smoking_data, color_discrete_sequence=["#2ECC71", "#FF4C4C"])
    smoking_fig = apply_global_chart_style(smoking_fig, is_pie=True)
    smoking_chart = pio.to_html(smoking_fig, full_html=False)

    # Family History (Pie)
    family_history_distribution = df_subset['FamilyHistoryAlzheimers'].value_counts(normalize=True) * 100
    family_history_data = [family_history_distribution.get(1, 0), family_history_distribution.get(0, 0)]
    family_history_labels = ['Yes', 'No']
    family_history_fig = px.pie(names=family_history_labels, values=family_history_data, color_discrete_sequence=["#FF4C4C", "#2ECC71"])
    family_history_fig = apply_global_chart_style(family_history_fig, is_pie=True)
    family_history_chart = pio.to_html(family_history_fig, full_html=False)

    # Cardiovascular Disease (Pie)
    cardiovascular_disease_distribution = df_subset['CardiovascularDisease'].value_counts(normalize=True) * 100
    cardiovascular_disease_data = [cardiovascular_disease_distribution.get(1, 0), cardiovascular_disease_distribution.get(0, 0)]
    cardiovascular_disease_labels = ['Yes', 'No']
    cardiovascular_disease_fig = px.pie(names=cardiovascular_disease_labels, values=cardiovascular_disease_data, color_discrete_sequence=["#FF4C4C", "#2ECC71"])
    cardiovascular_disease_fig = apply_global_chart_style(cardiovascular_disease_fig, is_pie=True)
    cardiovascular_disease_chart = pio.to_html(cardiovascular_disease_fig, full_html=False)

    # Diabetes (Pie)
    diabetes_distribution = df_subset['Diabetes'].value_counts(normalize=True) * 100
    diabetes_data = [diabetes_distribution.get(1, 0), diabetes_distribution.get(0, 0)]
    diabetes_labels = ['Yes', 'No']
    diabetes_fig = px.pie(names=diabetes_labels, values=diabetes_data, color_discrete_sequence=["#FF4C4C", "#2ECC71"])
    diabetes_fig = apply_global_chart_style(diabetes_fig, is_pie=True)
    diabetes_chart = pio.to_html(diabetes_fig, full_html=False)

    return {
        f'gender_chart_{group_type}': gender_chart,
        f'mmse_chart_{group_type}': mmse_chart,
        f'memory_complaints_chart_{group_type}': memory_complaints_chart,
        f'behavioral_problems_chart_{group_type}': behavioral_problems_chart,
        f'hdl_chart_{group_type}': hdl_chart,
        f'physical_activity_chart_{group_type}': physical_activity_chart,
        f'smoking_chart_{group_type}': smoking_chart,
        f'family_history_chart_{group_type}': family_history_chart,
        f'cardiovascular_disease_chart_{group_type}': cardiovascular_disease_chart,
        f'diabetes_chart_{group_type}': diabetes_chart
    }



def create_post_prediction_visuals(patient_input_scaled, prediction_proba, patient_input_raw):
    visuals = {}

    # 1. Prediction Pie Chart
    pie_fig = px.pie(
        values=[prediction_proba[0]*100, prediction_proba[1]*100],
        names=['Healthy', "Alzheimer's"],
        color_discrete_sequence=["#2ECC71", "#E74C3C"]
    )
    pie_fig.update_layout(
        paper_bgcolor="#001F3F",
        plot_bgcolor="#001F3F",
        font_color="white",
        height=400,
        width=400,
        margin=dict(t=0, b=0, l=0, r=0)
    )
    visuals['prediction_pie_chart'] = pio.to_html(pie_fig, full_html=False)


    # 2. Health Metric Comparison
    metrics = ['MMSE', 'CholesterolHDL', 'PhysicalActivity', 'FunctionalAssessment', 'ADL']
    comparison_data = {
        'Metric': [],
        'Patient': [],
        'AlzheimerAvg': [],
        'HealthyAvg': []
    }
    for metric in metrics:
        comparison_data['Metric'].append(metric)
        comparison_data['Patient'].append(patient_input_raw.get(metric, 0))
        comparison_data['AlzheimerAvg'].append(df_disease[metric].mean())
        comparison_data['HealthyAvg'].append(df_no_disease[metric].mean())

    comp_df = pd.DataFrame(comparison_data)
    comp_fig = px.bar(
        comp_df.melt(id_vars='Metric', var_name='Group', value_name='Value'),
        x='Metric', y='Value', color='Group', barmode='group'
    )
    comp_fig.update_layout(
        paper_bgcolor="#001F3F",
        plot_bgcolor="#001F3F",
        font_color="white",
        legend=dict(font_color="white"),
        margin=dict(t=10, b=10)
    )
    visuals['metric_comparison_chart'] = pio.to_html(comp_fig, full_html=False)



    # 3. Feature Contribution Bar Chart
    contrib_fig = px.bar(
        x=feature_names,
        y=patient_input_scaled.flatten(),
        labels={"x": "Feature", "y": "Standardized Value"},
        color_discrete_sequence=["#FFA500"]
    )
    contrib_fig.update_layout(
        paper_bgcolor="#001F3F",
        plot_bgcolor="#001F3F",
        font_color="white",
        margin=dict(t=10, b=10)
    )
    visuals['feature_contribution_chart'] = pio.to_html(contrib_fig, full_html=False)

    # 4. Radar Chart (Spider Plot)
    radar_features = ['MMSE', 'CholesterolHDL', 'PhysicalActivity', 'FunctionalAssessment', 'ADL']
    theta = radar_features + [radar_features[0]]
    patient_values = [patient_input_raw.get(f, 0) for f in radar_features] + [patient_input_raw.get(radar_features[0], 0)]
    disease_avg = [df_disease[f].mean() for f in radar_features] + [df_disease[radar_features[0]].mean()]
    healthy_avg = [df_no_disease[f].mean() for f in radar_features] + [df_no_disease[radar_features[0]].mean()]
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=patient_values, theta=theta, fill='toself', name='Patient'))
    radar_fig.add_trace(go.Scatterpolar(r=disease_avg, theta=theta, fill='toself', name='Alzheimer Avg'))
    radar_fig.add_trace(go.Scatterpolar(r=healthy_avg, theta=theta, fill='toself', name='Healthy Avg'))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                            paper_bgcolor='#001F3F', plot_bgcolor='#001F3F', font_color='white')
    visuals['radar_chart'] = pio.to_html(radar_fig, full_html=False)

    # 5. Gauge Chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba[1] * 100,
        title={'text': "Alzheimer's Risk (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#FF4C4C"},
               'steps': [
                   {'range': [0, 33], 'color': "#2ECC71"},
                   {'range': [33, 66], 'color': "#F1C40F"},
                   {'range': [66, 100], 'color': "#E74C3C"}]},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    gauge_fig.update_layout(paper_bgcolor='#001F3F', font_color='white')
    visuals['gauge_chart'] = pio.to_html(gauge_fig, full_html=False)

    return visuals

# ========================== ROUTES ==========================


# Home page route
@app.route('/')
def home():
    charts_disease = create_visualizations(df_disease, "disease")  # For Alzheimer's patients
    charts_no_disease = create_visualizations(df_no_disease, "no_disease")  # For healthy patients

    # Compute basic insights
    total_patients_disease = len(df_disease)
    avg_age_disease = df_disease['Age'].mean()

    total_patients_no_disease = len(df_no_disease)
    avg_age_no_disease = df_no_disease['Age'].mean()

    return render_template('index.html', 
                           total_patients_disease=total_patients_disease,
                           avg_age_disease=avg_age_disease,
                           total_patients_no_disease=total_patients_no_disease,
                           avg_age_no_disease=avg_age_no_disease,
                           **charts_disease, **charts_no_disease)

# Prediction page route
# Prediction page route with charts
# Prediction page route without charts
@app.route('/predict')
def predict():
    return render_template('predict.html', feature_names=feature_names)

@app.route('/predict-ui', methods=['POST'])
def predict_ui():
    try:
        features_input = []
        for feat in feature_names:
            val = request.form.get(feat)
            if val is None or val.strip() == "":
                raise ValueError(f"Missing value for: {feat}")
            features_input.append(float(val))
        features_array = np.array(features_input).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = round(proba[prediction] * 100, 2)
        result = "Possibility for Alzheimer's" if prediction == 1 else "Healthy"
        patient_dict = dict(zip(feature_names, features_input))
        visualizations = create_post_prediction_visuals(features_scaled, proba, patient_dict)

        return render_template('predict.html',
                               prediction=result,
                               confidence=confidence,
                               feature_names=feature_names,
                               **visualizations)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)


