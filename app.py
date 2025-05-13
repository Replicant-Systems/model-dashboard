import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Set page configuration
st.set_page_config(
    page_title="NTRM Object Detection Models Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default data - used when initializing the app for the first time
default_model_data = {
    "name": ["NTRM-Stem", "NTRM-Lamina", "NTRM-0.6ft", "NTRM-2_cams", "NTRM-yolo_v11n"],
    "map50": [0.86, 0.82, 0.89, 0.84, 0.81],
    "map50_95": [0.72, 0.68, 0.76, 0.70, 0.65],
    "iou": [0.78, 0.75, 0.81, 0.77, 0.73],
    "training_time": ["5h 20m", "8h 15m", "6h 45m", "7h 30m", "4h 50m"],
    "last_updated": ["2025-05-08", "2025-05-03", "2025-05-10", "2025-04-28", "2025-05-12"],
    "status": ["Production", "Production", "Production", "Testing", "Development"],
    "model_file": ["", "", "", "", ""]
}

# Initialize session state for persistent storage
if 'model_data' not in st.session_state:
    st.session_state.model_data = pd.DataFrame(default_model_data)

if 'model_files' not in st.session_state:
    st.session_state.model_files = {}

# Generate default historical data
def generate_default_history_data():
    dates = pd.date_range(start='2025-04-01', end='2025-05-10', freq='W')
    models = ["NTRM-YOLOv5", "NTRM-FasterRCNN", "NTRM-EfficientDet"]
    
    # Create empty dataframe
    history_data = []
    
    # Generate some realistic trending data
    for model in models:
        base_map = np.random.uniform(0.75, 0.80)
        base_iou = np.random.uniform(0.70, 0.75)
        
        for i, date in enumerate(dates):
            # Gradually improve metrics over time with some randomness
            improvement = min(0.02 * i, 0.10)  # Cap improvement at 0.10
            random_factor = np.random.uniform(-0.01, 0.01)
            
            history_data.append({
                "date": date,
                "model": model,
                "metric": "mAP@0.5",
                "value": base_map + improvement + random_factor
            })
            history_data.append({
                "date": date,
                "model": model,
                "metric": "IoU",
                "value": base_iou + improvement + random_factor
            })
    
    return pd.DataFrame(history_data)

if 'history_data' not in st.session_state:
    st.session_state.history_data = generate_default_history_data()

# Function to save model file to session state
def save_model_file(model_name, uploaded_file):
    # Clean model name for filename
    safe_name = "".join([c if c.isalnum() else "_" for c in model_name])
    filename = f"{safe_name}.pt"
    
    # In Streamlit Cloud, we store the file content in session state
    st.session_state.model_files[filename] = uploaded_file.getvalue()
    
    return filename

# Function to delete a model
def delete_model(model_name):
    # Get the model file name if it exists
    df = st.session_state.model_data
    history_df = st.session_state.history_data
    
    model_file = ""
    model_row = df[df["name"] == model_name]
    if not model_row.empty and model_row.iloc[0]["model_file"]:
        model_file = model_row.iloc[0]["model_file"]
    
    # Remove model from DataFrame
    df = df[df["name"] != model_name]
    
    # Remove associated history data
    history_df = history_df[history_df["model"] != model_name]
    
    # Delete model file if it exists
    if model_file and model_file in st.session_state.model_files:
        del st.session_state.model_files[model_file]
    
    # Save updated data
    st.session_state.model_data = df
    st.session_state.history_data = history_df
    
    return df, history_df

# Function to rename a model
def rename_model(old_name, new_name):
    df = st.session_state.model_data
    history_df = st.session_state.history_data
    
    # Update model name in main DataFrame
    df.loc[df["name"] == old_name, "name"] = new_name
    
    # Update model name in history DataFrame
    history_df.loc[history_df["model"] == old_name, "model"] = new_name
    
    # Rename model file if it exists
    model_row = df[df["name"] == new_name]
    if not model_row.empty and model_row.iloc[0]["model_file"]:
        old_file = model_row.iloc[0]["model_file"]
        
        # Create new file name
        safe_name = "".join([c if c.isalnum() else "_" for c in new_name])
        new_file = f"{safe_name}.pt"
        
        # Update file name in DataFrame
        df.loc[df["name"] == new_name, "model_file"] = new_file
        
        # Rename actual file in session state
        if old_file in st.session_state.model_files:
            st.session_state.model_files[new_file] = st.session_state.model_files[old_file]
            del st.session_state.model_files[old_file]
    
    # Save updated data
    st.session_state.model_data = df
    st.session_state.history_data = history_df
    
    return df, history_df

# Load data from session state
df = st.session_state.model_data
history_df = st.session_state.history_data

# Main App Header
st.title("NTRM Object Detection Models Dashboard")
st.markdown("Monitor and compare performance metrics for Non-tobacco Related Materials (NTRM) detection models")

# Add tabs at the top level
main_tabs = st.tabs(["Dashboard", "Model Management"])

with main_tabs[0]:  # Dashboard tab
    # Sidebar for filtering
    st.sidebar.header("Filters")
    status_filter = st.sidebar.multiselect(
        "Filter by Status",
        options=df["status"].unique(),
        default=df["status"].unique()
    )
    
    # Apply filters
    filtered_df = df[df["status"].isin(status_filter)]
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Models", len(filtered_df))
    
    with col2:
        if not filtered_df.empty:
            best_map_model = filtered_df.loc[filtered_df["map50"].idxmax()]
            st.metric("Best mAP@0.5", f"{best_map_model['map50']:.2f}", f"{best_map_model['name']}")
        else:
            st.metric("Best mAP@0.5", "N/A", "No models")
    
    with col3:
        if not filtered_df.empty:
            best_iou_model = filtered_df.loc[filtered_df["iou"].idxmax()]
            st.metric("Best IoU", f"{best_iou_model['iou']:.2f}", f"{best_iou_model['name']}")
        else:
            st.metric("Best IoU", "N/A", "No models")
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Overview", "Comparison", "Performance Trends"])
    
    with tab1:
        st.subheader("Model Performance Overview")
        
        if not filtered_df.empty:
            # Format the dataframe for display
            display_df = filtered_df.copy()
            display_df["map50"] = display_df["map50"].apply(lambda x: f"{x:.2f}")
            display_df["map50_95"] = display_df["map50_95"].apply(lambda x: f"{x:.2f}")
            display_df["iou"] = display_df["iou"].apply(lambda x: f"{x:.2f}")
            
            # Check if model_file column exists (for compatibility with older data)
            if "model_file" not in display_df.columns:
                display_df["model_file"] = ""
                
            # Add model file status column
            display_df["has_model_file"] = display_df["model_file"].apply(lambda x: "✓" if x else "✗")
            
            # Rename columns for better display
            display_df = display_df[["name", "map50", "map50_95", "iou", "training_time", "has_model_file", "last_updated", "status"]]
            display_df.columns = ["Model Name", "mAP@0.5", "mAP@0.5:0.95", "IoU", "Training Time", "Model File", "Last Updated", "Status"]
        else:
            display_df = pd.DataFrame(columns=["Model Name", "mAP@0.5", "mAP@0.5:0.95", "IoU", "Training Time", "Model File", "Last Updated", "Status"])
            
        st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        st.subheader("Model Metrics Comparison")
        
        if not filtered_df.empty:
            # Radar chart for comparing models across metrics
            fig_radar = go.Figure()
            
            for idx, row in filtered_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row["map50"]*100, row["map50_95"]*100, row["iou"]*100],
                    theta=["mAP@0.5", "mAP@0.5:0.95", "IoU"],
                    fill="toself",
                    name=row["name"]
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[50, 100]
                    )
                ),
                showlegend=True
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Bar chart comparison of key metrics
            metrics_to_plot = st.multiselect(
                "Select metrics to compare",
                options=["map50", "map50_95", "iou"],
                default=["map50", "iou"],
                format_func=lambda x: {
                    "map50": "mAP@0.5", 
                    "map50_95": "mAP@0.5:0.95", 
                    "iou": "IoU"
                }[x]
            )
            
            if metrics_to_plot:
                comparison_df = pd.melt(
                    filtered_df,
                    id_vars=["name"],
                    value_vars=metrics_to_plot,
                    var_name="metric",
                    value_name="value"
                )
                
                # Map the metric codes to more readable names
                metric_names = {
                    "map50": "mAP@0.5", 
                    "map50_95": "mAP@0.5:0.95", 
                    "iou": "IoU"
                }
                comparison_df["metric"] = comparison_df["metric"].map(metric_names)
                
                fig_comparison = px.bar(
                    comparison_df,
                    x="name",
                    y="value",
                    color="metric",
                    barmode="group",
                    labels={"name": "Model", "value": "Value", "metric": "Metric"},
                    text_auto='.2f'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
        else:
            st.info("No models to display")
    
    with tab3:
        st.subheader("Performance Trends Over Time")
        
        if not history_df.empty:
            # Get unique models from history data
            available_models = history_df["model"].unique()
            
            # Let user select which models to show in the trend
            selected_models = st.multiselect(
                "Select models to display",
                options=available_models,
                default=available_models[:3] if len(available_models) >= 3 else available_models
            )
            
            metric_choice = st.radio(
                "Select metric to track",
                options=["mAP@0.5", "IoU"],
                horizontal=True
            )
            
            # Filter history data for the selected metric and models
            trend_data = history_df[
                (history_df["metric"] == metric_choice) & 
                (history_df["model"].isin(selected_models))
            ]
            
            if not trend_data.empty:
                fig_trend = px.line(
                    trend_data, 
                    x="date", 
                    y="value", 
                    color="model",
                    markers=True,
                    labels={"date": "Date", "value": metric_choice, "model": "Model"}
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No trend data available for the selected models and metric")
        else:
            st.info("No historical data available")

with main_tabs[1]:  # Model Management tab
    # Create tabs for model management
    model_tabs = st.tabs(["Add New Model", "Update Existing Model", "Rename Model", "Delete Model", "Model Files"])
    
    with model_tabs[0]:  # Add New Model tab
        st.header("Add New Model")
        st.markdown("Enter the details of your newly trained model below.")
        
        with st.form("new_model_form"):
            # Model basic info
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("Model Name", value="NTRM-")
                status = st.selectbox("Status", options=["Development", "Testing", "Production"])
            
            with col2:
                training_time = st.text_input("Training Time (e.g., 5h 30m)")
                model_file = st.file_uploader("Upload Model File (.pt)", type=["pt"])
            
            # Performance metrics
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                map50 = st.number_input("mAP@0.5", min_value=0.0, max_value=1.0, value=0.85, format="%.2f")
            
            with col2:
                map50_95 = st.number_input("mAP@0.5:0.95", min_value=0.0, max_value=1.0, value=0.70, format="%.2f")
            
            with col3:
                iou = st.number_input("IoU", min_value=0.0, max_value=1.0, value=0.75, format="%.2f")
            
            # Add historical data points
            st.subheader("Add Historical Data (Optional)")
            st.markdown("Add historical performance data points for this model")
            
            add_history = st.checkbox("Add historical data point")
            
            if add_history:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    history_date = st.date_input("Date", value=datetime.now())
                
                with col2:
                    history_metric = st.selectbox("Metric", options=["mAP@0.5", "IoU"])
                
                with col3:
                    history_value = st.number_input("Value", min_value=0.0, max_value=1.0, value=0.8, format="%.2f")
            
            submitted = st.form_submit_button("Add Model")
            
            if submitted:
                if model_name and training_time:
                    # Check if model name already exists
                    if model_name in df["name"].values:
                        st.error(f"A model with the name '{model_name}' already exists. Please use a different name.")
                    else:
                        # Process model file if uploaded
                        model_file_path = ""
                        if model_file is not None:
                            model_file_path = save_model_file(model_name, model_file)
                        
                        # Add new model to dataframe
                        new_model = {
                            "name": model_name,
                            "map50": map50,
                            "map50_95": map50_95,
                            "iou": iou,
                            "training_time": training_time,
                            "last_updated": datetime.now().strftime("%Y-%m-%d"),
                            "status": status,
                            "model_file": model_file_path
                        }
                        
                        df = pd.concat([df, pd.DataFrame([new_model])], ignore_index=True)
                        st.session_state.model_data = df
                        
                        # Add historical data point if specified
                        if add_history and history_date:
                            new_history = {
                                "date": pd.Timestamp(history_date),
                                "model": model_name,
                                "metric": history_metric,
                                "value": history_value
                            }
                            
                            history_df = pd.concat([history_df, pd.DataFrame([new_history])], ignore_index=True)
                            st.session_state.history_data = history_df
                        
                        st.success(f"Model '{model_name}' added successfully!")
                else:
                    st.error("Please fill in all required fields (Model Name and Training Time)")
    
    with model_tabs[1]:  # Update Existing Model tab
        st.header("Update Existing Model")
        
        if not df.empty:
            with st.form("update_model_form"):
                model_to_update = st.selectbox("Select Model to Update", options=df["name"].tolist())
                update_type = st.radio("What would you like to update?", 
                                    ["Performance Metrics", "Add Historical Data Point"])
                
                if update_type == "Performance Metrics":
                    # Get current values for the selected model
                    current_model = df[df["name"] == model_to_update].iloc[0]
                    
                    st.subheader("Update Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        u_map50 = st.number_input("New mAP@0.5", min_value=0.0, max_value=1.0, 
                                                value=float(current_model["map50"]), format="%.2f")
                    
                    with col2:
                        u_map50_95 = st.number_input("New mAP@0.5:0.95", min_value=0.0, max_value=1.0, 
                                                value=float(current_model["map50_95"]), format="%.2f")
                    
                    with col3:
                        u_iou = st.number_input("New IoU", min_value=0.0, max_value=1.0, 
                                              value=float(current_model["iou"]), format="%.2f")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        u_status = st.selectbox("New Status", 
                                              options=["Development", "Testing", "Production"],
                                              index=["Development", "Testing", "Production"].index(current_model["status"]))
                    
                    with col2:
                        u_model_file = st.file_uploader("Replace Model File (.pt)", type=["pt"])
                
                else:  # Add Historical Data Point
                    st.subheader("Add New Historical Data Point")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        h_date = st.date_input("Date", value=datetime.now())
                    
                    with col2:
                        h_metric = st.selectbox("Metric", options=["mAP@0.5", "IoU"])
                    
                    with col3:
                        h_value = st.number_input("Value", min_value=0.0, max_value=1.0, value=0.8, format="%.2f")
                
                update_submitted = st.form_submit_button("Update Model")
                
                if update_submitted:
                    if update_type == "Performance Metrics":
                        # Process model file if uploaded
                        if u_model_file is not None:
                            model_file_path = save_model_file(model_to_update, u_model_file)
                            df.loc[df["name"] == model_to_update, "model_file"] = model_file_path
                        
                        # Update model metrics
                        df.loc[df["name"] == model_to_update, "map50"] = u_map50
                        df.loc[df["name"] == model_to_update, "map50_95"] = u_map50_95
                        df.loc[df["name"] == model_to_update, "iou"] = u_iou
                        df.loc[df["name"] == model_to_update, "status"] = u_status
                        df.loc[df["name"] == model_to_update, "last_updated"] = datetime.now().strftime("%Y-%m-%d")
                        
                        st.session_state.model_data = df
                        st.success(f"Model '{model_to_update}' updated successfully!")
                    
                    else:  # Add Historical Data Point
                        # Add new historical data point
                        new_history = {
                            "date": pd.Timestamp(h_date),
                            "model": model_to_update,
                            "metric": h_metric,
                            "value": h_value
                        }
                        
                        history_df = pd.concat([history_df, pd.DataFrame([new_history])], ignore_index=True)
                        st.session_state.history_data = history_df
                        st.success(f"Historical data point added for '{model_to_update}'!")
        else:
            st.info("No models available to update. Please add a model first.")

    with model_tabs[2]:  # Rename Model tab
        st.header("Rename Model")
        
        if not df.empty:
            with st.form("rename_model_form"):
                model_to_rename = st.selectbox("Select Model to Rename", options=df["name"].tolist())
                new_model_name = st.text_input("New Model Name", value="NTRM-")
                
                rename_submitted = st.form_submit_button("Rename Model")
                
                if rename_submitted:
                    if new_model_name:
                        # Check if new name already exists (excluding the model being renamed)
                        if new_model_name in df[df["name"] != model_to_rename]["name"].values:
                            st.error(f"A model with the name '{new_model_name}' already exists. Please use a different name.")
                        else:
                            # Rename the model
                            df, history_df = rename_model(model_to_rename, new_model_name)
                            st.success(f"Model '{model_to_rename}' has been renamed to '{new_model_name}'!")
                    else:
                        st.error("Please provide a new name for the model.")
        else:
            st.info("No models available to rename. Please add a model first.")
    
    with model_tabs[3]:  # Delete Model tab
        st.header("Delete Model")
        
        if not df.empty:
            with st.form("delete_model_form"):
                model_to_delete = st.selectbox("Select Model to Delete", options=df["name"].tolist())
                
                # Add warnings and confirmation
                st.warning(f"⚠️ Deleting a model will remove all its data, metrics, and files. This action cannot be undone.")
                confirm_delete = st.checkbox("I understand the consequences and want to delete this model")
                
                delete_submitted = st.form_submit_button("Delete Model")
                
                if delete_submitted:
                    if confirm_delete:
                        # Delete the model
                        df, history_df = delete_model(model_to_delete)
                        st.success(f"Model '{model_to_delete}' has been deleted successfully!")
                    else:
                        st.error("Please confirm deletion by checking the confirmation box.")
        else:
            st.info("No models available to delete. Please add a model first.")
    
    with model_tabs[4]:  # Model Files tab
        st.header("Model Files Management")
        
        # Filter models that have associated files
        models_with_files = df[df["model_file"].astype(bool)]
        
        if not models_with_files.empty:
            st.subheader("Uploaded Model Files")
            
            for idx, row in models_with_files.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{row['name']}** ({row['status']})")
                    st.text(f"File: {row['model_file']}")
                
                with col2:
                    # In Streamlit Cloud, we can download files stored in session state
                    if row['model_file'] in st.session_state.model_files:
                        st.download_button(
                            label="Download Model",
                            data=st.session_state.model_files[row['model_file']],
                            file_name=row['model_file'],
                            mime="application/octet-stream",
                            key=f"download_{idx}"
                        )
                    else:
                        st.write("File not available")
                
                with col3:
                    # Delete model file button (using a form to avoid rerunning when multiple delete buttons exist)
                    with st.form(key=f"delete_form_{idx}"):
                        delete_button = st.form_submit_button("Delete File")
                        if delete_button:
                            # Remove the file reference from the dataframe
                            df.loc[idx, "model_file"] = ""
                            
                            # Remove the file from session state
                            if row['model_file'] in st.session_state.model_files:
                                del st.session_state.model_files[row['model_file']]
                            
                            st.session_state.model_data = df
                            st.success(f"Model file for {row['name']} removed successfully!")
                            st.rerun()
                
                st.markdown("---")
        else:
            st.info("No model files have been uploaded yet. Upload model files when adding or updating models.")

with st.sidebar:
    st.markdown("---")
    if st.button("Reset to Default Data (Dev only)"):
        st.session_state.model_data = pd.DataFrame(default_model_data)
        st.session_state.history_data = generate_default_history_data()
        st.session_state.model_files = {}
        st.success("Reset to default data!")
        st.rerun()

st.markdown("---")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
