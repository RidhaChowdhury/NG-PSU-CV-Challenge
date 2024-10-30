import streamlit as st
import time
import random

st.set_page_config(layout="wide")

progress_messages = [
    "Starting up the engines...",
    "Analyzing pixels closely...",
    "Trying our best!",
    "Almost there...",
    "Frying our GPU...",
    "Just a bit longer!",
    "Finishing touches...",
]

# Page Title
st.title("Ensemble Model Selection and Aggregation")

# Step 1: Model Selection
st.header("Select Models")
selected_models = {
    "YOLO": st.checkbox("YOLO"),
    "F-RCNN": st.checkbox("F-RCNN"),
    "DETR": st.checkbox("DETR")
}

# Step 2: Choose Aggregation Method
st.header("Select Aggregation Method")
aggregation_method = st.selectbox(
    "Choose an aggregation method",
    ["NMS (Non-Maximum Suppression)", "WBF (Weighted Box Fusion)", "IOV (Intersection over Union)"]
)

# Step 3: Image Upload
st.header("Upload Images")
uploaded_images = st.file_uploader(
    "Choose image files", 
    accept_multiple_files=True, 
    type=["jpg", "jpeg", "png"]
)

if st.button("Run"):
    selected_model_list = [model for model, checked in selected_models.items() if checked]
    
    # Check if at least one model is selected
    if not selected_model_list:
        st.error("Please select at least one model for analysis.")
    
    # Check if images are uploaded
    elif not uploaded_images:
        st.error("Please upload at least one image to process.")
    
    # Proceed only if both models and images are selected
    else:
        st.header("Image Results")
        
        # Initialize progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()  # Placeholder for progress messages
        
        # Loop through each uploaded image
        for i, image in enumerate(uploaded_images):
            # Display random progress message
            message_index = random.randint(0, len(progress_messages)-1)
            status_text.text(progress_messages[message_index])
            
            # Simulate processing time
            time.sleep(0.5)  # Replace with actual processing code if needed

            # Create a dynamic column layout based on the number of models selected
            num_columns = len(selected_model_list) + 2  # Original + models + aggregated result
            if num_columns == 3:
                num_columns -= 1
            columns = st.columns(num_columns)
            
            # Original image in the first column
            with columns[0]:
                st.image(image, caption="Original Image")
            
            # Placeholder for each model's result in subsequent columns
            for idx, model_name in enumerate(selected_model_list):
                with columns[idx + 1]:
                    st.image(image, caption=f"{model_name} Result (placeholder)")

            if num_columns > 2:
                # Placeholder for aggregated result in the last column
                with columns[-1]:
                    st.image(image, caption="Aggregated Result (placeholder)")
            
            # Update the progress bar
            progress = int((i + 1) / len(uploaded_images) * 100)
            progress_bar.progress(progress)
        
        # Final success message
        status_text.text("Done! All images processed 🎉")
        st.success("Processing complete!")
        
        # Display Selected Options (for debugging/checking the UI)
        st.write("Selected Models:", selected_model_list)
        st.write("Aggregation Method:", aggregation_method)

# Run Streamlit with: python -m streamlit run app.py
