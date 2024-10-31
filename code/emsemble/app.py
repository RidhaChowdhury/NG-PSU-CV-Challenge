import streamlit as st
import time
import random
import tempfile
from Ensemble import run_ensemble
import os

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
    "YoloModel": st.checkbox("YOLO"),
    "FasterRCNNModel": st.checkbox("F-RCNN"),
    "DetrModel": st.checkbox("DETR")
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

        image_paths = []
        # saving images
        for i, image in enumerate(uploaded_images):
            file_path = os.path.join("./data/uploaded_images/", image.name)
            with open(file_path, "wb") as f:
                f.write(image.read())
            # Store and display the path
            image_paths.append(file_path)

        # # Display random progress message
        # message_index = random.randint(0, len(progress_messages)-1)
        # status_text.text(progress_messages[message_index])
        status_text.text("Processing images...")
        
        # # Simulate processing time
        run_ensemble(selected_model_list, "./data/uploaded_images/")

        for i, image in enumerate(uploaded_images):
            # Create a dynamic column layout based on the number of models selected
            num_columns = len(selected_model_list) + 2  # Original + models + aggregated result
            if num_columns == 3:
                num_columns -= 1
            columns = st.columns(num_columns)
            
            # Original image in the first column
            with columns[0]:
                st.image(image, caption="Original Image")
            
            image_name_without_extension = os.path.splitext(image.name)[0]

            # Placeholder for each model's result in subsequent columns
            for idx, model_name in enumerate(selected_model_list):
                with columns[idx + 1]:
                    st.image('./data/outputted_images/' + image_name_without_extension + '_' + model_name + '.jpg', caption=f"{model_name} Result (placeholder)")

            if num_columns > 2:
                # Placeholder for aggregated result in the last column
                with columns[-1]:
                    st.image('./data/outputted_images/' + image_name_without_extension + '_aggregated_image.jpg', caption="Aggregated Result (placeholder)")
            
            # Update the progress bar
            progress = int((i + 1) / len(uploaded_images) * 100)
            progress_bar.progress(progress)
        
    # Final success message
    status_text.text("Done! All images processed 🎉")
    st.success("Processing complete!")

# Run Streamlit with: python -m streamlit run app.py
