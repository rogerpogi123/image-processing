import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import subprocess 
import sys 
import tempfile # NEW: For creating temporary file paths
import shutil # NEW: For copying file content

# --- MAC OS FIX ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- CONFIGURATION ---
MODEL_PATH = 'my_animal_model.keras'
ORIGINAL_CLASS_NAMES = ['Cats', 'Dogs'] 

# --- LOAD MODEL (Kept for status, but training is still forced) ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None

try:
    model = load_model()
except Exception:
    model = None

# --- UI LAYOUT ---
st.title("🐶 Animal Recognizer 🐱")
st.markdown("### Upload an image to detect if it is a **Cat** or a **Dog**.")

# --- SIDEBAR ---
st.sidebar.header("Status")
if model is not None:
    st.sidebar.info("Model file found. The 'Analyze' button will still rerun the full training script.")
else:
    st.sidebar.warning("Model file not found. Running the script is mandatory.")

# --- FILE UPLOADER ---
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if file is not None:
    # 1. Display the uploaded image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # 2. Add 'Analyze' Button
    if st.button('Analyze Image', type="primary"):
        
        temp_path = None
        try:
            # --- STEP 1: Save the uploaded file to a temporary location ---
            file.seek(0)
            
            # Create a temporary file path using the original extension
            extension = file.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + extension) as temp_img_file:
                shutil.copyfileobj(file, temp_img_file)
                temp_path = temp_img_file.name

            # --- STEP 2: Execute the external script (recognition_script.py) ---
            
            # Function to execute the external script
            def run_external_script(image_path):
                # Use sys.executable to guarantee the correct virtual environment's Python is used
                python_executable = sys.executable 
                script_path = "recognition_script.py"
                
                # Execute the script, passing the temporary image path as an argument
                result = subprocess.run(
                    [python_executable, script_path, image_path], # <-- Image path passed here
                    capture_output=True,
                    text=True,
                    check=True,
                    env=dict(os.environ, KMP_DUPLICATE_LIB_OK='True')
                )
                return result.stdout

            with st.spinner("Starting Model Training... (This process takes time on every click)"):
                output = run_external_script(temp_path)
                
                # --- STEP 3: Parse the Final Prediction from the Output ---
                
                predicted_line = [line for line in output.split('\n') if 'Predicted Class:' in line]
                
                if predicted_line:
                    prediction_text = predicted_line[0].split(':')[-1].strip()
                    confidence_line = [line for line in output.split('\n') if 'Confidence:' in line]
                    confidence_text = confidence_line[0].split(':')[-1].strip() if confidence_line else "N/A"
                    
                    try:
                        confidence_value = float(confidence_text.replace('%', '').strip())
                    except ValueError:
                        confidence_value = 0
                    
                    st.divider()
                    st.success(f"Final Predicted Class: **{prediction_text}**")
                    st.write(f"Confidence: **{confidence_text}**")
                    st.info(f"Note: The script analyzed the **uploaded image** after retraining the model.")
                    st.progress(int(confidence_value))

                else:
                    st.error("Could not find final prediction result in script output. Check the terminal for full logs.")
                    st.code(output, language='bash') # Display full log on error

        except subprocess.CalledProcessError as e:
            st.error("Error executing the script. Check your terminal for details.")
            st.code(f"Stderr: {e.stderr}\nStdout: {e.stdout}", language='bash')
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            # --- STEP 4: Clean up the temporary file ---
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

elif model is None:
    st.warning("Please wait for the script execution to finish.")