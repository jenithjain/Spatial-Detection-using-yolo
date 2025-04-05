import streamlit as st
from PIL import Image
import numpy as np
import os
import io
import base64
import google.generativeai as genai
from google.api_core import retry
import time

# Set page configuration
st.set_page_config(
    page_title="Gemini Spatial Understanding",
    page_icon="🏠",
    layout="wide"
)

# Configure Google API key
@st.cache_resource
def configure_genai():
    # Set your API key directly
    api_key = "AIzaSyALnrV7Cb5fM8_PdYJGGcn2xIC932m8XVQ"
    
    # Alternative method using secrets or input
    # api_key = st.secrets.get("GOOGLE_API_KEY", None)
    # if api_key is None:
    #     api_key = st.text_input("Enter your Google API Key:", type="password")
    #     if not api_key:
    #         st.warning("Please enter a valid Google API key to continue.")
    #         st.stop()
    
    genai.configure(api_key=api_key)
    
    # Configure the Gemini Pro Vision model
    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    return model

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Import additional libraries for image annotation
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

# Function to analyze room with Gemini
@retry.Retry(predicate=retry.if_exception_type(Exception))
def analyze_room_with_gemini(model, image, prompt):
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        st.error(f"Error generating content: {e}")
        time.sleep(1)  # Wait before retry
        raise

# Enhanced function to label objects with coordinates
def label_room_objects(model, image):
    """Generate labels with coordinates for objects in the room image"""
    prompt = """
    Detect and identify ALL objects present in this hotel room image with precise bounding boxes.
    
    IMPORTANT INSTRUCTIONS:
    - Identify every visible object in the room (furniture, fixtures, decor, etc.)
    - Create tight, accurate bounding boxes around each object
    - Ensure coordinates are precise and match the actual object position
    - Label common hotel room items: bed, TV, chairs, tables, lamps, artwork, etc.
    - Don't miss small but important objects
    - Limit to no more than 20 most significant items
    
    For each object, provide:
    1. A simple, clear object name (single word preferred)
    2. A brief description
    3. PRECISE bounding box coordinates as [x1, y1, x2, y2] where:
       - x1, y1: top-left corner (0.0 to 1.0)
       - x2, y2: bottom-right corner (0.0 to 1.0)
    
    Return ONLY a valid JSON array in this exact format:
    [
        {
            "object": "object_name",
            "description": "brief description",
            "coordinates": [x1, y1, x2, y2]
        },
        ...
    ]
    
    CRITICAL: Ensure coordinates are extremely accurate - objects like TV should have boxes that precisely match their actual position in the image.
    """
    
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        st.error(f"Error labeling objects: {e}")
        time.sleep(1)  # Wait before retry
        raise

# Function to draw labeled objects on image with fixed coordinates
def draw_labeled_objects(image, labels_json):
    """Draw bounding boxes and labels on the image in the style of the starter app"""
    try:
        # Parse the JSON response
        if isinstance(labels_json, str):
            # Try to extract JSON from text response if needed
            try:
                # Find JSON array in the response
                json_start = labels_json.find('[')
                json_end = labels_json.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = labels_json[json_start:json_end]
                    labels = json.loads(json_str)
                else:
                    labels = json.loads(labels_json)
            except Exception as json_error:
                st.error(f"Could not parse JSON response: {str(json_error)}")
                st.text(labels_json)
                return image
        else:
            labels = labels_json
        
        # Convert PIL Image to numpy array for matplotlib
        img_array = np.array(image)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img_array)
        
        # Use a consistent blue color for all objects like in the starter app
        box_color = '#0066ff'  # Bright blue similar to the reference image
        
        # Draw each object with its label
        for i, obj in enumerate(labels):
            # Extract object info
            obj_name = obj["object"].lower()  # Convert to lowercase like in reference
            description = obj.get("description", "")
            coords = obj["coordinates"]
            
            # Validate coordinates
            if len(coords) != 4:
                continue  # Skip invalid coordinates
                
            # Convert coordinates to percentages if they're not already
            if any(c > 1.0 for c in coords):
                coords = [c/100 if c > 1.0 else c for c in coords]
            
            # Ensure coordinates are within bounds
            coords = [max(0.0, min(c, 1.0)) for c in coords]
            
            # Ensure x1 < x2 and y1 < y2
            x1, y1, x2, y2 = coords
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Convert percentage coordinates to pixel values
            img_height, img_width = img_array.shape[:2]
            x1_px = int(x1 * img_width)
            y1_px = int(y1 * img_height)
            x2_px = int(x2 * img_width)
            y2_px = int(y2 * img_height)
            
            # Draw rectangle with blue color and no fill
            rect = Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px, 
                            linewidth=2, edgecolor=box_color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add label with blue background at the top of the box
            # Similar to the reference image style
            text_bg = dict(boxstyle="round,pad=0.3", fc=box_color, ec=box_color, alpha=0.8)
            ax.text(x1_px, y1_px-5, obj_name, color='white', fontweight='bold',
                   bbox=text_bg, fontsize=10, verticalalignment='bottom')
        
        # Remove axis ticks and frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Save the figure to a buffer with higher DPI
        buf = io.BytesIO()
        plt.tight_layout(pad=0)
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        
        # Return the image with annotations
        return Image.open(buf)
    
    except Exception as e:
        st.error(f"Error drawing labels: {str(e)}")
        return image  # Return original image if labeling fails

# Main app
st.title("Gemini Spatial Understanding")
st.write("Upload a room image to get AI-powered spatial understanding and analysis.")

# Initialize Gemini model
try:
    model = configure_genai()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# File uploader and webcam option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
use_webcam = st.checkbox("Use Webcam Instead")

if use_webcam and not uploaded_file:
    # Webcam input using Streamlit's camera input
    camera_image = st.camera_input("Capture Room Image")
    if camera_image:
        image = Image.open(camera_image)
    else:
        image = None
elif uploaded_file:
    image = Image.open(uploaded_file)
else:
    image = None

if image is not None:
    # Create columns for original image and analysis
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        st.image(image, use_column_width=True)
        
        # Add automatic labeling option
        if st.button("Auto-Label Objects"):
            with st.spinner("Analyzing and labeling objects in the image..."):
                try:
                    # Process image for Gemini
                    img_data = {"mime_type": "image/jpeg", "data": encode_image(image)}
                    
                    # Get object labels with coordinates from Gemini
                    labels_json = label_room_objects(model, img_data)
                    
                    # Save labels to session state
                    st.session_state.object_labels = labels_json
                    
                    try:
                        # Draw labels on the image
                        labeled_image = draw_labeled_objects(image, labels_json)
                        
                        # Display labeled image
                        st.subheader("Labeled Objects")
                        st.image(labeled_image, use_column_width=True)
                        
                        # Hide JSON data in an expandable section
                        with st.expander("View Technical Details"):
                            st.text("JSON Object Data:")
                            try:
                                # Try to extract JSON from text response if needed
                                json_start = labels_json.find('[')
                                json_end = labels_json.rfind(']') + 1
                                if json_start >= 0 and json_end > json_start:
                                    json_str = labels_json[json_start:json_end]
                                    st.json(json.loads(json_str))
                                else:
                                    st.json(json.loads(labels_json))
                            except:
                                st.text(labels_json)
                        
                    except Exception as e:
                        st.error(f"Error displaying labeled image: {str(e)}")
                        st.text("Raw response:")
                        st.text(labels_json)
                        
                except Exception as e:
                    st.error(f"Error during object labeling: {e}")

    # Analysis options
    st.sidebar.header("Analysis Options")
    # Add a new analysis type for spatial labeling
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "General Room Description",
            "Object Identification",
            "Spatial Relationships",
            "Room Layout Analysis",
            "Design Style Analysis",
            "Spatial Labeling",  # New option
            "Custom Question"
        ]
    )
    
    # Prepare prompt based on analysis type
    if analysis_type == "General Room Description":
        prompt = "Describe this room in detail. What type of room is it? What are the main features and objects?"
    elif analysis_type == "Object Identification":
        prompt = "Identify all objects visible in this room. List them in order of prominence."
    elif analysis_type == "Spatial Relationships":
        prompt = "Describe the spatial relationships between objects in this room. How are they arranged relative to each other?"
    elif analysis_type == "Room Layout Analysis":
        prompt = "Analyze the layout of this room. Describe the floor plan, furniture arrangement, and overall space utilization."
    elif analysis_type == "Design Style Analysis":
        prompt = "What design style is this room? Analyze the colors, furniture, decorations, and overall aesthetic."
    elif analysis_type == "Spatial Labeling":
        prompt = """
        Create a detailed spatial map of this room. For each object:
        1. Identify what it is
        2. Describe its precise location in the room
        3. Note any relationships with nearby objects
        
        Format as a structured list with clear spatial indicators.
        """
    elif analysis_type == "Custom Question":
        prompt = st.sidebar.text_area("Enter your question about the room:", "What can you tell me about this room?")
    
    # Run analysis button
    if st.button("Analyze Room"):
        with st.spinner("Analyzing room with Gemini AI..."):
            try:
                # Process image for Gemini
                img_data = {"mime_type": "image/jpeg", "data": encode_image(image)}
                
                # Get analysis from Gemini
                analysis = analyze_room_with_gemini(model, img_data, prompt)
                
                # Display results
                with col2:
                    st.header("Gemini Analysis")
                    st.markdown(analysis)
                
                # Save analysis to session state for further use
                st.session_state.room_analysis = analysis
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
    
    # Additional features section
    if 'room_analysis' in st.session_state:
        st.header("Follow-up Questions")
        follow_up = st.text_input("Ask a follow-up question about the room:")
        
        if follow_up and st.button("Get Answer"):
            with st.spinner("Getting answer..."):
                try:
                    # Process image for Gemini
                    img_data = {"mime_type": "image/jpeg", "data": encode_image(image)}
                    
                    # Create context-aware prompt
                    context_prompt = f"""
                    Previous analysis: {st.session_state.room_analysis}
                    
                    New question: {follow_up}
                    
                    Please answer the new question based on the image and previous analysis.
                    """
                    
                    # Get follow-up answer from Gemini
                    follow_up_answer = analyze_room_with_gemini(model, img_data, context_prompt)
                    
                    # Display results
                    st.subheader("Answer")
                    st.markdown(follow_up_answer)
                    
                except Exception as e:
                    st.error(f"Error getting follow-up answer: {e}")

# Add tips for better results
st.sidebar.header("Tips for Better Results")
st.sidebar.write("""
- Use well-lit images for clearer analysis
- Ensure the camera captures the entire room for complete analysis
- For specific details, use the Custom Question option
- Try different analysis types for comprehensive understanding
""")

# Add information about the model
st.sidebar.header("About")
st.sidebar.write("""
This tool uses Google's Gemini Pro Vision model to analyze room images and provide 
detailed spatial understanding. It can identify objects, analyze layouts, and answer 
specific questions about the room.
""")