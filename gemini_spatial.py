import streamlit as st
from PIL import Image
import numpy as np
import os
import io
import base64
import google.generativeai as genai
from google.api_core import retry
import time
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2  # Move cv2 import to the top

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
    Analyze this hotel room image and create precise object detection with accurate bounding boxes.
    
    CRITICAL REQUIREMENTS:
    - Identify exactly where each object is located in the image
    - Create tight, pixel-perfect bounding boxes around each object
    - Use coordinates that match the EXACT position of objects (TV, bed, lamps, etc.)
    - Focus on furniture, fixtures, and major room elements
    - Label up to 15 most important objects only
    
    For each object:
    1. Use a simple, clear object name (single word preferred: "tv", "bed", "lamp", etc.)
    2. Provide a brief description
    3. Give EXACT bounding box coordinates as [x1, y1, x2, y2] where:
       - x1, y1: top-left corner (0.0 to 1.0)
       - x2, y2: bottom-right corner (0.0 to 1.0)
    
    IMPORTANT COORDINATE GUIDELINES:
    - Coordinates must be between 0.0 and 1.0
    - Ensure x1 < x2 and y1 < y2
    - Make sure boxes tightly fit each object (no extra space)
    - Double-check that coordinates match the actual object position
    
    Return ONLY a valid JSON array in this exact format:
    [
        {
            "object": "object_name",
            "description": "brief description",
            "coordinates": [x1, y1, x2, y2]
        },
        ...
    ]
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
        img_height, img_width = img_array.shape[:2]
        
        # Create figure and axis with exact image dimensions to prevent distortion
        dpi = 100
        fig_width = img_width / dpi
        fig_height = img_height / dpi
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Display the image without distortion
        ax.imshow(img_array)
        
        # Use a consistent blue color for all objects like in the reference image
        box_color = '#0066ff'  # Bright blue similar to the reference image
        
        # Draw each object with its label
        for i, obj in enumerate(labels):
            # Extract object info
            obj_name = obj["object"].lower()  # Convert to lowercase like in reference
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
            x1_px = int(x1 * img_width)
            y1_px = int(y1 * img_height)
            x2_px = int(x2 * img_width)
            y2_px = int(y2 * img_height)
            
            # Skip boxes that are too small (likely errors)
            if (x2_px - x1_px < 10) or (y2_px - y1_px < 10):
                continue
                
            # Draw rectangle with blue color and no fill
            rect = Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px, 
                            linewidth=1.5, edgecolor=box_color, facecolor='none', alpha=0.9)
            ax.add_patch(rect)
            
            # Add label with blue background at the top of the box
            # Similar to the reference image style
            text_bg = dict(boxstyle="round,pad=0.2", fc=box_color, ec=box_color, alpha=1.0)
            ax.text(x1_px, max(0, y1_px-2), obj_name, color='white', fontweight='bold',
                   bbox=text_bg, fontsize=9, verticalalignment='bottom')
        
        # Remove axis ticks and frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')  # Turn off all axis elements
        
        # Save the figure to a buffer with tight layout
        buf = io.BytesIO()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        
        # Return the image with annotations
        return Image.open(buf)
    
    except Exception as e:
        st.error(f"Error drawing labels: {str(e)}")
        st.text(f"Error details: {str(e)}")
        return image  # Return original image if labeling fails

# Add new function for damage assessment
def detect_room_changes(model, before_image, after_image):
    """Detect changes between before and after room images for damage assessment"""
    prompt = """
    Compare these two hotel room images (BEFORE and AFTER guest stay) and identify all changes and potential damages.
    
    CRITICAL REQUIREMENTS:
    - Identify ALL differences between the before and after images
    - Focus on potential damages, missing items, and room condition changes
    - Classify each change by severity: MINOR, MODERATE, or SEVERE
    - Estimate if the change warrants charging the guest
    - Specifically identify any missing items that were present in the before image
    - Highlight areas that require cleaning attention
    
    For each identified change:
    1. Describe what changed specifically
    2. Classify severity (MINOR/MODERATE/SEVERE)
    3. Indicate if this typically warrants a charge (YES/NO)
    4. Provide estimated cost range if applicable
    
    GUIDELINES FOR CHARGING:
    - Normal wear and tear or minor mess: NO CHARGE
    - Moderate damages (small stains, minor breakage): POSSIBLE CHARGE
    - Severe damages (broken furniture, major stains): DEFINITE CHARGE
    - Missing items: DEFINITE CHARGE based on item value
    
    Return a detailed analysis in markdown format with these specific sections:
    1. Summary of room condition changes
    2. Missing Items (with estimated replacement costs)
    3. Damages (itemized with severity and charge recommendations)
    4. Areas Requiring Cleaning (with severity ratings)
    5. Total estimated charges (if applicable)
    6. Final recommendation
    
    Use markdown formatting with headers, bullet points, and tables for clarity.
    """
    
    try:
        # Combine both images with prompt
        response = model.generate_content([prompt, 
                                          {"mime_type": "image/jpeg", "data": encode_image(before_image)},
                                          {"mime_type": "image/jpeg", "data": encode_image(after_image)}])
        return response.text
    except Exception as e:
        st.error(f"Error analyzing room changes: {e}")
        time.sleep(1)  # Wait before retry
        raise

# Enhanced function to visualize changes between images
def visualize_room_changes(before_image, after_image):
    """Create a visual comparison highlighting changes between before and after images"""
    try:
        # Convert images to numpy arrays
        before_array = np.array(before_image)
        after_array = np.array(after_image)
        
        # Ensure images are the same size
        if before_array.shape != after_array.shape:
            after_array = cv2.resize(after_array, (before_array.shape[1], before_array.shape[0]))
        
        # Calculate absolute difference between images
        diff = cv2.absdiff(before_array, after_array)
        
        # Convert to grayscale for thresholding
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to highlight significant changes
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Create a heatmap of changes
        heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
        
        # Overlay heatmap on after image
        overlay = cv2.addWeighted(after_array, 0.7, heatmap, 0.3, 0)
        
        # Convert back to PIL Image
        return Image.fromarray(overlay)
    except Exception as e:
        st.error(f"Error visualizing changes: {e}")
        return after_image  # Return original image if visualization fails

# New function to create a cleaning heatmap
def create_cleaning_heatmap(before_image, after_image):
    """Create a heatmap specifically highlighting areas that need cleaning"""
    try:
        # Convert images to numpy arrays
        before_array = np.array(before_image)
        after_array = np.array(after_image)
        
        # Ensure images are the same size
        if before_array.shape != after_array.shape:
            after_array = cv2.resize(after_array, (before_array.shape[1], before_array.shape[0]))
        
        # Convert to HSV for better color analysis
        before_hsv = cv2.cvtColor(before_array, cv2.COLOR_RGB2HSV)
        after_hsv = cv2.cvtColor(after_array, cv2.COLOR_RGB2HSV)
        
        # Calculate difference in saturation and value channels (good for detecting stains and dirt)
        s_diff = cv2.absdiff(after_hsv[:,:,1], before_hsv[:,:,1])
        v_diff = cv2.absdiff(after_hsv[:,:,2], before_hsv[:,:,2])
        
        # Combine differences
        combined_diff = cv2.addWeighted(s_diff, 0.5, v_diff, 0.5, 0)
        
        # Apply threshold to highlight significant changes
        _, thresh = cv2.threshold(combined_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((7,7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Create a custom colormap for cleaning (green to red)
        cleaning_heatmap = np.zeros_like(after_array)
        cleaning_heatmap[:,:,0] = 0  # Blue channel
        cleaning_heatmap[:,:,1] = 0  # Green channel
        cleaning_heatmap[:,:,2] = thresh  # Red channel
        
        # Overlay heatmap on after image
        overlay = cv2.addWeighted(after_array, 0.7, cleaning_heatmap, 0.7, 0)
        
        # Convert back to PIL Image
        return Image.fromarray(overlay)
    except Exception as e:
        st.error(f"Error creating cleaning heatmap: {e}")
        return after_image  # Return original image if visualization fails

# New function to detect missing items
def detect_missing_items(before_image, after_image):
    """Create a visualization highlighting potentially missing items"""
    try:
        # Convert images to numpy arrays
        before_array = np.array(before_image)
        after_array = np.array(after_image)
        
        # Ensure images are the same size
        if before_array.shape != after_array.shape:
            after_array = cv2.resize(after_array, (before_array.shape[1], before_array.shape[0]))
        
        # Convert to grayscale
        before_gray = cv2.cvtColor(before_array, cv2.COLOR_RGB2GRAY)
        after_gray = cv2.cvtColor(after_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate difference (before - after) to highlight things in before but not in after
        diff = cv2.subtract(before_gray, after_gray)
        
        # Apply threshold to highlight significant changes
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours of potential missing items
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the before image to draw on
        missing_vis = before_array.copy()
        
        # Draw contours of potentially missing items
        cv2.drawContours(missing_vis, contours, -1, (0, 0, 255), 2)
        
        # Highlight areas of potential missing items
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Only highlight significant areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(missing_vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(missing_vis, "Missing?", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Convert back to PIL Image
        return Image.fromarray(missing_vis)
    except Exception as e:
        st.error(f"Error detecting missing items: {e}")
        return before_image  # Return original image if visualization fails

# Main app
st.title("Gemini Spatial Understanding")
st.write("Upload room images for AI-powered spatial understanding and damage assessment.")

# Initialize Gemini model
try:
    model = configure_genai()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Room Analysis", "Damage Assessment"])

with tab1:
    # Original room analysis functionality
    st.header("Room Analysis")
    st.write("Upload a room image to get AI-powered spatial understanding and analysis.")
    
    # File uploader and webcam option
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="single_image")
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
                                        parsed_json = json.loads(json_str)
                                        st.json(parsed_json)
                                        
                                        # Display number of objects detected
                                        st.text(f"Total objects detected: {len(parsed_json)}")
                                    else:
                                        parsed_json = json.loads(labels_json)
                                        st.json(parsed_json)
                                        st.text(f"Total objects detected: {len(parsed_json)}")
                                except Exception as json_error:
                                    st.error(f"JSON parsing error: {str(json_error)}")
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
                "Spatial Labeling",
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

with tab2:
    # New damage assessment functionality
    st.header("Damage Assessment")
    st.write("Compare before and after images to detect changes and assess potential damages.")
    
    # Create columns for before and after images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("BEFORE Check-in")
        before_file = st.file_uploader("Upload BEFORE image", type=["jpg", "jpeg", "png"], key="before_image")
        use_webcam_before = st.checkbox("Use Webcam for BEFORE image")
        
        if use_webcam_before and not before_file:
            before_camera = st.camera_input("Capture BEFORE Image")
            if before_camera:
                before_image = Image.open(before_camera)
                st.image(before_image, caption="BEFORE Image", use_column_width=True)
            else:
                before_image = None
        elif before_file:
            before_image = Image.open(before_file)
            st.image(before_image, caption="BEFORE Image", use_column_width=True)
        else:
            before_image = None
    
    with col2:
        st.subheader("AFTER Check-out")
        after_file = st.file_uploader("Upload AFTER image", type=["jpg", "jpeg", "png"], key="after_image")
        use_webcam_after = st.checkbox("Use Webcam for AFTER image")
        
        if use_webcam_after and not after_file:
            after_camera = st.camera_input("Capture AFTER Image")
            if after_camera:
                after_image = Image.open(after_camera)
                st.image(after_image, caption="AFTER Image", use_column_width=True)
            else:
                after_image = None
        elif after_file:
            after_image = Image.open(after_file)
            st.image(after_image, caption="AFTER Image", use_column_width=True)
        else:
            after_image = None
    
    # Assess damages button
    if before_image is not None and after_image is not None:
        if st.button("Assess Damages"):
            with st.spinner("Analyzing changes and potential damages..."):
                try:
                    # Create visual comparison
                    change_visualization = visualize_room_changes(before_image, after_image)
                    
                    # Display change visualization
                    st.subheader("Change Visualization")
                    st.image(change_visualization, caption="Areas of Change Highlighted", use_column_width=True)
                    
                    # Get damage assessment from Gemini
                    damage_assessment = detect_room_changes(model, before_image, after_image)
                    
                    # Display damage assessment
                    st.subheader("Damage Assessment Report")
                    st.markdown(damage_assessment)
                    
                    # Save assessment to session state
                    st.session_state.damage_assessment = damage_assessment
                    
                except Exception as e:
                    st.error(f"Error during damage assessment: {e}")
    else:
        st.info("Please upload or capture both BEFORE and AFTER images to assess damages.")

# Add tips for better results
st.sidebar.header("Tips for Better Results")
st.sidebar.write("""
- Use well-lit images for clearer analysis
- Ensure the camera captures the entire room from the same angle for both before/after images
- For damage assessment, try to capture images from the same position and angle
- Make sure important areas like furniture, walls, and fixtures are clearly visible
""")

# Add information about the model
st.sidebar.header("About")
st.sidebar.write("""
This tool uses Google's Gemini Pro Vision model to analyze room images and provide 
detailed spatial understanding. The damage assessment feature compares before and after 
images to detect changes and potential damages that may warrant charging guests.
""")