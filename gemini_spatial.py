import os
import io
import base64
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
from PIL import Image
import time
import google.generativeai as genai
from google.api_core import retry

# Configure Google API key for Django environment
def configure_genai():
    # Set your API key directly
    api_key = "AIzaSyALnrV7Cb5fM8_PdYJGGcn2xIC932m8XVQ"
    
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

# Function to analyze room with Gemini
@retry.Retry(predicate=retry.if_exception_type(Exception))
def analyze_room_with_gemini(model, image, prompt):
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
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
        print(f"Error labeling objects: {e}")
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
                print(f"Could not parse JSON response: {str(json_error)}")
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
        plt.tight_layout()
        
        # Save figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        # Return the image with annotations
        return Image.open(buf)
        
    except Exception as e:
        print(f"Error drawing labels: {e}")
        return image

def detect_damages(image_path, output_path=None):
    """Detect damages in checkout image using Gemini Vision"""
    try:
        # Configure Gemini model
        model = configure_genai()
        
        # Load the image
        checkout_image = Image.open(image_path)
        
        # Encode image for Gemini
        checkout_img_data = {"mime_type": "image/jpeg", "data": encode_image(checkout_image)}
        
        # Create the damage detection prompt
        prompt = """
        Analyze this hotel room checkout image for any visible damages, stains, or issues.
        
        Identify and locate ALL of the following:
        1. Any visible damages to furniture, walls, floors, etc.
        2. Stains or discoloration
        3. Missing or broken items
        4. Any abnormalities that would require attention
        
        For each damage or issue, provide:
        - Type of damage (e.g., stain, scratch, broken item)
        - Severity (minor, moderate, major)
        - Precise location using normalized coordinates [x1, y1, x2, y2] where:
          - x1, y1: top-left corner (0.0 to 1.0)
          - x2, y2: bottom-right corner (0.0 to 1.0)
        
        Return a valid JSON with this structure:
        {
            "damages": [
                {
                    "type": "damage type",
                    "severity": "severity level",
                    "description": "brief description",
                    "coordinates": [x1, y1, x2, y2]
                }
            ],
            "summary": "overall assessment of damages found"
        }
        
        If no damages are found, return an empty damages array with appropriate summary.
        """
        
        # Get damage detection from Gemini
        response = model.generate_content([prompt, checkout_img_data])
        damage_text = response.text
        
        # Extract the JSON from the response
        try:
            # Find JSON in the response
            json_start = damage_text.find('{')
            json_end = damage_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = damage_text[json_start:json_end]
                damages_data = json.loads(json_str)
            else:
                damages_data = json.loads(damage_text)
            
            # Create visualization if output path is provided
            if output_path:
                # Create visualization
                # Convert PIL Image to numpy array for matplotlib
                img_array = np.array(checkout_image)
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.imshow(img_array)
                
                # Add a title with the summary
                if 'damages' in damages_data and damages_data['damages']:
                    # Color for damage markings - use red for damages
                    damage_color = '#ff0000'  # Red color for damages
                    
                    # Draw each damage with its label
                    for damage in damages_data['damages']:
                        # Extract damage info
                        damage_type = damage.get("type", "Unknown").capitalize()
                        severity = damage.get("severity", "Unknown").capitalize()
                        coords = damage.get("coordinates", [0, 0, 0, 0])
                        
                        # Validate coordinates
                        if len(coords) != 4:
                            continue  # Skip invalid coordinates
                            
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
                        
                        # Draw rectangle with red color and no fill
                        rect = Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px, 
                                        linewidth=2, edgecolor=damage_color, facecolor='none', alpha=0.8)
                        ax.add_patch(rect)
                        
                        # Add label with red background
                        label_text = f"{damage_type} ({severity})"
                        text_bg = dict(boxstyle="round,pad=0.3", fc=damage_color, ec=damage_color, alpha=0.8)
                        ax.text(x1_px, y1_px-5, label_text, color='white', fontweight='bold',
                               bbox=text_bg, fontsize=10, verticalalignment='bottom')
                
                # Remove axis ticks and frame
                ax.set_xticks([])
                ax.set_yticks([])
                plt.tight_layout()
                
                # Add title with damage summary
                if 'summary' in damages_data:
                    plt.suptitle(damages_data['summary'], fontsize=14, y=0.98)
                else:
                    plt.suptitle("No damages detected", fontsize=14, y=0.98)
                
                # Create output directory if it doesn't exist
                # Use the global os module to make directories
                if output_path:
                    output_dir = os.path.dirname(output_path)
                    os.makedirs(output_dir, exist_ok=True)
                
                # Save the image
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
            
            return damages_data
            
        except Exception as e:
            print(f"Error parsing damage detection result: {e}")
            return {"error": f"Failed to parse damage detection: {str(e)}"}
        
    except Exception as e:
        print(f"Error in damage detection: {e}")
        return {"error": str(e)}

# Function to analyze room images for a Django environment
def analyze_room_images(checkin_image_path, checkout_image_path, output_path=None):
    """Compare checkin and checkout room images and provide detailed analysis"""
    try:
        # Configure Gemini model
        model = configure_genai()
        
        # Load the images
        checkin_image = Image.open(checkin_image_path)
        checkout_image = Image.open(checkout_image_path)
        
        # Encode images for Gemini
        checkin_img_data = {"mime_type": "image/jpeg", "data": encode_image(checkin_image)}
        checkout_img_data = {"mime_type": "image/jpeg", "data": encode_image(checkout_image)}
        
        # Create the comparison prompt
        prompt = """
        Compare these two hotel room images (check-in and check-out) in detail. Identify any:
        1. Missing items
        2. Moved furniture
        3. New objects
        4. Visual differences
        5. Signs of damage or wear
        
        For each difference, provide:
        - What specific item/object changed
        - The nature of the change
        - Location in the room
        
        Return a JSON structure with these components:
        1. "differences": Array of all differences found
        2. "summary": Brief overall assessment
        3. "similarity_score": 0-100 percentage representing how similar the rooms are
        
        Format the response as valid JSON:
        {
            "differences": [
                {"item": "item name", "change": "description of change", "location": "location in room"}
            ],
            "summary": "overall assessment text",
            "similarity_score": percentage
        }
        """
        
        # Get the analysis from Gemini
        response = model.generate_content([prompt, checkin_img_data, checkout_img_data])
        analysis_text = response.text
        
        # Extract the JSON from the response
        try:
            # Find JSON in the response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                analysis_result = json.loads(json_str)
            else:
                analysis_result = json.loads(analysis_text)
                
            # Create visualization
            if output_path:
                # Create comparison visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                
                # Display images
                ax1.imshow(np.array(checkin_image))
                ax1.set_title("Check-in Image", fontsize=16)
                ax1.axis('off')
                
                ax2.imshow(np.array(checkout_image))
                ax2.set_title("Check-out Image", fontsize=16)
                ax2.axis('off')
                
                # Add summary as subtitle
                if 'summary' in analysis_result:
                    plt.suptitle(analysis_result['summary'], fontsize=16, y=0.98)
                
                # Ensure the output directory exists
                if output_path:
                    output_dir = os.path.dirname(output_path)
                    os.makedirs(output_dir, exist_ok=True)
                
                # Save the visualization
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
            
            return analysis_result
            
        except Exception as e:
            print(f"Error parsing analysis result: {e}")
            return {"error": f"Failed to parse analysis: {str(e)}"}
        
    except Exception as e:
        print(f"Error in analyze_room_images: {e}")
        return {"error": str(e)}