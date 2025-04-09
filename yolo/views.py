from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import uuid
import json
from django.conf import settings
import base64
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageColor, ImageFont
import io
from .models import Detection
from django.core.files.base import ContentFile
from pydantic import BaseModel
from typing import List

# Configure matplotlib to use a non-interactive backend to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from google.api_core import retry
import time

# Define BoundingBox class using Pydantic for validation
class BoundingBox(BaseModel):
    box_2d: List[int]
    label: str

# Initialize Gemini model
try:
    import google.generativeai as genai
    
    # Replace with your API key
    API_KEY = "*********************************"  # You should store this securely, e.g., in environment variables
    genai.configure(api_key=API_KEY)
    
    # Configure the Gemini Pro Vision model with better settings from gemini_spatial.py
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
    
    # Test that the model works
    test_response = model.generate_content("Hello")
    if not test_response:
        raise Exception("Model response test failed")
        
    print(f"Successfully loaded Gemini 2.0-flash model")
except Exception as e:
    print(f"Error loading Gemini model: {str(e)}")
    model = None

def encode_image(image_path):
    """Convert image to base64 encoding for Gemini API"""
    if isinstance(image_path, str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # If we're passed a PIL Image object
        buffered = io.BytesIO()
        image_path.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Enhanced function to label objects with coordinates from gemini_spatial.py
@retry.Retry(predicate=retry.if_exception_type(Exception))
def detect_objects_with_gemini(image_path):
    """
    Use Gemini to detect objects in an image with enhanced spatial understanding
    Returns a list of detected objects with their descriptions
    """
    try:
        # Read the image
        img = Image.open(image_path)
        
        # Use the improved prompt format from gemini_spatial.py
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
        
        # Generate response from Gemini
        response = model.generate_content([prompt, img])
        response_text = response.text
        
        # Extract JSON data from the response
        try:
            # Parse the JSON response
            # First, try to find a JSON array in the response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                labels = json.loads(json_str)
            else:
                labels = json.loads(response_text)
            
            # Convert to BoundingBox objects
            bounding_boxes = []
            for item in labels:
                if 'object' in item and 'coordinates' in item:
                    try:
                        # Extract coordinates and ensure they are within range 0-1
                        coords = item['coordinates']
                        
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
                        
                        # Convert coordinates from 0-1 range to pixel coordinates
                        img_width, img_height = img.size
                        x1_px = int(x1 * img_width)
                        y1_px = int(y1 * img_height)
                        x2_px = int(x2 * img_width)
                        y2_px = int(y2 * img_height)
                        
                        bounding_boxes.append(BoundingBox(
                            box_2d=[x1_px, y1_px, x2_px, y2_px],
                            label=item['object'].lower()  # Convert to lowercase
                        ))
                    except Exception as validation_error:
                        print(f"Error validating bounding box: {validation_error}")
                        # Skip this entry if it can't be validated
            
            return bounding_boxes
            
        except Exception as parsing_error:
            print(f"Error parsing JSON response: {str(parsing_error)}")
            print(f"Raw response: {response_text}")
            
            # Try with a more explicit follow-up prompt
            try:
                follow_up_prompt = """
                Please reformat your previous response as a valid JSON array.
                Each object should have:
                - "object": the object name as a string
                - "description": brief description as a string
                - "coordinates": [x1, y1, x2, y2] with values between 0.0 and 1.0
                
                Example:
                [
                  {"object": "bed", "description": "queen size bed with white sheets", "coordinates": [0.1, 0.3, 0.7, 0.8]},
                  {"object": "chair", "description": "wooden desk chair", "coordinates": [0.05, 0.1, 0.15, 0.25]}
                ]
                """
                follow_up_response = model.generate_content(follow_up_prompt)
                follow_up_text = follow_up_response.text.strip()
                
                # Extract JSON from the follow-up response
                json_match = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                
                if json_match >= 0 and json_end > json_match:
                    json_str = response_text[json_match:json_end]
                    labels = json.loads(json_str)
                else:
                    labels = json.loads(follow_up_text)
                
                # Convert to BoundingBox objects
                bounding_boxes = []
                for item in labels:
                    if 'object' in item and 'coordinates' in item:
                        try:
                            # Process coordinates as before
                            coords = item['coordinates']
                            # Convert to percentages if needed and ensure within bounds
                            if any(c > 1.0 for c in coords):
                                coords = [c/100 if c > 1.0 else c for c in coords]
                            coords = [max(0.0, min(c, 1.0)) for c in coords]
                            
                            # Ensure x1 < x2 and y1 < y2
                            x1, y1, x2, y2 = coords
                            if x1 > x2:
                                x1, x2 = x2, x1
                            if y1 > y2:
                                y1, y2 = y2, y1
                            
                            # Convert to pixel coordinates
                            img_width, img_height = img.size
                            x1_px = int(x1 * img_width)
                            y1_px = int(y1 * img_height)
                            x2_px = int(x2 * img_width)
                            y2_px = int(y2 * img_height)
                            
                            bounding_boxes.append(BoundingBox(
                                box_2d=[x1_px, y1_px, x2_px, y2_px],
                                label=item['object'].lower()
                            ))
                        except:
                            # Skip invalid entries
                            pass
                return bounding_boxes
            except Exception as e:
                print(f"Error in follow-up response: {str(e)}")
                # Return empty list as last resort
                return []
            
    except Exception as e:
        print(f"Error in Gemini detection: {str(e)}")
        return []

# Improved function to draw labeled objects on image with matplotlib
def plot_bounding_boxes(image, bounding_boxes):
    """
    Plots bounding boxes on an image with better styling from gemini_spatial.py
    
    Args:
        image: The image to draw on (CV2 or PIL format)
        bounding_boxes: A list of BoundingBox objects containing labels and coordinates
    
    Returns:
        CV2 image with bounding boxes drawn
    """
    # Convert CV2 image to PIL if needed
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = image
        
    # Convert PIL Image to numpy array for matplotlib
    img_array = np.array(pil_image)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_array)
    
    # Use a consistent blue color for all objects
    box_color = '#0066ff'  # Bright blue
    
    # Draw each object with its label
    for i, bbox in enumerate(bounding_boxes):
        # Get coordinates
        x1, y1, x2, y2 = bbox.box_2d
        
        # Draw rectangle with blue color and no fill
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor=box_color, facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add label with blue background at the top of the box
        text_bg = dict(boxstyle="round,pad=0.3", fc=box_color, ec=box_color, alpha=0.8)
        ax.text(x1, y1-5, bbox.label, color='white', fontweight='bold',
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
    
    # Convert back to cv2 format for our pipeline
    processed_image = Image.open(buf)
    return cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)

@csrf_exempt
def process_image(request):
    if model is None:
        return JsonResponse({
            'success': False,
            'error': 'Gemini model not loaded. Please check server configuration.'
        })
        
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Get the detection type from the form
            detection_type = request.POST.get('detection_type', 'checkin')
            session_id = request.POST.get('session_id', '')
            room_number = request.POST.get('room_number', '')
            staff_member_id = request.POST.get('staff_member_id', None)
            
            # If no session_id is provided and this is a checkin image, create a new one
            if not session_id and detection_type == 'checkin':
                session_id = str(uuid.uuid4())
            
            image_file = request.FILES['image']
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', image_file.name)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)
            
            # Read image
            image = cv2.imread(temp_path)
            original_image = image.copy()
            
            # Run detection with Gemini's spatial understanding
            bounding_boxes = detect_objects_with_gemini(temp_path)
            
            # Use our new plotting function to draw bounding boxes
            if bounding_boxes:
                original_image = plot_bounding_boxes(original_image, bounding_boxes)
            
            # Convert final image to base64
            _, buffer = cv2.imencode('.jpg', original_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_str = base64.b64encode(buffer).decode()
            img_data = ContentFile(buffer)
            
            # Convert BoundingBox objects to dictionary form for database storage
            detections = []
            for bbox in bounding_boxes:
                detections.append({
                    'class': bbox.label,
                    'box': bbox.box_2d,
                    'confidence': 0.9  # Default high confidence
                })
            
            # Save detection to database
            detection = Detection(
                detection_type=detection_type,
                session_id=session_id,
                room_number=room_number
            )
            
            # If staff_member_id is provided, associate with staff
            if staff_member_id:
                from staff.models import StaffMember
                try:
                    staff_member = StaffMember.objects.get(id=staff_member_id)
                    detection.staff_member = staff_member
                except StaffMember.DoesNotExist:
                    pass
            
            detection.image.save(f"{session_id}_{detection_type}_{uuid.uuid4()}.jpg", image_file)
            detection.processed_image.save(f"{session_id}_{detection_type}_processed_{uuid.uuid4()}.jpg", img_data)
            detection.set_detection_data(detections)
            detection.save()
            
            # Prepare response data
            response_data = {
                'success': True,
                'image': img_str,
                'detections': detections,
                'session_id': session_id,
                'detection_id': detection.id
            }
            
            # Check inventory at check-in time
            if detection_type == 'checkin' and staff_member_id:
                from staff.models import RoomActivity
                from manager.models import Room, RoomInventory
                
                try:
                    # Get the room activity
                    room_activity = RoomActivity.objects.get(
                        yolo_session_id=session_id,
                        staff_member_id=staff_member_id
                    )
                    
                    # Get room inventory if there is a linked room
                    if room_activity.room:
                        # Get expected inventory from database
                        room_inventory = {}
                        inventory_items = room_activity.room.inventory_items.all()
                        for item in inventory_items:
                            room_inventory[item.item_name.lower()] = item.quantity
                        
                        # Get detected objects
                        detected_objects = {}
                        for item in detections:
                            class_name = item['class'].lower()
                            if class_name not in detected_objects:
                                detected_objects[class_name] = 0
                            detected_objects[class_name] += 1
                        
                        # Compare expected vs. detected
                        inventory_comparison = {}
                        
                        # Check each inventory item
                        for item_name, expected_count in room_inventory.items():
                            detected = False
                            actual_count = 0
                            
                            # Check for exact matches or similar items
                            for det_name, det_count in detected_objects.items():
                                if item_name in det_name or det_name in item_name:
                                    detected = True
                                    actual_count = det_count
                                    break
                            
                            # A match is valid only if we detected at least the expected quantity
                            match = actual_count >= expected_count
                            
                            inventory_comparison[item_name] = {
                                'expected': expected_count,
                                'detected': actual_count,
                                'match': match
                            }
                        
                        # Add missing items that weren't in inventory
                        for det_name, det_count in detected_objects.items():
                            found = False
                            for item_name in room_inventory.keys():
                                if item_name in det_name or det_name in item_name:
                                    found = True
                                    break
                            
                            if not found:
                                inventory_comparison[det_name] = {
                                    'expected': 0,
                                    'detected': det_count,
                                    'match': False,
                                    'extra': True
                                }
                        
                        # Add inventory comparison to response
                        response_data['inventory_comparison'] = inventory_comparison
                        
                except (RoomActivity.DoesNotExist, Room.DoesNotExist):
                    pass
            
            # Update RoomActivity if this is a checkout image
            elif detection_type == 'checkout' and staff_member_id:
                from staff.models import RoomActivity
                try:
                    # Get the latest room activity for this session
                    room_activity = RoomActivity.objects.get(
                        yolo_session_id=session_id,
                        staff_member_id=staff_member_id
                    )
                    
                    # Compare checkin and checkout detections
                    try:
                        checkin_detection = Detection.objects.filter(
                            session_id=session_id, 
                            detection_type='checkin'
                        ).latest('timestamp')
                        
                        checkin_objects = {}
                        for item in checkin_detection.get_detection_data():
                            if 'class' in item:
                                class_name = item['class'].lower()
                                if class_name not in checkin_objects:
                                    checkin_objects[class_name] = 0
                                checkin_objects[class_name] += 1
                        
                        checkout_objects = {}
                        for item in detections:
                            if 'class' in item:
                                class_name = item['class'].lower()
                                if class_name not in checkout_objects:
                                    checkout_objects[class_name] = 0
                                checkout_objects[class_name] += 1
                        
                        # Find missing objects
                        missing_objects = {}
                        for obj, count in checkin_objects.items():
                            checkout_count = checkout_objects.get(obj, 0)
                            if checkout_count < count:
                                missing_objects[obj] = count - checkout_count
                        
                        # Update room activity with missing items
                        if missing_objects:
                            room_activity.has_missing_items = True
                            # Ensure proper dictionary format
                            print(f"SAVING MISSING ITEMS: {missing_objects}")
                            room_activity.missing_items_details = json.dumps(missing_objects)
                            room_activity.save()
                            print(f"SAVED FORMAT: {room_activity.missing_items_details}")
                    except Detection.DoesNotExist:
                        pass
                    
                except RoomActivity.DoesNotExist:
                    pass
            
            # Clean up - with error handling
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_path}: {str(e)}")
                # Continue processing instead of failing
            
            return JsonResponse(response_data)
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return JsonResponse({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            })
    
    return JsonResponse({
        'success': False,
        'error': 'No image file provided'
    })

def upload_page(request):
    return render(request, 'yolo/yolo/upload.html')

def compare_detections(request, session_id=None):
    if not session_id and request.GET.get('session_id'):
        session_id = request.GET.get('session_id')
        
    context = {
        'session_id': session_id
    }
    
    if session_id:
        # Get the checkin and checkout detections for this session
        try:
            checkin_detection = Detection.objects.filter(session_id=session_id, detection_type='checkin').latest('timestamp')
            context['checkin_detection'] = checkin_detection
            context['has_checkin'] = True
            context['room_number'] = checkin_detection.room_number
        except Detection.DoesNotExist:
            context['has_checkin'] = False
            
        try:
            checkout_detection = Detection.objects.filter(session_id=session_id, detection_type='checkout').latest('timestamp')
            context['checkout_detection'] = checkout_detection
            context['has_checkout'] = True
        except Detection.DoesNotExist:
            context['has_checkout'] = False
            
        # If we have both, compare them
        if context.get('has_checkin') and context.get('has_checkout'):
            checkin_data = checkin_detection.get_detection_data()
            checkout_data = checkout_detection.get_detection_data()
            
            # Create a dictionary of objects detected in checkin image
            checkin_objects = {}
            for item in checkin_data:
                if 'class' in item:
                    class_name = item['class'].lower()
                    if class_name not in checkin_objects:
                        checkin_objects[class_name] = 0
                    checkin_objects[class_name] += 1
                
            # Create a dictionary of objects detected in checkout image
            checkout_objects = {}
            for item in checkout_data:
                if 'class' in item:
                    class_name = item['class'].lower()
                    if class_name not in checkout_objects:
                        checkout_objects[class_name] = 0
                    checkout_objects[class_name] += 1
                
            # Find missing objects (in checkin but not in checkout)
            missing_objects = {}
            for obj, count in checkin_objects.items():
                checkout_count = checkout_objects.get(obj, 0)
                if checkout_count < count:
                    missing_objects[obj] = count - checkout_count
            
            context['checkin_objects'] = checkin_objects
            context['checkout_objects'] = checkout_objects
            context['missing_objects'] = missing_objects
            context['comparison_complete'] = True
            
            # If staff_member is present, we can link to the room activity
            if checkin_detection.staff_member:
                from staff.models import RoomActivity
                try:
                    room_activity = RoomActivity.objects.get(
                        yolo_session_id=session_id,
                        staff_member=checkin_detection.staff_member
                    )
                    context['room_activity'] = room_activity
                    
                    # Update room activity with missing items
                    if missing_objects:
                        room_activity.has_missing_items = True
                        # Ensure proper dictionary format
                        print(f"SAVING MISSING ITEMS: {missing_objects}")
                        room_activity.missing_items_details = json.dumps(missing_objects)
                        room_activity.save()
                        print(f"SAVED FORMAT: {room_activity.missing_items_details}")
                except RoomActivity.DoesNotExist:
                    pass
    
    return render(request, 'yolo/yolo/compare.html', context)

@csrf_exempt
def get_comparison_json(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        
        if not session_id:
            return JsonResponse({'success': False, 'error': 'No session ID provided'})
            
        try:
            shelf_detection = Detection.objects.filter(session_id=session_id, detection_type='shelf').latest('timestamp')
            checkout_detection = Detection.objects.filter(session_id=session_id, detection_type='checkout').latest('timestamp')
            
            shelf_data = shelf_detection.get_detection_data()
            checkout_data = checkout_detection.get_detection_data()
            
            # Create a dictionary of objects detected in shelf image
            shelf_objects = {}
            for item in shelf_data:
                if item['class'] not in shelf_objects:
                    shelf_objects[item['class']] = 0
                shelf_objects[item['class']] += 1
                
            # Create a dictionary of objects detected in checkout image
            checkout_objects = {}
            for item in checkout_data:
                if item['class'] not in checkout_objects:
                    checkout_objects[item['class']] = 0
                checkout_objects[item['class']] += 1
                
            # Find missing objects (in shelf but not in checkout)
            missing_objects = {}
            for obj, count in shelf_objects.items():
                checkout_count = checkout_objects.get(obj, 0)
                if checkout_count < count:
                    missing_objects[obj] = count - checkout_count
            
            return JsonResponse({
                'success': True,
                'shelf_objects': shelf_objects,
                'checkout_objects': checkout_objects,
                'missing_objects': missing_objects,
                'shelf_image': shelf_detection.processed_image.url if shelf_detection.processed_image else None,
                'checkout_image': checkout_detection.processed_image.url if checkout_detection.processed_image else None
            })
            
        except Detection.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Detection not found'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    return JsonResponse({'success': False, 'error': 'Invalid request method'})
