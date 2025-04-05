from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
import os
import uuid
import json
from django.conf import settings
import base64
import cv2
import torch
import numpy as np
from .models import Detection
from django.core.files.base import ContentFile

# Initialize YOLO model
try:
    model = YOLO('yolov8x.pt')  # Load YOLOv8x model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Loaded YOLOv8x model on {device}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@csrf_exempt
def process_image(request):
    if model is None:
        return JsonResponse({
            'success': False,
            'error': 'Model not loaded. Please check server configuration.'
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
            
            # Run detection
            results = model(
                image,
                conf=0.25,  # Lower confidence threshold for better detection
                iou=0.45,   # IOU threshold
                max_det=20  # Maximum detections per image
            )
            
            # Define colors for visualization
            colors = {
                'bed': (0, 255, 0),      # Green
                'chair': (255, 0, 0),    # Blue
                'tv': (0, 0, 255),       # Red
                'couch': (128, 0, 128),  # Purple
                'dining table': (255, 0, 255),  # Magenta
                'laptop': (255, 165, 0),  # Orange
                'remote': (128, 128, 0),  # Olive
                'clock': (0, 128, 128),   # Teal
                'book': (70, 130, 180),   # Steel Blue
                'vase': (219, 112, 147)   # Pale Violet Red
            }
            
            # Process detections
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    
                    if class_name.lower() in colors:
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'class': class_name,
                            'confidence': conf
                        })
                        
                        # Draw detection
                        color = colors[class_name.lower()]
                        label = f"{class_name} {conf:.2f}"
                        
                        # Draw box
                        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw label
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(original_image, (x1, y1 - 35), (x1 + w + 10, y1), color, -1)
                        cv2.putText(original_image, label, (x1 + 5, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Convert final image to base64
            _, buffer = cv2.imencode('.jpg', original_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_str = base64.b64encode(buffer).decode()
            img_data = ContentFile(buffer)
            
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
            
            # Update RoomActivity if this is a checkout image
            if detection_type == 'checkout' and staff_member_id:
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
                            if item['class'] not in checkin_objects:
                                checkin_objects[item['class']] = 0
                            checkin_objects[item['class']] += 1
                        
                        checkout_objects = {}
                        for item in detections:
                            if item['class'] not in checkout_objects:
                                checkout_objects[item['class']] = 0
                            checkout_objects[item['class']] += 1
                        
                        # Find missing objects
                        missing_objects = {}
                        for obj, count in checkin_objects.items():
                            checkout_count = checkout_objects.get(obj, 0)
                            if checkout_count < count:
                                missing_objects[obj] = count - checkout_count
                        
                        # Update room activity with missing items
                        if missing_objects:
                            room_activity.has_missing_items = True
                            room_activity.missing_items_details = json.dumps(missing_objects)
                            room_activity.save()
                    except Detection.DoesNotExist:
                        pass
                    
                except RoomActivity.DoesNotExist:
                    pass
            
            # Clean up
            os.remove(temp_path)
            
            return JsonResponse({
                'success': True,
                'image': img_str,
                'detections': detections,
                'session_id': session_id,
                'detection_id': detection.id
            })
            
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
                if item['class'] not in checkin_objects:
                    checkin_objects[item['class']] = 0
                checkin_objects[item['class']] += 1
                
            # Create a dictionary of objects detected in checkout image
            checkout_objects = {}
            for item in checkout_data:
                if item['class'] not in checkout_objects:
                    checkout_objects[item['class']] = 0
                checkout_objects[item['class']] += 1
                
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
                        room_activity.missing_items_details = json.dumps(missing_objects)
                        room_activity.save()
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
