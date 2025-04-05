from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
import os
from django.conf import settings
import base64
import cv2
import torch
import numpy as np

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
            
            # Clean up
            os.remove(temp_path)
            
            return JsonResponse({
                'success': True,
                'image': img_str,
                'detections': detections
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
