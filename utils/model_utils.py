import os
import numpy as np
import cv2
import tensorflow as tf

def print_model_details(interpreter):
    """Print detailed information about the model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n=== MODEL DETAILS ===")
    print("\nINPUT DETAILS:")
    for i, detail in enumerate(input_details):
        print(f"Input #{i}")
        print(f"  name: {detail['name']}")
        print(f"  shape: {detail['shape']}")
        print(f"  dtype: {detail['dtype']}")
        print(f"  quantization: {detail.get('quantization', 'none')}")
    
    print("\nOUTPUT DETAILS:")
    for i, detail in enumerate(output_details):
        print(f"Output #{i}")
        print(f"  name: {detail['name']}")
        print(f"  shape: {detail['shape']}")
        print(f"  dtype: {detail['dtype']}")
        print(f"  quantization: {detail.get('quantization', 'none')}")
    
    print("\n=========================\n")

def load_tflite_model(model_path):
    """Load the TFLite model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    return interpreter

def process_image(image_path, target_size=(640, 640)):
    """Process an image for TFLite model input
    
    Args:
        image_path: Path to the input image
        target_size: Target size for the model input (640x640)
        
    Returns:
        Processed image as numpy array
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    print(f"Original image shape: {img.shape}")
    
    # Resize image to target size
    img = cv2.resize(img, target_size)
    print(f"Resized image shape: {img.shape}")
    
    # Convert to RGB (TFLite models typically expect RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Expand dimensions to match model input shape [1, height, width, channels]
    img = np.expand_dims(img, axis=0)
    print(f"Final input shape: {img.shape}")
    
    return img

def detect_defects_tflite(interpreter, image, classes):
    """Run defect detection on the image using TFLite model
    
    Args:
        interpreter: TFLite interpreter
        image: Processed image array
        classes: List of class names
        
    Returns:
        Dictionary with detection results
    """
    try:
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set the tensor to point to the input data
        interpreter.set_tensor(input_details[0]['index'], image)
        
        # Run inference
        interpreter.invoke()
        
        # Get output data
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"Output shape: {output_data.shape}")
        
        # For YOLO v8 TFLite, output format is [1, 10, 8400]
        # This is different from typical YOLO formats!
        predictions = []
        
        # Get the output
        output = output_data[0]  # Remove batch dimension - now [10, 8400]
        
        # YOLOv8 TFLite has a unique format where:
        # Rows 0-3: normalized box coordinates (cx, cy, w, h)
        # Rows 4-9: class scores for each class (6 classes in your case)
        # Row 4: objectness * class_0_probability
        # Row 5: objectness * class_1_probability  
        # Row 6: objectness * class_2_probability
        # Row 7: objectness * class_3_probability
        # Row 8: objectness * class_4_probability
        # Row 9: objectness * class_5_probability
        
        # Debug prints
        print(f"Row 0 (cx) stats - min: {np.min(output[0]):.6f}, max: {np.max(output[0]):.6f}")
        print(f"Row 4 (class scores) stats - min: {np.min(output[4]):.6f}, max: {np.max(output[4]):.6f}")
        
        # Find high confidence detections across all class rows (4-9)
        all_scores = output[4:10, :]  # Get all class scores
        max_scores_per_detection = np.max(all_scores, axis=0)  # Max score for each detection
        class_indices = np.argmax(all_scores, axis=0)  # Which class has the max score
        
        # Set confidence threshold
        conf_threshold = 0.25
        
        # Find detections above threshold
        valid_indices = np.where(max_scores_per_detection > conf_threshold)[0]
        
        print(f"Number of valid detections (confidence > {conf_threshold}): {len(valid_indices)}")
        
        # Process each valid detection
        for idx in valid_indices:
            # Extract box coordinates (YOLO format - normalized)
            cx = float(output[0, idx])  # center x
            cy = float(output[1, idx])  # center y
            w = float(output[2, idx])   # width
            h = float(output[3, idx])   # height
            
            # Get the best class and its score
            class_id = int(class_indices[idx])
            confidence = float(max_scores_per_detection[idx])
            
            # Debug print for first few detections
            if len(predictions) < 5:
                print(f"Detection {idx}: class={class_id}, confidence={confidence:.4f}, cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}")
            
            # Convert from relative coordinates to pixel coordinates
            img_width, img_height = 640, 640
            
            # Convert from center to corner coordinates
            x1 = (cx - w/2) * img_width
            y1 = (cy - h/2) * img_height
            x2 = (cx + w/2) * img_width
            y2 = (cy + h/2) * img_height
            
            # Ensure coordinates are within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Skip tiny boxes
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue
                
            class_name = classes[class_id] if class_id < len(classes) else f"unknown_{class_id}"
            
            predictions.append({
                'defect_type': class_name,
                'confidence': confidence,
                'location': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        # Apply Non-Maximum Suppression (NMS) to remove duplicate detections
        if len(predictions) > 0:
            predictions = apply_nms(predictions, iou_threshold=0.5)
        
        # Sort by confidence
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Print the number of predictions
        print(f"Found {len(predictions)} predictions after NMS")
        
        return {
            'defects_found': len(predictions) > 0,
            'predictions': predictions
        }
        
    except Exception as e:
        print(f"Error in detect_defects_tflite: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a safe default
        return {
            'defects_found': False,
            'predictions': [],
            'error': str(e)
        }

def apply_nms(predictions, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if len(predictions) == 0:
        return []
    
    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    selected_predictions = []
    indices_to_skip = set()
    
    for i in range(len(predictions)):
        if i in indices_to_skip:
            continue
            
        selected_predictions.append(predictions[i])
        box_i = predictions[i]['location']
        
        for j in range(i + 1, len(predictions)):
            if j in indices_to_skip:
                continue
                
            box_j = predictions[j]['location']
            
            # Calculate IoU
            iou = calculate_iou(box_i, box_j)
            
            # If IoU is high and same class, skip the lower confidence box
            if (iou > iou_threshold and 
                predictions[i]['defect_type'] == predictions[j]['defect_type']):
                indices_to_skip.add(j)
    
    return selected_predictions

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two boxes"""
    x1_tl, y1_tl, x1_br, y1_br = box1
    x2_tl, y2_tl, x2_br, y2_br = box2
    
    # Calculate intersection area
    x_inter_tl = max(x1_tl, x2_tl)
    y_inter_tl = max(y1_tl, y2_tl)
    x_inter_br = min(x1_br, x2_br)
    y_inter_br = min(y1_br, y2_br)
    
    if x_inter_br < x_inter_tl or y_inter_br < y_inter_tl:
        return 0.0
    
    intersection_area = (x_inter_br - x_inter_tl) * (y_inter_br - y_inter_tl)
    
    # Calculate union area
    box1_area = (x1_br - x1_tl) * (y1_br - y1_tl)
    box2_area = (x2_br - x2_tl) * (y2_br - y2_tl)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area

def draw_defect_boxes(image_path, predictions, output_path=None):
    """
    Draw bounding boxes on the image to show detected defects
    
    Args:
        image_path: Path to the original image
        predictions: List of prediction dictionaries with defect_type, confidence, and location
        output_path: Path to save the output image (if None, will append '_result' to original)
    
    Returns:
        Path to the result image
    """
    try:
        # Create output path if not provided
        if output_path is None:
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_result{ext}"
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Get original image dimensions
        orig_height, orig_width = img.shape[:2]
        
        # Define colors for different defect types (BGR format)
        colors = {
            'missing_hole': (0, 0, 255),      # Red
            'mouse_bite': (0, 255, 0),        # Green
            'open_circuit': (255, 0, 0),      # Blue
            'short': (0, 255, 255),           # Yellow
            'spur': (255, 0, 255),            # Magenta
            'spurious_copper': (255, 255, 0)  # Cyan
        }
        
        # Default color for unknown defect types
        default_color = (255, 255, 255)  # White
        
        # Draw each prediction
        for i, pred in enumerate(predictions[:20]):  # Limit to top 20 predictions
            # Extract information
            defect_type = pred['defect_type']
            confidence = pred['confidence']
            
            # Skip if no location data
            if 'location' not in pred or pred['location'] is None:
                continue
                
            # Extract bounding box coordinates
            try:
                x_min, y_min, x_max, y_max = map(float, pred['location'])
                
                # Scale coordinates to original image size
                scale_x = orig_width / 640.0
                scale_y = orig_height / 640.0
                
                x_min = int(x_min * scale_x)
                y_min = int(y_min * scale_y)
                x_max = int(x_max * scale_x)
                y_max = int(y_max * scale_y)
                
                # Ensure coordinates are within image bounds
                x_min = max(0, min(x_min, orig_width - 1))
                y_min = max(0, min(y_min, orig_height - 1))
                x_max = max(x_min + 1, min(x_max, orig_width))
                y_max = max(y_min + 1, min(y_max, orig_height))
                
            except (ValueError, TypeError):
                continue
            
            # Get color for this defect type
            color = colors.get(defect_type, default_color)
            
            # Draw rectangle
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Create label text
            label = f"{i+1}. {defect_type}: {confidence:.2f}"
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Put label above box if there's space, otherwise below
            if y_min > text_size[1] + 10:
                label_y = y_min - 5
            else:
                label_y = y_max + 15
            
            cv2.rectangle(img, (x_min, label_y - text_size[1] - 5), 
                         (x_min + text_size[0], label_y + 5), color, -1)
            
            # Draw text
            cv2.putText(img, label, (x_min, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save the annotated image
        cv2.imwrite(output_path, img)
        
        return output_path
    
    except Exception as e:
        print(f"Error in draw_defect_boxes: {str(e)}")
        import traceback
        traceback.print_exc()
        return None