import cv2
import numpy as np
import tensorflow as tf

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the labels file safely
try:
    class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
    print("✅ Labels loaded successfully!")
except Exception as e:
    print(f"❌ Error loading labels.txt: {e}")
    print("Make sure 'labels.txt' is in the same folder as this script.")
    exit()

# Load the TFLite model and allocate tensor buffers
try:
    # MATCHES YOUR EXACT FILENAME FROM VS CODE: model_unquant.tflite
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    
    # Get model input and output structures
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ TFLite Engine initialized successfully! No Keras errors found.")
except Exception as e:
    print(f"❌ Error loading model_unquant.tflite: {e}")
    print("Please make sure 'model_unquant.tflite' is in your project folder.")
    exit()

# CAMERA can be 0 or 1 based on your computer's setup
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Error: Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1).")
    exit()

print("🎥 Starting webcam stream... Press 'ESC' on the video window to exit.")

while True:
    # Grab the webcamera's image
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Clone the original frame to draw on, so we don't distort our AI's input
    display_frame = frame.copy()

    # Pre-process image for TFLite model constraints: Resize to (224, 224)
    input_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    input_image = np.asarray(input_image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize image array tensors exactly like Teachable Machine expects
    input_image = (input_image / 127.5) - 1

    # Feed input data to the TFLite runtime tensor buffer
    interpreter.set_tensor(input_details[0]['index'], input_image)
    
    # Run inference
    interpreter.invoke()

    # Extract the prediction probabilities array
    prediction = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(prediction)
    
    # Extract clean text components
    raw_class = class_names[index]
    # Removes index prefix if Teachable Machine includes it (e.g., "0 Paper" -> "Paper")
    class_name = raw_class[2:] if raw_class[0].isdigit() else raw_class
    confidence_score = prediction[0][index]
    conf_percent = int(np.round(confidence_score * 100))

    # --- VISUAL UI OVERLAY ON WEBCAM ---
    # Create a translucent background bar for the text overlay
    cv2.rectangle(display_frame, (0, 0), (450, 60), (0, 0, 0), -1)
    
    # Choose overlay color based on confidence (Green for strong match, Orange for weak match)
    text_color = (0, 255, 0) if conf_percent > 70 else (0, 165, 255)
    
    # Render Classification outcome directly on the desktop frame window
    ui_text = f"Class: {class_name} ({conf_percent}%)"
    cv2.putText(display_frame, ui_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)

    # Show the interactive UI window
    cv2.imshow("Waste Classification AI", display_frame)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)
    
    # 27 is the ASCII key code for the Escape ('ESC') key
    if keyboard_input == 27:  
        break

# Clean up system buffers completely
camera.release()
cv2.destroyAllWindows()
print("👋 Camera pipeline closed down cleanly.")