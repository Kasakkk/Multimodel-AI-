import cv2

def capture_image(output_path: str = "current_view.jpg") -> str:
    """
    Captures a frame from the primary webcam and saves it to output_path.
    Raises:
        RuntimeError: If the webcam cannot be initialized or a frame cannot be read.
    """
    # 0 is the ID for the default camera attached to the machine
    capture = cv2.VideoCapture(0)
    
    if not capture.isOpened():
        raise RuntimeError("Error: Camera could not be opened. Please check physical or software connections.")
    
    # Read a single frame
    ret, frame = capture.read()
    
    # Release the camera immediately after capturing to free system resources
    capture.release()
    
    if not ret:
        raise RuntimeError("Error: Failed to fetch a frame from the camera.")
    
    # Save the obtained frame to disk
    cv2.imwrite(output_path, frame)
    return output_path

if __name__ == "__main__":
    # Small utility test block
    try:
        path = capture_image("test_cam.jpg")
        print(f"Image successfully captured and saved to {path}")
    except Exception as e:
        print(e)
