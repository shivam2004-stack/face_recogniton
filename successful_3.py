import face_recognition
import cv2

# Load the first known image and learn how to recognize it
known_image_1 = face_recognition.load_image_file("C:/Users/lenovo/OneDrive/Desktop/apple/mystyle/known_image/shivam.jpg.jpg")
known_face_encoding_1 = face_recognition.face_encodings(known_image_1)[0]

# Load the second known image and learn how to recognize it
known_image_2 = face_recognition.load_image_file("C:/Users/lenovo/OneDrive/Desktop/apple/mystyle/known_image/prateek.jpg")
known_face_encoding_2 = face_recognition.face_encodings(known_image_2)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [known_face_encoding_1, known_face_encoding_2]
known_face_names = ["shivam", "surendra"]

# Load the pre-trained Haar Cascade classifier for face detection
face_cap = cv2.CascadeClassifier('C:/Users/lenovo/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Initialize the video capture object
video_cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, video_data = video_cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2RGB)

    # Find all the face locations and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the current frame of video
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Recognize faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # Get the best match with the smallest distance
            best_match_index = min(range(len(face_distances)), key=face_distances.__getitem__)

            name = "Unknown"
            confidence = 1 - face_distances[best_match_index]

            # Use the name of the best match if the distance is below a certain threshold
            if matches[best_match_index] and confidence > 0.4:  # Adjust threshold as needed
                name = known_face_names[best_match_index]

            # Draw a label with the name below the face
            cv2.rectangle(video_data, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(video_data, f"{name} ({confidence:.2f})", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow("video_live", video_data)

    # Break the loop on 'a' key press
    if cv2.waitKey(10) == ord("a"):
        break

# Release the video capture object and close all OpenCV windows
video_cap.release()
cv2.destroyAllWindows()
