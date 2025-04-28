import face_recognition
import cv2
import numpy as np
import datetime
import csv
from PIL import Image, ImageDraw

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
your_image = face_recognition.load_image_file("image.jpg")
your_face_encoding = face_recognition.face_encodings(your_image)[0]

face_landmarks_list = face_recognition.face_landmarks(your_image)

# Create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(your_image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:
    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)

# Show the picture
pil_image.show()

# Create arrays of known face encodings and their names
known_face_encodings = [
    your_face_encoding,
]
known_face_names = [
    "Your Name",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def determine_expression(face_landmarks):
    bottom_lip = face_landmarks['bottom_lip']
    
    
    # Calculate average y-coordinate of the bottom lip points
    average_y = np.mean([point[1] for point in bottom_lip])
    
    # Extract the y-coordinates of the left, center, and right points of the bottom lip
    left_corner_y = bottom_lip[0][1]
    right_corner_y = bottom_lip[-1][1]
    center_y = bottom_lip[5][1]  # typically the central bottom lip point
    
    # Calculate the y-difference between corners and the center
    left_diff = average_y - left_corner_y
    right_diff = average_y - right_corner_y
    center_diff = center_y - average_y
    
    # Determine if it's a smile or frown based on the differences
    if center_diff > 0 and left_diff < 0 and right_diff < 0:
        return "Subject is Frowning (Angry)"
    elif center_diff < 0 and left_diff > 0 and right_diff > 0:
        return "Subject is Smiling (Happy)"
    else:
        return "Neutral Expression"

# Determine expression
expression = determine_expression(face_landmarks)
print(expression)

# Create/open an attendance file in write mode
with open("attendance.csv", "a", newline='') as attendance_file:
    fieldnames = ['Name', 'Timestamp']
    writer = csv.DictWriter(attendance_file, fieldnames=fieldnames)
    
    # Write the header only if the file is empty
    if attendance_file.tell() == 0:
        writer.writeheader()

    # Set to keep track of dates for which attendance has been recorded
    recorded_dates = set()

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    
                    # Write to the attendance file only if the date hasn't been recorded yet
                    now = datetime.datetime.now()
                    date = now.date()
                    if date not in recorded_dates:
                        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                        writer.writerow({'Name': name, 'Timestamp': timestamp})
                        recorded_dates.add(date)

                face_names.append(name)

            # Process facial expressions
            for face_landmarks in face_landmarks_list:
                expression = determine_expression(face_landmarks)
                print(expression)  # print to console for now, but you can display it on the frame if needed

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Check if it's a match, if so, print "Match." on the top right
            if name != "Unknown":
                cv2.putText(frame, "Match.", (10, 30), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
