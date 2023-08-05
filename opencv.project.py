import cv2
import numpy as np
import face_recognition as fr
import sys
video_capture = cv2.VideoCapture(0)
image = fr.load_image_file('C:\\Users\\EVOLUTION X DARVIN\\AppData\\Local\Programs\\Python\\Python38\Lib\\site-packages\\face_recognition-1.3.0\\tests\\test_images\\Brisk.jpg')
image_face_encoding = fr.face_encodings(image)[0]
known_face_encodings = [image_face_encoding]
known_face_names = ["DARVIN"]
while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    fc_locations = fr.face_locations(rgb_frame )
    face_encoding = fr.face_encodings(rgb_frame, fc_locations)

    for(top, right, bottom, left), face_encoding in zip(fc_locations, face_encoding):
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name="unknown"
        fc_distances = fr.face_distance(known_face_encodings, face_encoding)
        match_index = np.argmin(fc_distances)
        if matches[match_index]:
            name = known_face_names[match_index] 
            cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 2)
            cv2.rectangle(frame, (left,bottom -35), (right,bottom), (0,0,255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, name, (left +6,bottom -6), font,1.0, (255,255,255), 1)
            cv2.imshow('Simplilearn face detection system', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
video_capture.release()
cv2.destroyAllWindows()
