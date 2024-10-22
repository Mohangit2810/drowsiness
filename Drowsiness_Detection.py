from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame

from twilio.rest import Client

account_sid = 'AC525fdd7e6e725b60c834c2776697f27e'
auth_token = 'a073b73479bc5e24c662506741395ccf'
client = Client(account_sid, auth_token)
twilio_phone_number = '+17745652237'
driver_phone_number = '+917904888721'  
co_driver_phone_number = '+918072822124'  
company_phone_number = '+918925413499'

pygame.mixer.init()

mild_alert = pygame.mixer.Sound('mild-alert.mp3')
medium_alert = pygame.mixer.Sound('medium-alert.mp3')
high_alert = pygame.mixer.Sound('high-alert.mp3')

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
driver_frame_check = 20  # Alert the driver after 20 frames (roughly 2/3 seconds)
co_driver_frame_check = 30  # Alert the co-driver after 30 frames (1 second)
company_frame_check = 50  # Alert the company after 50 frames (1.5 seconds)

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
driver_flag = 0
co_driver_flag = 0
company_flag = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            driver_flag += 1
            co_driver_flag += 1
            company_flag += 1

            if driver_flag >= driver_frame_check:
                message = client.messages.create(
                    body="ALERT: Driver is drowsy!",
                    from_=twilio_phone_number,
                    to=driver_phone_number
                )
                print("****************ALERT SENT TO DRIVER!****************")
                driver_flag = 0
                mild_alert.play()  # Play mild alert sound
                pygame.time.delay(3000)  # Delay for 3000 milliseconds (3 seconds)
                mild_alert.fadeout(3000) 

            if co_driver_flag >= co_driver_frame_check:
                message = client.messages.create(
                    body="ALERT: Co-driver is drowsy!",
                    from_=twilio_phone_number,
                    to=co_driver_phone_number
                )
                print("****************ALERT SENT TO CO-DRIVER!****************")
                co_driver_flag = 0
                medium_alert.play()  # Play medium alert sound
                pygame.time.delay(3000)  # Delay for 3000 milliseconds (3 seconds)
                medium_alert.fadeout(3000)  # Fade out the sound over 3000 milliseconds (3 seconds)


            if company_flag >= company_frame_check:
                message = client.messages.create(
                    body="ALERT: Driver and Co-driver are drowsy! Take action!",
                    from_=twilio_phone_number,
                    to=company_phone_number
                )
                print("****************ALERT SENT TO COMPANY!****************")
                company_flag = 0
                high_alert.play()  # Play high alert sound
                pygame.time.delay(3000)  # Delay for 3000 milliseconds (3 seconds)
                high_alert.fadeout(3000)  # Fade out the sound over 3000 milliseconds (3 seconds)


        else:
            driver_flag = 0
            co_driver_flag = 0
            company_flag = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

message = client.messages.create(
    body="ALERT: Driver and Co-driver are drowsy! Take action!",
    from_=twilio_phone_number,
    to=company_phone_number
)