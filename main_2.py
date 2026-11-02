import cv2
import datetime
import time
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

target_classes = [0, 2, 67]

class_names = {
    0: "Person",
    2: "Car",
    67: "Phone"
}


os.makedirs("video", exist_ok=True)
os.makedirs("photo", exist_ok=True)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_frame_time = 0

recording = False
out = None



while True:

    ret, frame = cap.read()

    if not ret:
        break



    results = model(
        frame,classes=target_classes,verbose=False)

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int,box.xyxy[0].tolist())

            class_id = int(box.cls[0])

            confidence = float(box.conf[0]) * 100

            name = class_names.get(class_id,"Unknown")

            cv2.rectangle(frame,(x1, y1), (x2, y2),(0, 255, 0),2)

            label = f"{name} {confidence:.1f}%"

            cv2.putText(frame,label,(x1, max(y1 - 10, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)

    current_time = datetime.datetime.now()

    date_text = current_time.strftime(
        "%Y-%m-%d"
    )

    time_text = current_time.strftime(
        "%H:%M:%S"
    )

    cv2.putText(frame, "MOTION WEB", (20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),2)

    cv2.putText(frame,f"DATE: {date_text}",(20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)

    cv2.putText(frame,f"TIME: {time_text}",(20, 115),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)

    new_frame_time = time.time()

    fps = 0

    if prev_frame_time > 0:
        fps = 1 / max(new_frame_time - prev_frame_time,0.001)

    prev_frame_time = new_frame_time

    cv2.putText(frame,f"FPS: {int(fps)}",(20, 150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2)

    if recording:

        cv2.circle(frame,(30, 190),8,(0, 0, 255),-1)

        cv2.putText(frame,"RECORDING",(50, 198),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2)

        out.write(frame)

    cv2.imshow("frame",frame)
    key = cv2.waitKey(1) & 0xFF


    if key == ord("v") or key == ord("V"):

        if not recording:

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = os.path.join(
                "video",
                f"video_{timestamp}.mp4"
            )

            height, width = frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(filename,fourcc,20.0,(width, height))

            if out.isOpened():

                recording = True


            else:

                out.release()
                out = None


        else:

            recording = False

            out.release()
            out = None



    elif key == ord("s") or key == ord("S"):

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = os.path.join("photo",f"photo_{timestamp}.jpg")

        cv2.imwrite(filename, frame)



    elif key == ord("q") or key == ord("Q"):

        break


if out is not None:
    out.release()

cap.release()

cv2.destroyAllWindows()

