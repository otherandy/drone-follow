import cv2
from ultralytics import YOLO

# Load the model
yolo = YOLO("best.pt")

# Load the video capture
videoCap = cv2.VideoCapture(0)
frameWidth = 640
frameHeight = 480
videoCap.set(3, frameWidth)
videoCap.set(4, frameHeight)
deadZone = 100


# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        base_colors[color_index][i]
        + increments[color_index][i] * (cls_num // len(base_colors)) % 256
        for i in range(3)
    ]
    return tuple(color)


while True:
    ret, frame = videoCap.read()

    if not ret:
        continue

    results = yolo.track(frame, stream=True)

    for result in results:
        # get the classes names
        classes_names = result.names
        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.6:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # get the class
                cls = int(box.cls[0])
                # get the class name
                class_name = classes_names[cls]
                # get the respective colour
                colour = getColours(cls)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.circle(frame, (cx, cy), 10, colour, 3)
                # put the class name and confidence on the image
                cv2.putText(
                    frame,
                    f"{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}",
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    colour,
                    2,
                )
                if cx < int(frameWidth / 2) - deadZone:
                    cv2.putText(
                        frame,
                        " Izquierda ",
                        (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                    cv2.rectangle(
                        frame,
                        (0, int(frameHeight / 2 - deadZone)),
                        (
                            int(frameWidth / 2) - deadZone,
                            int(frameHeight / 2) + deadZone,
                        ),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                elif cx > int(frameWidth / 2) + deadZone:
                    cv2.putText(
                        frame,
                        " Derecha ",
                        (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                    cv2.rectangle(
                        frame,
                        (
                            int(frameWidth / 2 + deadZone),
                            int(frameHeight / 2 - deadZone),
                        ),
                        (frameWidth, int(frameHeight / 2) + deadZone),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                elif cy < int(frameHeight / 2) - deadZone:
                    cv2.putText(
                        frame,
                        " Arriba ",
                        (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                    cv2.rectangle(
                        frame,
                        (int(frameWidth / 2 - deadZone), 0),
                        (
                            int(frameWidth / 2 + deadZone),
                            int(frameHeight / 2) - deadZone,
                        ),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                elif cy > int(frameHeight / 2) + deadZone:
                    cv2.putText(
                        frame,
                        " Abajo ",
                        (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                    cv2.rectangle(
                        frame,
                        (
                            int(frameWidth / 2 - deadZone),
                            int(frameHeight / 2) + deadZone,
                        ),
                        (int(frameWidth / 2 + deadZone), frameHeight),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                cv2.line(
                    frame,
                    (int(frameWidth / 2), int(frameHeight / 2)),
                    (cx, cy),
                    (0, 0, 255),
                    3,
                )

    # show the image
    cv2.imshow("frame", frame)
    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# release the video capture and destroy all windows

videoCap.release()

cv2.destroyAllWindows()
