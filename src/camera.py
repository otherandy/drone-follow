import cv2
from djitellopy import Tello
from ultralytics import YOLO

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SPEED = 20
DESIRED_HEIGHT = 130

yolo = YOLO("best.pt")


def tracking_loop(tello):
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = yolo.track(frame)

    if results:
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    if box.conf[0] > 0.6:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        if x_center < FRAME_WIDTH * 0.3:
                            lr = -SPEED
                        elif x_center > FRAME_WIDTH * 0.7:
                            lr = SPEED

                        if y_center < FRAME_HEIGHT * 0.3:
                            fb = SPEED
                        elif y_center > FRAME_HEIGHT * 0.7:
                            fb = -SPEED

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


def __main__():
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()

    try:
        while True:
            frame = tracking_loop(tello)
            cv2.imshow("drone", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tello.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    __main__()
