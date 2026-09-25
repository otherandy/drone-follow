import cv2
from djitellopy import Tello
from ultralytics import YOLO

FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

DIST: int = 25
DEADZONE: int = 100

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
                        x_center = int(x1 + x2) / 2
                        y_center = int(y1 + y2) / 2
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        tello.send_rc_control(0, 0, 0, 0)

                        if x_center < int(FRAME_WIDTH / 2) - DEADZONE:
                            tello.move_left(DIST)
                        elif x_center > int(FRAME_WIDTH / 2) + DEADZONE:
                            tello.move_right(DIST)

                        if y_center < int(FRAME_HEIGHT / 2) - DEADZONE:
                            tello.move_back(DIST)
                        elif y_center > int(FRAME_HEIGHT / 2) + DEADZONE:
                            tello.move_forward(DIST)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


def __main__():
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    tello.takeoff()

    try:
        while True:
            frame = tracking_loop(tello)
            cv2.imshow("drone", frame)
            keydown = cv2.waitKey(1) & 0xFF
            if keydown == ord("q"):
                tello.land()
                break
            if keydown == ord("e"):
                tello.emergency()
                break

    finally:
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    __main__()
