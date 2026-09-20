import cv2
from djitellopy import Tello
from ultralytics import YOLO

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SPEED = 20
DEADZONE = 100

yolo = YOLO("best.pt")


def tracking_loop(tello):
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = yolo.track(frame)

    lr = 0
    fb = 0

    if results:
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    if box.conf[0] > 0.6:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x_center = int(x1 + x2) / 2
                        y_center = int(y1 + y2) / 2
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        if x_center < int(FRAME_WIDTH / 2) - DEADZONE:
                            lr = -SPEED
                        elif x_center > int(FRAME_WIDTH / 2) + DEADZONE:
                            lr = SPEED

                        if y_center < int(FRAME_HEIGHT / 2) - DEADZONE:
                            fb = SPEED
                        elif y_center > int(FRAME_HEIGHT / 2) + DEADZONE:
                            fb = -SPEED

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    tello.send_rc_control(lr, fb, 0, 0)
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
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    __main__()
