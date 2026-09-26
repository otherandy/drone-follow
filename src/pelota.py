import cv2
from djitellopy import Tello
from ultralytics import YOLO

FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

SPEED: int = 25
DEADZONE: int = 60

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

                        lr, fb = 0, 0

                        x_error = x_center - FRAME_WIDTH / 2
                        y_error = y_center - FRAME_HEIGHT / 2

                        if abs(x_error) > DEADZONE:
                            lr = int(x_error / (FRAME_WIDTH / 2 - DEADZONE) * SPEED)
                        if abs(y_error) > DEADZONE:
                            fb = int(y_error / (FRAME_HEIGHT / 2 - DEADZONE) * SPEED)

                        lr = max(-SPEED, min(SPEED, lr))
                        fb = max(-SPEED, min(SPEED, fb))

                        tello.send_rc_control(lr, fb, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        return frame
            else:
                tello.send_rc_control(0, 0, 0, 0)

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
