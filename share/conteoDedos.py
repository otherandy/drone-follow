import cv2
import mediapipe as mp
from djitellopy import Tello
import time

tello = Tello()
tello.connect()
print(f'Battery: {tello.get_battery()}%')

# Iniciar la transmisión de video desde el dron

tello.streamon()
tello.takeoff()
if tello.get_battery()<15:
    tello.land()

desired_height = 130  # La altura en centímetros

# Obtener la altura actual (en cm)
current_height = tello.get_height()

# Ajustar la altura si es necesario
while abs(current_height - desired_height) > 10:  # Permitir un margen de error
    if current_height < desired_height:
        # Subir si está por debajo
        tello.send_rc_control(0, 0, 20, 0)
    elif current_height > desired_height:
        # Bajar si está por encima
        tello.send_rc_control(0, 0, -20, 0)

    time.sleep(0.1)  # Pausa para permitir ajustes
    current_height = tello.get_height()

# Detener el movimiento vertical una vez alcanzada la altura
tello.send_rc_control(0, 0, 0, 0)
#tello.send_rc_control(0, 0, 30, 0)

# Inicializar los parámetros de la cámara
wCam, hCam = 640, 480

#cap = cv2.VideoCapture(0)
#cap.set(3, wCam)
#cap.set(4, hCam)

#Inicializo los valores del drone
lr = 0  # Left/Right (izquierda/derecha)
fb = 0  # Forward/Backward (adelante/atrás)
ud = 0  # Up/Down (arriba/abajo)
yaw = 0  # Yaw (rotación)
SPEED = 25

# Establecer un rango de distancias para el movimiento adelante/atrás
MIN_HAND_SIZE = 100  # Umbral de mano "alejada"
MAX_HAND_SIZE = 200  # Umbral de mano "cercana"



# Inicializar mediapipe para detección de manos
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Inicializar la cascada para la detección de rostros
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

# Coordenadas para los dedos
fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinate = (4, 2)

# Bucle de detección
while True:
    img = tello.get_frame_read().frame
    img = cv2.resize(img, (640, 480))  # Redimensionar si es necesario

    # Voltear horizontalmente la imagen para evitar efecto espejo
    img = cv2.flip(img, 1)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detección de rostro
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 10)

    for (x, y, w, h) in faces:
        # Dibuja un recuadro alrededor del rostro
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Definir el área para la detección de manos (a la derecha del rostro)
        rect_x1 = x + w + 10  # 10 píxeles a la derecha del rostro
        rect_y1 = y
        rect_x2 = rect_x1 + w
        rect_y2 = y + h

        # Dibuja el rectángulo azul en el área especificada
        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 2)

        # Etiqueta el área de detección de la mano
        cv2.putText(img, "Area de Mano", (rect_x1, rect_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Procesar la detección de manos
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks
    handedness = results.multi_handedness  # Información sobre mano derecha/izquierda

    if multiLandMarks and handedness:
        for hand_idx, handLms in enumerate(multiLandMarks):
            # Obtener la etiqueta correcta de la mano (derecha o izquierda)
            hand_label = handedness[hand_idx].classification[0].label  # "Right" o "Left"

            # Dibujar la mano detectada
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Etiquetar si la mano es derecha o izquierda
            h, w, c = img.shape
            cx, cy = int(handLms.landmark[0].x * w), int(handLms.landmark[0].y * h)  # Coordenadas de la mano
            cv2.putText(img, hand_label, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Mostrar puntos de la mano
            handPoint = []
            for idx, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoint.append((cx, cy))

            # Verificar si la mano está dentro del área del rectángulo azul
            hand_inside_rect = False
            for point in handPoint:
                if rect_x1 <= point[0] <= rect_x2 and rect_y1 <= point[1] <= rect_y2:
                    hand_inside_rect = True
                    break

            if hand_inside_rect:
                # Mostrar puntos de la mano dentro del rectángulo
                for point in handPoint:
                    cv2.circle(img, point, 10, (255, 0, 255), cv2.FILLED)

                # Contar dedos levantados
                upCount = 0
                for coordinates in fingerCoordinates:
                    if handPoint[coordinates[0]][1] < handPoint[coordinates[1]][1]:
                        upCount += 1
                if handPoint[thumbCoordinate[1]][0] > handPoint[thumbCoordinate[0]][0]:
                    upCount += 1

                cv2.putText(img, str(upCount), (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 6, 255), 2)

                if upCount == 0:
                    #quieto
                    tello.send_rc_control(0,0,0,0)

                elif upCount == 1:
                    #Adelante

                    tello.send_rc_control(0,SPEED,0,0)
                elif upCount == 2:
                    #atras
                    tello.send_rc_control(0, -SPEED, 0, 0)
                elif upCount == 3:
                    #Giro
                    tello.send_rc_control(0, 0, 0, SPEED+10)
                elif upCount == 4:
                    print("Cuatro")
                    #abajo
                    tello.send_rc_control(0,0,-SPEED,0)
                elif upCount == 5:
                    #Pirueta
                    #tello.land()
                    if tello.get_battery() < 50:
                        print("Poca Pila")
                    else:
                        print("Cinco")
                        #tello.flip_right()
                        #time.sleep(2)
                        tello.flip_back()
                        time.sleep(2)
                        #tello.flip_left()
                        #time.sleep(2)
                        #tello.flip_forward()
                        tello.land()





                    #tello.land()
                    # Moverse hacia abajo
                else:
                    tello.send_rc_control(0, 0, 0, 0)
                    print("No se detectó la mano")
            else:
                tello.send_rc_control(0, 0, 0, 0)
                print("La mano no está dentro del rectángulo azul")

            #tello.send_rc_control(lr, fb, ud, yaw)


    cv2.imshow('rostro y mano', img)

    if cv2.waitKey(1) & 0xFF == ord('m'):
        tello.streamoff()
        tello.land()
        break

tello.streamoff()
cv2.destroyAllWindows()
