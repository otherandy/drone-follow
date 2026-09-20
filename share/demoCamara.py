import cv2
import mediapipe as mp
from djitellopy import Tello
import time
import math

#Programa para seguir mano
#Ya calibrado y probado
# Función para calcular la distancia entre dos puntos (landmarks)
def calculate_distance(landmark1, landmark2, frame_width, frame_height):
    x1, y1 = landmark1.x * frame_width, landmark1.y * frame_height
    x2, y2 = landmark2.x * frame_width, landmark2.y * frame_height
    return math.hypot(x2 - x1, y2 - y1)

# Inicializar el dron Tello
#tello = Tello()
#tello.connect()
#print(f'Battery: {tello.get_battery()}%')

# Iniciar la transmisión de video desde el dron
#tello.streamon()
cap = cv2.VideoCapture(0)

# Inicializar MediaPipe para la detección de manos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Despegar el dron
#El dron despega para prueba de parametros no se activa
#tello.takeoff()
#desired_height = 150
#current_height = tello.get_height()


# Detener el movimiento vertical una vez alcanzada la altura
#tello.send_rc_control(0, 0, 0, 0)

# Establecer una velocidad base para el dron
SPEED = 25

# Establecer un rango de distancias para el movimiento adelante/atrás
MIN_HAND_SIZE = 100  # Umbral de mano "alejada"
MAX_HAND_SIZE = 200  # Umbral de mano "cercana"
STOP_THRESHOLD = 250
MAX_HEIGHT = 150

try:
    while True:
        #Calculo la distancia actual del dron
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))  # Redimensionar si es necesario

        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)

        # Inicializar las velocidades de movimiento
        lr = 0  # Left/Right (izquierda/derecha)
        fb = 0  # Forward/Backward (adelante/atrás)
        ud = 0  # Up/Down (arriba/abajo)
        yaw = 0  # Yaw (rotación)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)



                # Obtener la posición del centro de la mano (wrist)
                wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1]
                wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0]

                # Determinar los movimientos del dron basados en la posición de la mano
                if wrist_x < frame.shape[1] * 0.3:
                    lr = -SPEED  # Moverse a la izquierda
                    print("muevo a la izquierdo")
                elif wrist_x > frame.shape[1] * 0.7:
                    lr = SPEED  # Moverse a la derecha
                    print("muevo a la derecha")

                # if wrist_y < frame.shape[0] * 0.3:
                #     ud = SPEED  # Moverse hacia arriba
                # elif wrist_y > frame.shape[0] * 0.7:
                #     ud = -SPEED  # Moverse hacia abajo

                #Nueva rutina para arriba y abajo
                if wrist_y < frame.shape[0] * 0.3:
                    print('Es mas alto que 1.5')
                    ud = SPEED  # Moverse hacia arriba si la altura es menor a 1.5 metros
                elif wrist_y > frame.shape[0] * 0.7:
                    ud = -SPEED  # Moverse hacia abajo
                    print("me muevo hacia abajo")

                # Calcular la distancia entre el pulgar y el meñique como un proxy para la "cercanía" de la mano
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                hand_size = calculate_distance(thumb_tip, pinky_tip, frame.shape[1], frame.shape[0])
                #print("El valor de la handsize es")
                #print(hand_size)

                # Ajustar movimiento hacia adelante o atrás basado en el tamaño de la mano
                # if hand_size < MIN_HAND_SIZE:
                #     print('debo moverme estoy lejos')
                #     print(hand_size)
                #     fb = SPEED  # Moverse hacia adelante si la mano está lejos
                # elif hand_size > MAX_HAND_SIZE:
                #     #print('debo moverme')
                #     print('debo moverme estoy muy cerca')
                #     print(hand_size)
                #     fb = -SPEED  # Moverse hacia atrás si la mano está muy cerca

                #Nuevo ajuste para las  nuevas distancias
                if hand_size < MIN_HAND_SIZE:
                    print("Me muevo adelante")
                    fb = SPEED  # Moverse hacia adelante si la mano está lejos
                elif hand_size > MAX_HAND_SIZE and hand_size < STOP_THRESHOLD:
                    fb = -SPEED  # Moverse hacia atrás si la mano está muy cerca
                    print("Me muevo atrá")
                elif hand_size >= STOP_THRESHOLD:
                    fb = 0  # Detenerse si la mano está muy, muy cerca
                    print("Me detengo estoy muy cerca")

        # Enviar los comandos de control al dron
        else:
            print("No detecté manos")
            lr = 0  # Left/Right (izquierda/derecha)
            fb = 0  # Forward/Backward (adelante/atrás)
            ud = 0  # Up/Down (arriba/abajo)
            yaw = 0  # Yaw (rotación)

        #tello.send_rc_control(lr, fb, ud, yaw)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('m'):
            break

finally:
    #tello.land()  # Aterrizar el dron al finalizar
    #tello.streamoff()  # Apagar la transmisión de video
    cv2.destroyAllWindows()
