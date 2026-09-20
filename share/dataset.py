import cv2
import os
from djitellopy import Tello

# Carpeta donde se guardarán las imágenes
output_dir = "dataset_trutle/images"
os.makedirs(output_dir, exist_ok=True)

# Abrir la cámara (0 = cámara por defecto)
#cap = cv2.VideoCapture(0)

# Contador de imágenes
img_count = 0
me = Tello()
me.connect()
me.streamon()
width = 640  # WIDTH OF THE IMAGE
height = 480
print("Presiona 's' para guardar imagen, 'q' para salir.")

while True:
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img_rec = cv2.cvtColor(myFrame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img_rec, (width, height))


    # Mostrar la ventana
    cv2.imshow("Recolectando dataset", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        # Guardar imagen
        img_path = os.path.join(output_dir, f"img_{img_count:04d}.jpg")
        cv2.imwrite(img_path, img)
        print(f"✅ Guardada {img_path}")
        img_count += 1

    elif key == ord("q"):
        me.streamoff()
        break

#cap.release()
cv2.destroyAllWindows()
