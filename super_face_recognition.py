import face_recognition
import cv2

# Cargar la imagen de referencia
try:
    known_image = face_recognition.load_image_file("images/mick.jpg")
    known_encodings = face_recognition.face_encodings(known_image)

    if len(known_encodings) == 0:
        raise ValueError("No se encontraron encodings en la imagen de referencia.")
    
    known_encoding = known_encodings[0]  # Solo necesitamos una codificación

except Exception as e:
    print(f"Ocurrió un error al cargar la imagen: {e}")
    exit()

# Inicializar la captura de video
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("No se pudo capturar el video.")
        break

    # Convertir la imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar las caras en el fotograma actual
    face_locations = face_recognition.face_locations(rgb_frame)
    print(f"Ubicaciones de caras detectadas: {face_locations}")

    # Obtener las codificaciones faciales para las caras detectadas
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(face_encodings) == 0:
        print("No se encontraron codificaciones faciales en el fotograma.")
        continue

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Comparar con la imagen conocida
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        top, right, bottom, left = face_location
        # Dibujar un recuadro verde si hay coincidencia
        if True in matches:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            print("¡Coincidencia encontrada!")
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Rojo si no coincide

    # Mostrar el fotograma
    cv2.imshow('Video', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
video_capture.release()
cv2.destroyAllWindows()
