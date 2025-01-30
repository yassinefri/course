import cv2
import face_recognition
import numpy as np

# Charger les deux images
image1_path = "image.jpg"
image2_path = "image1.jpg"

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Vérifier si les images existent
if image1 is None or image2 is None:
    print("Erreur : Une ou plusieurs images sont introuvables.")
    exit()

# Afficher les deux images
cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 🔹 **TRAITEMENT DES IMAGES**
# Conversion en niveaux de gris et sauvegarde
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray_image1.jpg", gray_image1)
cv2.imwrite("gray_image2.jpg", gray_image2)

# Redimensionner en maintenant le ratio
height1, width1 = image1.shape[:2]
height2, width2 = image2.shape[:2]

new_width = 300
new_height1 = int((new_width / width1) * height1)
new_height2 = int((new_width / width2) * height2)

resized_image1 = cv2.resize(image1, (new_width, new_height1))
resized_image2 = cv2.resize(image2, (new_width, new_height2))

cv2.imshow("Resized Image 1", resized_image1)
cv2.imshow("Resized Image 2", resized_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Normalisation
normalized_image1 = (image1 / 255.0).astype(np.float32)
normalized_image2 = (image2 / 255.0).astype(np.float32)

norm_image1 = cv2.normalize(image1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
norm_image2 = cv2.normalize(image2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

cv2.imshow("Normalized Image 1", norm_image1)
cv2.imshow("Normalized Image 2", norm_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 🔹 **DÉTECTION ET COMPARAISON DES VISAGES**
# Charger les images avec face_recognition
known_image = face_recognition.load_image_file(image1_path)
unknown_image = face_recognition.load_image_file(image2_path)

# Extraire les encodages des visages
known_encodings = face_recognition.face_encodings(known_image)
unknown_encodings = face_recognition.face_encodings(unknown_image)

# Vérifier si un visage est détecté dans chaque image
if len(known_encodings) == 0:
    print("Aucun visage détecté dans 'image.jpg'")
elif len(unknown_encodings) == 0:
    print("Aucun visage détecté dans 'image1.jpg'")
else:
    # Comparer les visages
    result = face_recognition.compare_faces([known_encodings[0]], unknown_encodings[0])
    print("Les visages sont-ils similaires ?", result)
