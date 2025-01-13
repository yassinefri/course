from flask import Flask, request, jsonify
import face_recognition

app = Flask(__name__)


@app.route("/compare_faces", methods=["POST"])
def compare_faces():
    known_image_file = request.files["known_image"]
    unknown_image_file = request.files["unknown_image"]

    known_image = face_recognition.load_image_file(known_image_file)
    unknown_image = face_recognition.load_image_file(unknown_image_file)

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    return jsonify({"match": results[0]})


if __name__ == "__main__":
    app.run(debug=True)
