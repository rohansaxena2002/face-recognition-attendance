from flask import Flask, render_template, Response, send_file
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime, date
app = Flask(__name__)


global camera
camera = cv2.VideoCapture(0)


f = open("Attendance.csv", "w")
f.truncate()
f.close()
with open('Attendance.csv', 'r+') as f:
    date = date.today()
    f.writelines(f'{date}')
    f.writelines(f'\nName,Time')


path = 'ImagesAttendance'
images = []
names = []
myList = os.listdir(path)


for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    names.append(os.path.splitext(cl)[0])


def faceEncodings(images):
    encodeList = []
    for im in images:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(im)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = faceEncodings(images)
print("Encoding done")


def gen_frames():
    while True:
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # ret, img = camera.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(
                imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(
                    encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(
                    encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = names[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    with open('Attendance.csv', 'r+') as f:
                        myDataList = f.readlines()
                        nameList = []
                        for line in myDataList:
                            entry = line.split(',')
                            nameList.append(entry[0])
                        if name not in nameList:
                            now = datetime.now()
                            dtString = now.strftime('%H:%M:%S')
                            f.writelines(f'\n{name},{dtString}')
            ret, jpg = cv2.imencode('.jpg', img)
            frame = jpg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/downloads')
def download_attendance():
    global camera
    camera.release()
    p = 'Attendance.csv'
    return send_file(p, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
