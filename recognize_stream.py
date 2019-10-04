# !/

# imports
import pickle
import time
import os
from os import listdir
from os.path import isfile, join
import cv2
import imutils
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream


class Program:
    def __init__(self):
        # arguments / settings
        self.args = dict()
        self.args["image"] = ''
        self.args["dataset"] = 'dataset'
        self.args["detector_txt"] = 'deploy.prototxt'
        self.args["detector_model"] = 'res10_300x300_ssd_iter_140000.caffemodel'
        self.args["embedding_model"] = 'openface_nn4.small2.v1.t7'
        self.args["confidence"] = float(0.5)
        self.args["embeddings"] = 'embeddings.pickle'
        self.args["recognizer"] = 'recognizer.pickle'
        self.args["le"] = 'le.pickle'

        self.initialized = True

    def __enter__(self):

        """

        :return:
        """
        try:
            if self.initialized:
                s, d = self.preflight()
                if d['status']:
                    print('pre-flight: ok')

                    s, d = self.flight(d['detector'], d['embedder'], d['recognizer'], d['le'], d['vs'])
                    if d['status']:
                        print('flight: ok')
                    else:
                        print('flight: failed')

                else:
                    print('pre-flight: failed')
            else:
                print('initialized: failed')
        except Exception as e:
            print(e)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def preflight(self):
        """
        :return:
        """
        status = False
        data = dict()
        data['status'] = False
        try:
            # load our serialized face detector from disk
            print("[INFO] loading face detector...")
            protoPath = self.args["detector_txt"]
            modelPath = self.args["detector_model"]
            data['detector'] = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

            # load our serialized face embedding model from disk
            print("[INFO] loading face recognizer...")
            data['embedder'] = cv2.dnn.readNetFromTorch(self.args["embedding_model"])

            # load the actual face recognition model along with the label encoder
            data['recognizer'] = pickle.loads(open(self.args["recognizer"], "rb").read())
            data['le'] = pickle.loads(open(self.args["le"], "rb").read())

            # initialize the video stream, then allow the camera sensor to warm up
            print("[INFO] starting video stream...")
            data['vs'] = VideoStream(src=0).start()
            time.sleep(2.0)

            data['status'] = True
            status = True
        except Exception as e:
            print(e)
        return status, data

    def flight(self, detector, embedder, recognizer, le, vs):
        """
        :return:
        """
        status = False
        data = dict()
        data['status'] = False
        try:

            # start the FPS throughput estimator
            fps = FPS().start()

            # loop over frames from the video file stream
            while True:
                # grab the frame from the threaded video stream
                frame = vs.read()

                # resize the frame to have a width of 600 pixels (while
                # maintaining the aspect ratio), and then grab the image
                # dimensions
                frame = imutils.resize(frame, width=600)
                (h, w) = frame.shape[:2]

                # construct a blob from the image
                imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)

                # apply OpenCV's deep learning-based face detector to localize
                # faces in the input image
                detector.setInput(imageBlob)
                detections = detector.forward()

                # loop over the detections
                for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections
                    if confidence > self.args["confidence"]:

                        # compute the (x, y)-coordinates of the bounding box for
                        # the face
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # extract the face ROI
                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # ensure the face width and height are sufficiently large
                        if fW < 20 or fH < 20:
                            continue
                        # construct a blob for the face ROI, then pass the blob
                        # through our face embedding model to obtain the 128-d
                        # quantification of the face
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()

                        # perform classification to recognize the face
                        preds = recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)

                        proba = preds[j]
                        name = le.classes_[j]

                        # draw the bounding box of the face along with the
                        # associated probability
                        text = "{}: {:.2f}%".format(name, proba * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (255, 0, 0), 2)
                        cv2.putText(frame, text, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                        probability = proba * 100
                        s, d = self.capture_faces(name, probability, frame, startY, startX, endY, endX)
                        if d['status']:
                            print("possible face match of " + str(int(probability)) + "% captured for "
                                  + str(name) + " ...")
                        else:
                            print("face caputure failed.")

                # update the FPS counter
                fps.update()

                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            # stop the timer and display FPS information
            fps.stop()
            print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

            # do a bit of cleanup
            cv2.destroyAllWindows()
            vs.stop()

            data['status'] = True
            status = True

        except Exception as e:
            print(e)

        return status, data

    @staticmethod
    def capture_faces(name, proba, frame, startY, startX, endY, endX):
        """
        :return:
        """
        status = False
        data = dict()
        data['status'] = False
        try:
            ts = time.time()
            # capture faces images
            if 20 <= proba < 100:
                im = frame[startY:endY, startX:endX]
                files = [f for f in listdir("captured/" + name + "/") if isfile(join("captured/" + name + "/", f))]
                percent = [file.split('_', 1)[0] for file in files]
                if str(int(proba)) not in percent:
                    cv2.imwrite("captured/" + name + "/" + str(int(proba)) + "_" + str(int(ts)) + ".jpg", im)
                elif str(int(proba)) in percent:
                    cv2.imwrite("captured/" + name + "/" + str(files[percent.index(str(int(proba)))]), im)
                    os.rename("captured/" + name + "/" + str(files[percent.index(str(int(proba)))]), "captured/"
                              + name + "/" + str(int(proba)) + "_" + str(int(ts)) + ".jpg")
            else:
                im = frame[startY:endY, startX:endX]
                files = [f for f in listdir("captured/unknowns/") if isfile(join("captured/unknowns/", f))]
                percent = [file.split('_', 1)[0] for file in files]
                if str(int(proba)) not in percent:
                    cv2.imwrite("captured/unknowns/" + str(int(proba)) + "_" + str(int(ts)) + ".jpg", im)
                elif str(int(proba)) in percent:
                    cv2.imwrite("captured/unknowns/" + str(files[percent.index(str(int(proba)))]), im)
                    os.rename("captured/unknowns/" + str(files[percent.index(str(int(proba)))]),
                              "captured/unknowns/" + str(int(proba)) + "_" + str(int(ts)) + ".jpg")
            data['status'] = True
            status = True
        except Exception as e:
            print(e)
        return status, data


if __name__ == "__main__":
    """
    """
    with Program():
        pass
