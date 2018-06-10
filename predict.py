import cv2, os
import dlib
import openface
import pickle
from scipy.spatial import distance
from model import create_model
import numpy as np

clf, x_train, y_labels = pickle.load(open("models/svmConfid20nf.pkl", 'rb'))
stats =[]

baseDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(baseDir, "pics/newtest/")

predictor_model = "shape_predictor_68_face_landmarks.dat"
total = []
encodings = []

def read_img(path):
    image = cv2.imread(path)
    return image

def face_detect(image):
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)
    detected_faces = face_detector(image, 1)

    for i, face_rect in enumerate(detected_faces):
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i+1, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))

        alignedFace = face_aligner.align(96, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return alignedFace

model = create_model()
model.load_weights('weights/nn4.small2.v1.h5')

def embed(aligned):
    if aligned is not None:
        encodings.append(model.predict(np.expand_dims(aligned, axis=0))[0])
    return encodings

def dist(emb1, emb2):
    return distance.euclidean(emb1,emb2)
    #dist_vect = np.sum(np.square(np.subtract(emb1, emb2)), axis=0, dtype=np.float32, keepdims=True)
    #dist_vect = np.squeeze(dist_vect)
    #print(dist_vect)
    #return dist_vect

def cal():
    for face in face_encoding:
        dis_list = []
        for x in x_train:
            dis = dist(x, face)
            if dis > 0.0001:
                dis_list.append(dis)
        m = np.argmin(dis_list)
        index = y_labels[m]
    print(dis_list)
    print('minimum is in the position:',m)
    print('minimum label is:',index)
    return index

for root, dirs, files in os.walk(imageDir):
    count = 0
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):

            path = os.path.join(root, file)
            test_image = cv2.imread(path)

            print_image = cv2.resize(test_image, (500, 500), interpolation=cv2.INTER_CUBIC)
            aligned = face_detect(test_image)
            face_encoding = embed(aligned)
            #index = clf.predict(face_encoding)[-1]
            #print(index)
            index = cal()

            if index == 0:
                name = 'trump'
            elif index == 1:
                name = 'obama'
            elif index == 2:
                name = 'lee min ho'

            filename = os.path.split(path)[1]
            filename = str(filename)
            if name in filename:
                count += 1
                total.append(count)
            total_correct = len(total)
            accu = (total_correct / (len(files))) * 100
            print("Accuracy of model:", accu)

            cv2.putText(print_image, name, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 211, 173), 1, cv2.LINE_AA)
            cv2.imshow('image', print_image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
#predicted = clf.predict(face_encoding)
#print("predicted as",predicted)
