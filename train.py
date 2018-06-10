import numpy as np
import cv2, os ,openface, dlib
from model import create_model
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

model = create_model()
model.load_weights('weights/nn4.small2.v1.h5')
image_path = 'pics/t1'
labels = os.listdir(image_path)
predictor_model = "shape_predictor_68_face_landmarks.dat"

y_labels = []
x_train = []
currentId = 0
labelIds = {}

def read_img(path):
    image = cv2.imread(path)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
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

def embed(aligned):
    if aligned is not None:
        x_train.append(model.predict(np.expand_dims(aligned, axis=0))[0])
        y_labels.append(id_)
    return x_train, y_labels


for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            filename = os.path.split(path)[1]
            filename = str(filename)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in labelIds:
                labelIds[label] = currentId
                currentId += 1
            id_ = labelIds[label]

            image = read_img(path)
            #image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
            aligned = face_detect(image)
            x_train, y_labels = embed(aligned)
        print(x_train, y_labels)

print("spliting train and test dataset")
X_train, X_test, y_train, y_test = train_test_split(np.array(x_train), np.array(y_labels),test_size=0.25, random_state=33)

#training
print('Training model......')
clf = SVC(C=1, kernel='linear', probability=True)
clf.fit(X_train, y_train)
clf.score(X_train,y_train)
print('Model trained')

estimators = []
estimators.append(('standardize', preprocessing.StandardScaler()))
estimators.append(('svm', SVC()))
pipe = Pipeline(estimators)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

scores = cross_val_score(pipe, x_train, y_labels, cv=4)
print("scores:",scores)
print("Confidence: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

model = (clf, x_train, y_labels)
pickle.dump(model, open("models/svmConfid20nf.pkl", 'wb'))
print("Saved model to disk")