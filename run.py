import numpy as np
import cv2
import os
import numpy as np
from download_landmarks import setup_landmarks
from align import AlignDlib

setup_landmarks()
alignment = AlignDlib('models/landmarks.dat')


from model import create_model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]
def allign_image(image_to_align):
  bb = alignment.getLargestFaceBoundingBox(image_to_align)
  if bb is None:
      raise ValueError('Nema lica')
  aligned = alignment.align(96, image_to_align, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
  aligned=aligned/255.0
  return aligned,bb

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))
#Posecuje sve slike, smesta u listu vektor_tag (vektor, ime)
vektor_ime=[]
img_path = os.path.join(os.getcwd(),'images')
for fldr in os.listdir(img_path):
  for img in os.listdir(os.path.join(img_path, fldr)):
    loaded = load_image(os.path.join(img_path,fldr,img))
    
    aligned,bb = allign_image(loaded)
    predicted = nn4_small2_pretrained.predict(np.expand_dims(aligned, axis=0))[0]
    vektor_ime.append((predicted,fldr))

def check(vector, vektor_tag):
  minimum = 10
  name = 'Unknown'
  for vektor_tag_par in vektor_tag:
    dst = distance(vector, vektor_tag_par[0]) 
    if dst < 0.6:
      if dst<minimum:
        minimum=dst
        name = vektor_tag_par[1]
  return name, minimum




cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    try:
        aligned,bb = allign_image(frame)
        cv2.rectangle(frame,(bb.left(), bb.top()),(bb.right(), bb.bottom()),(0,255,0),3)
    except ValueError as Error:
        print('Nema lica')
    embedded = nn4_small2_pretrained.predict(np.expand_dims(aligned, axis=0))[0]
    identity, dst = check(embedded,vektor_ime)
    print(identity)
    print(dst)
    print (bb)
    print (bb.top())
    print (bb.left())
    txt = identity + ' ' + str(dst)
    cv2.putText(frame, identity, (bb.left(), bb.top()), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA) 
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()