import time

import cv2

import FaceMeshModule as fmm


def add_fps(img, prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    img = cv2.putText(img, str(int(fps)), (10, 70),
                      cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    return img, current_time


def display(img):
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        return False
    return True


def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = fmm.FaceDetector()
    while True:
        _, img = cap.read()
        img = detector.detect_mesh(img)
        lm_lst = detector.find_position(img)
        if len(lm_lst) != 0:
            print(lm_lst)
        img, prev_time = add_fps(img, prev_time)
        if not display(img):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
