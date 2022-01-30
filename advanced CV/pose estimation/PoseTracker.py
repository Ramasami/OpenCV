import cv2
import time
import PoseEstimationModule as pem


def add_fps(img, prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    img = cv2.flip(img, 1)
    img = cv2.putText(img, str(int(fps)), (10, 70),
                      cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    return img, current_time


def display(img):
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        return False
    return True


def main():
    cap = cv2.VideoCapture("./Resources/Dance.mp4")
    prev_time = 0
    detector = pem.PoseDetector()
    while True:
        _, img = cap.read()
        img = detector.find_pose(img)
        lm_lst = detector.find_position(img, False)
        if len(lm_lst) != 0:
            print(lm_lst[detector.mpPose.PoseLandmark.NOSE])
        img, prev_time = add_fps(img, prev_time)
        if not display(img):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
