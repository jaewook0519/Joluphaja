import cv2
import numpy as np

left_camera_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float32)

left_dist_coeffs = np.zeros((5, 1))

right_camera_matrix = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]], dtype=np.float32)

right_dist_coeffs = np.zeros((5, 1))

R = np.array([[9.99938586e-01, -1.10825893e-02, -2.42765850e-05],
              [1.10825889e-02, 9.99938586e-01, -1.64385711e-05],
              [2.44572760e-05, 1.61685141e-05, 1.00000000e+00]], dtype=np.float32)

T = np.array([[-2.83332123e-01],
              [-1.29551065e-02],
              [-5.96412636e-05]], dtype=np.float32)

stereo_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM)

left_cap = cv2.VideoCapture(0)
right_cap = cv2.VideoCapture(1)

if not left_cap.isOpened() or not right_cap.isOpened():
    print("카메라 오류")
    exit()
    
while True:
    ret_left, left_image = left_cap.read()
    ret_right, right_image = right_cap.read()
    
    if not ret_left or not ret_right:
        print("프레임 오류")
        break
    
    disparity = stereo_matcher.compute(left_image, right_image).astype(np.float32) / 16.0
    depth_map = cv2.normalize(disparity, None, 255, 0, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow('Depth Map', depth_map)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
left_cap.release()
right_cap.release()
cv2.destroyAllWindows()
