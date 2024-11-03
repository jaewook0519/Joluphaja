import cv2
import numpy as np
import glob

left_images = sorted(glob.glob(r"C:\Users\MiniPC\Desktop\stereocel\CAM1\left_*.jpg"))
right_images = sorted(glob.glob(r"C:\Users\MiniPC\Desktop\stereocel\CAM2\right_*.jpg"))

pattern_size = (6, 9)
square_size = 0.041

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
left_imgpoints = []
right_imgpoints = []

for i in range(len(left_images)):
    left_img = cv2.imread(left_images[i])
    right_img = cv2.imread(right_images[i])

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    ret_left, left_corners = cv2.findChessboardCorners(left_gray, pattern_size)
    ret_right, right_corners = cv2.findChessboardCorners(right_gray, pattern_size)

    if ret_left and ret_right:
        objpoints.append(objp)
        left_imgpoints.append(left_corners)
        right_imgpoints.append(right_corners)

        left_img = cv2.drawChessboardCorners(left_img, pattern_size, left_corners, ret_left)
        right_img = cv2.drawChessboardCorners(right_img, pattern_size, right_corners, ret_right)

        cv2.imshow(f'Left image {i}', left_img)
        cv2.imshow(f'Right image {i}', right_img)
        cv2.waitKey(0)
    else:
        print(f"image {i}: 코너 감지 오류")

cv2.destroyAllWindows()

if len(objpoints) > 0:
    left_camera_matrix = np.eye(3)
    right_camera_matrix = np.eye(3)
    left_dist_coeffs = np.zeros((5, 1))
    right_dist_coeffs = np.zeros((5, 1))

    ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, E, F = cv2.stereoCalibrate(
        objpoints, left_imgpoints, right_imgpoints, left_camera_matrix, left_dist_coeffs,
        right_camera_matrix, right_dist_coeffs, imageSize=left_gray.shape[::-1],
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    if ret:
        print("Left Camera Matrix:\n", left_camera_matrix)
        print("Left Distortion Coefficients:\n", left_dist_coeffs.flatten())
        print("Right Camera Matrix:\n", right_camera_matrix)
        print("Right Distortion Coefficients:\n", right_dist_coeffs.flatten())
        print("Rotation Matrix R:\n", R)
        print("Translation Vector T:\n", T)
        print("Essential Matrix E:\n", E)
        print("Fundamental Matrix F:\n", F)

    else:
        print("캘리브레이션 실패")
else:
    print("파라미터 재확인하거나 사진 다시 찍어야할듯")
