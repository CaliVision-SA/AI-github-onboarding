import cv2 as cv
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 6 #number of checkerboard rows.
    columns = 9 #number of checkerboard columns.
    world_scaling = 0.031 #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
 
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            #cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            #cv.imshow('img', frame)
            #k = cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints.append(corners)
 
 
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    # print('Rs:\n', rvecs)
    # print('Ts:\n', tvecs)
 
    return mtx, dist

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synched frames
    frames_folder1 = os.path.join(frames_folder, "1\\*")
    frames_folder2 = os.path.join(frames_folder, "2\\*")

    images_names1 = glob.glob(frames_folder1)
    images_names1 = sorted(images_names1)

    images_names2 = glob.glob(frames_folder2)
    images_names2 = sorted(images_names2)

    c1_images = []
    c2_images = []
    for im1, im2 in zip(images_names1, images_names2):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 6 #number of checkerboard rows.
    columns = 9 #number of checkerboard columns.
    world_scaling = 0.031 #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (6, 9), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (6, 9), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            #cv.drawChessboardCorners(frame1, (6,9), corners1, c_ret1)
            #cv.imshow('img', frame1)
 
            #cv.drawChessboardCorners(frame2, (6,9), corners2, c_ret2)
            #cv.imshow('img2', frame2)
            #k = cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T

def triangulate(im1_path, im2_path, mtx1, mtx2, R, T):
    # Load images
    img1 = cv.imread(im1_path)
    img2 = cv.imread(im2_path)

    if img1 is None or img2 is None:
        print("Error loading images.")
        return

    # Convert images to RGB (OpenCV uses BGR)
    img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

    # Initialize lists to store points
    uvs1 = []
    uvs2 = []

    # Set up the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Display images
    axes[0].imshow(img1_rgb)
    axes[0].set_title('Image 1')
    axes[0].axis('off')

    axes[1].imshow(img2_rgb)
    axes[1].set_title('Image 2')
    axes[1].axis('off')

    plt.suptitle('Click corresponding points in left and right images.\nLeft-click to select points, right-click to finish.', fontsize=16)

    # Variables to keep track of clicks
    coords_img1 = []
    coords_img2 = []
    current_axis = 0  # Start with the left image

    # Event handler for mouse clicks
    def onclick(event):
        nonlocal current_axis
        # Check if the click is in one of our axes
        if event.inaxes == axes[current_axis]:
            if event.button == 1:  # Left mouse button
                x, y = event.xdata, event.ydata
                if current_axis == 0:
                    coords_img1.append([x, y])
                    axes[current_axis].plot(x, y, 'r.', markersize=10)
                    fig.canvas.draw()
                    current_axis = 1  # Switch to the right image
                elif current_axis == 1:
                    coords_img2.append([x, y])
                    axes[current_axis].plot(x, y, 'r.', markersize=10)
                    fig.canvas.draw()
                    current_axis = 0  # Switch back to the left image
            elif event.button == 3:  # Right mouse button
                # Finish selection
                plt.close(fig)

    # Connect the event handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the images and wait for clicks
    plt.show()

    # Convert lists to numpy arrays
    uvs1 = np.array(coords_img1)
    uvs2 = np.array(coords_img2)

    if len(uvs1) == 0 or len(uvs2) == 0 or len(uvs1) != len(uvs2):
        print("No points were selected or the number of points do not match.")
        return

    # RT matrix for camera 1 is identity
    RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = mtx1 @ RT1  # Projection matrix for camera 1

    # RT matrix for camera 2 is R and T from stereo calibration
    RT2 = np.hstack((R, T))
    P2 = mtx2 @ RT2  # Projection matrix for camera 2

    def DLT(P1, P2, point1, point2):
        A = np.array([
            point1[1]*P1[2,:] - P1[1,:],
            P1[0,:] - point1[0]*P1[2,:],
            point2[1]*P2[2,:] - P2[1,:],
            P2[0,:] - point2[0]*P2[2,:]
        ])
        # Solve for the null space of A
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Homogeneous coordinates
        return X[:3]

    # Triangulate points
    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(p3d)
        print(f"Triangulated 3D point: {p3d}")

    p3ds = np.array(p3ds)

    # Plot the 3D points
    from mpl_toolkits.mplot3d import Axes3D

    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.scatter(p3ds[:, 0], p3ds[:, 1], p3ds[:, 2], c='r', marker='o')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Triangulated 3D Points')

    ax.set_xlim3d(0, 3)
    ax.set_ylim3d(-1.5, 1.5)
    ax.set_zlim3d(1.5, 4.5)

    ax.view_init(azim=-90, elev=90)

    plt.show()


mtx1, dist1 = calibrate_camera(images_folder = '.\\data\\intrinsic\\1\\*')
mtx2, dist2 = calibrate_camera(images_folder = '.\\data\\intrinsic\\2\\*')

R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, '.\\data\\extrinsic')
print(R)
print(T)

im1 = ".\\data\\extrinsic\\1\\610.png"
im2 = ".\\data\\extrinsic\\2\\610.png"

triangulate(im1, im2, mtx1, mtx2, R, T)
