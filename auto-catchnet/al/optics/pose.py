import math

"""
In computer vision the pose of an object refers to its relative orientation and position with respect to a camera.
You can change the pose by either moving the object with respect to the camera, or the camera with respect to the object

[1] Perspective-n-Point problem (PNP)
- prepare: calibrated camera
- result: know the locations of n 3d points on the object

# Motion of 3D rigid object
- Translation: move (x, y, z) -> (x', y', z') (3 degrees of freedom)
- Rotation: rotate camera about x, y, z axes

- angle representation 
  - Euler angles(roll, pitch, yaw) 
  - rotation matrix [3x3]

# estimating the pose of 3D object 
- find 6 parameter above (translation + rotation)
- requires:
- A. 2D coordinates of a few points in image (얼굴 특징점)
- B. 3D locations of the same points 
    - ★얼굴 모델 전체가 필요한 것이 아니라 특징점 좌표만으로도 가능
    - 6점 PNP면 6개의 vector3 준비 (World coordinates 기준)
- C. Intrinsic parameters of the camera
    - 1. focal length of the camera 
    - 2. optical center in the image (approximate center of the image)
    - 3. radial distortion parameters (approximate radial distortion does not exist)
     
# three coordinate systems
- A. facial features (world 3d coordinates)
- B. camera 3d coordinates 
   - pose (translation + rotation) 정보를 알면 world to camera coordinate로 transform 가능
- C. image coordinate (from camera coordinate projected onto image plane using intrinsic params; focal, optical center)  
"""





















