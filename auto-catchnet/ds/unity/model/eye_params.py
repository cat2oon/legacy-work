"""
 [ Eye Param in Unity ]
 - (eye_x, eye_y, d_x, d_y)
 - (pitch, yaw, . . )
 - 아래쪽이 +, 왼쪽이 +
 - (-20 ~ 20), (-20 ~ 20)
 - 사용한 base_41에서 eye_x가 낮은 자리수
"""


def class_num_to_eye_param(num):
    eye_x = num % 41
    num -= eye_x
    eye_y = num / 41

    return eye_x - 20, eye_y - 20


def eye_param_to_class_num(eye_x, eye_y):
    eye_x += 20
    eye_y += 20
    num = eye_y
    num *= 41
    num += eye_x
    return num
