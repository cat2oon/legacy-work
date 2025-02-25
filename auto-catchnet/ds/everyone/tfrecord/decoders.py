import tensorflow as tf


def deserialize_example(example, frame_ch=1, face_ch=1, eye_ch=1, frame_hw=224, face_hw=224, eye_hw=112):
    keys_to_features = {
        'image/frame': tf.FixedLenFeature([], tf.string),
        'image/face': tf.FixedLenFeature([], tf.string),
        'image/left_eye': tf.FixedLenFeature([], tf.string),
        'image/right_eye': tf.FixedLenFeature([], tf.string),
        'model/candide': tf.FixedLenFeature([], tf.string),
        'label/uid': tf.FixedLenFeature([], tf.int64),
        'label/orientation': tf.FixedLenFeature([], tf.int64),
        'label/cam_x': tf.FixedLenFeature([], tf.float32),
        'label/cam_y': tf.FixedLenFeature([], tf.float32),
        'label/cam_to_screen_x': tf.FixedLenFeature([], tf.float32),
        'label/cam_to_screen_y': tf.FixedLenFeature([], tf.float32),
    }

    features = tf.parse_single_example(example, keys_to_features)

    face = features['image/face']
    face = tf.image.decode_jpeg(face, channels=face_ch)
    face = tf.image.resize_images(face,
                                  [face_hw, face_hw],
                                  preserve_aspect_ratio=True,
                                  method=tf.image.ResizeMethod.AREA)
    face = tf.image.resize_image_with_pad(face,
                                          face_hw,
                                          face_hw,
                                          method=tf.image.ResizeMethod.BILINEAR)

    frame = features['image/frame']
    frame = tf.image.decode_jpeg(frame, channels=frame_ch)
    frame = tf.image.resize_images(frame,
                                   [frame_hw, frame_hw],
                                   preserve_aspect_ratio=True,
                                   method=tf.image.ResizeMethod.AREA)
    frame = tf.image.resize_image_with_pad(frame,
                                           frame_hw,
                                           frame_hw,
                                           method=tf.image.ResizeMethod.BILINEAR)

    left_eye = features['image/left_eye']
    left_eye = tf.image.decode_jpeg(left_eye, channels=eye_ch)
    left_eye = tf.image.resize_images(left_eye, [eye_hw, eye_hw])

    right_eye = features['image/right_eye']
    right_eye = tf.image.decode_jpeg(right_eye, channels=eye_ch)
    right_eye = tf.image.resize_images(right_eye, [eye_hw, eye_hw])

    candide = features['model/candide']
    candide = tf.decode_raw(candide, tf.float64)
    candide = tf.cast(candide, tf.float32)

    uid = features['label/uid']
    so = features['label/orientation']
    cam_x = features['label/cam_x']
    cam_y = features['label/cam_y']
    cam_to_scr_x = features['label/cam_to_screen_x'] / 10.0
    cam_to_scr_y = features['label/cam_to_screen_y'] / 10.0

    return uid, frame, face, left_eye, right_eye, so, cam_x, cam_y, cam_to_scr_x, cam_to_scr_y, candide


def legacy():
    def legacy_example(example, frame_ch=3, eye_ch=1, frame_h=480, frame_w=680, eye_h=64, eye_w=64):
        features = None

        frame = features['image/frame']
        left = features['image/left_eye']
        right = features['image/right_eye']

        frame = tf.image.decode_jpeg(frame, channels=frame_ch)
        left = tf.image.decode_jpeg(left, channels=eye_ch)
        right = tf.image.decode_jpeg(right, channels=eye_ch)

        # 눈 영역의 경우 crop_or_pad 보다 일반 resize가 품질이 좋고 선명
        left = tf.image.resize_images(left, [eye_h, eye_w])
        right = tf.image.resize_images(right, [eye_h, eye_w])

        # frame = tf.image.resize_image_with_crop_or_pad(frame, 480, 480)
        # 거의 항상 축소인 경우이므로 Interpolation Area 방식으로 resize
        # NOTE! aspect_ratio 유지하지만 정각 차원으로 padding을 넣어주지는 않음
        # 한번에 aspect ratio와 padding을 넣어주는 API는 없는 듯
        frame = tf.image.resize_images(frame, [frame_h, frame_w],
                                       preserve_aspect_ratio=True,
                                       method=tf.image.ResizeMethod.AREA)
        frame = tf.image.resize_image_with_pad(frame, frame_h, frame_w,
                                               method=tf.image.ResizeMethod.BILINEAR)

        cam_x = features['label/cam_x']
        cam_y = features['label/cam_y']

        cam_to_scr_x = features['label/cam_to_screen_x']
        cam_to_scr_y = features['label/cam_to_screen_y']

        return frame, left, right, cam_x, cam_y, cam_to_scr_x, cam_to_scr_y
    return None
