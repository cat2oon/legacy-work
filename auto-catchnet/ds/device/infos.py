
# TODO: android 핸드폰 실물 가져와서 추가하기

# CameraToScreenXMm, CameraToScreenYMm,
# CameraXMm(?), CameraYMm(?), PixelsPerInch, ScreenXMm, ScreenYMm,
# ScreenWidthMm, ScreenWidthPoints, ScreenWidthPointsZoomed,
# ScreenHeightMm, ScreenHeightPoints, ScreenHeightPointsZoomed

_DEVICE_INFOS = {
    "IPHONE 6S PLUS": [23.54, 8.66, 28.33, 9.68, 401, 4.79, 18.34, 68.36, 414, 375, 121.54, 736, 667],
    "IPHONE 6S": [18.61, 8.04, 22.92, 9.08, 326, 4.31, 17.12, 58.49, 375, 320, 104.05, 667, 568],
    "IPHONE 6 PLUS": [23.54, 8.65, 28.25, 9.61, 401, 4.71, 18.26, 68.36, 414, 375, 121.54, 736, 667],
    "IPHONE 6": [18.61, 8.03, 22.85, 9.01, 326, 4.24, 17.04, 58.5, 375, 320, 104.05, 667, 568],
    "IPHONE 5S": [25.85, 10.65, 29.28, 6.07, 326, 3.43, 16.72, 51.7, 320, None, 90.39, 568, None],
    "IPHONE 5C": [25.85, 10.64, 29.59, 6.38, 326, 3.74, 17.02, 51.7, 320, None, 90.39, 568, None],
    "IPHONE 5": [25.85, 10.65, 29.28, 6.07, 326, 3.43, 16.72, 51.7, 320, None, 90.39, 568, None],
    "IPHONE 4S": [14.96, 9.78, 19.27, 10.35, 326, 4.31, 20.13, 49.92, 320, None, 74.88, 480, None],

    "IPAD MINI": [60.7, 8.7, 67.4, 10.7, 326, 6.7, 19.4, 121.3, 768, None, 161.2, 1024, None],
    "IPAD AIR 2": [76.86, 7.37, 84.74, 11.07, 264, 7.88, 18.44, 153.71, 768, None, 203.11, 1024, None],
    "IPAD AIR": [74.4, 9.9, 84.7, 11.1, 264, 10.3, 21, 149, 768, None, 198.1, 1024, None],
    "IPAD 4": [74.5, 10.5, 92.9, 11.1, 264, 18.4, 21.6, 149, 768, None, 198.1, 1024, None],
    "IPAD 3": [74.5, 10.5, 92.9, 11.1, 132, 18.4, 21.6, 149, 768, None, 198.1, 1024, None],
    "IPAD 2": [74.5, 10.5, 92.9, 11.1, 132, 18.4, 21.6, 149, 768, None, 198.1, 1024, None],
    "IPAD PRO": [98.31, 10.69, 110.29, 11.08, 264, 11.99, 21.77, 196.61, 1024, 768, 262.15, 1366, 1024],
}


def get_cam_to_screen(device_name):
    device_name = device_name.upper()
    if device_name not in _DEVICE_INFOS:
        print(device_name)
        return 20, 8        # 평균 거리 

    info = _DEVICE_INFOS[device_name]
    return info[0], info[1]
