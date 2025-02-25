

class VCItem:
    export_meta_keys = None

    def __init__(self):
        self.uid = None
        self.pid = None                 # like 3AWETUDC93BFD95NT2OBKUUXB6VIZ4
        self.sid = None                 # 1-0-0, 2-2-3 (scenario-stage-seq)
        self.device = None              # device knows focal length, camera mat, etc ...

        self.point_x = None
        self.point_y = None
        self.camera_x = None
        self.camera_y = None
        self.screen_h = None
        self.screen_w = None
        self.orientation = None

        self.timestamp = None           # epoch time
        self.image_frame = None

        # TODO: move to VCItemExt
        self.face_rect = None
        self.l_eye_rect = None
        self.r_eye_rect = None
        self.refer_vec = None
        self.image_face = None
        self.image_uv_frame = None      # normalized coordinates

    @staticmethod
    def set_export_meta_keys():
        VCItem.export_meta_keys = []



