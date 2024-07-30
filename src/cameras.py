import cv2


class Camera:
    def read_frame(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class CVCamera(Camera):
    def __init__(self, recording_res, index_cam=0):
        self.index_cam = index_cam
        self.rec_res = recording_res

    def read_frame(self):
        ret, frame = self.vid.read()
        return frame

    def start(self):
        self.vid = cv2.VideoCapture(self.index_cam)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.rec_res[0])
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rec_res[1])

    def stop(self):
        self.vid.release()

    def close(self):
        self.vid.release()


class PICamera(Camera):
    def __init__(self, recording_res):
        try:
            from picamera import Picamera2
            import libcamera

            picam2 = Picamera2()
            preview_config = picam2.create_preview_configuration(
                main={"size": recording_res},
                controls={
                    "AwbEnable": False,
                    # "AwbMode": libcamera.controls.AwbModeEnum.Indoor,
                    "AwbMode": libcamera.controls.AwbModeEnum.Auto,
                    "AnalogueGain": 1.0,
                },
            )
            picam2.configure(preview_config)

        except ModuleNotFoundError:
            print(
                "Cannnot initialize PiCamera on this device. Check if you are using a Raspberry Pi and it is up to date."
            )
            exit()

    def read_frame(self):
        return super().read_frame()
