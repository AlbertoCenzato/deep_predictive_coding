import numpy as np
import os


class Writer(object):

    def __init__(self):
        self.opened = False
        self.resolution = (0, 0)
        self.FPS = 0

    def open(self, path):
        self.opened = True

    def close(self):
        self.opened = False

    def write(self, image):
        pass

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, traceback):
        self.close()

    @property
    def resolution(self):
        return self.__res

    @resolution.setter
    def resolution(self, res):
        self.__res = res

    @property
    def FPS(self):
        return self.__fps

    @FPS.setter
    def FPS(self, fps):
        self.__fps = fps

    def is_open(self):
        return self.opened



class BufferedWriter(Writer):

    def __init__(self, buffer_size=-1):
        super(BufferedWriter, self).__init__()
        self.bufferSize = buffer_size
        self.buffer = []

    def close(self):
        self.flush_buffer()
        super(BufferedWriter, self).close()

    def flush_buffer(self):
        self.buffer = []

    def write(self, image):
        self.buffer.append(image)
        if self.bufferSize != -1 and len(self.buffer) >= self.bufferSize:
            self.flush_buffer()



class BufferedVideoWriter(BufferedWriter):
    """Buffered video writer"""

    def __init__(self, buffer_size=-1):
        self.cv2 = __import__('cv2')
        super(BufferedVideoWriter, self).__init__(buffer_size)
        self.writer = self.cv2.VideoWriter()

    def flush_buffer(self):
        for frame in self.buffer:
            self.writer.write(frame)
        super(BufferedVideoWriter, self).flush_buffer()

    def open(self, path):
        self.close()
        return self.writer.open(path, -1, self.FPS, self.resolution, isColor=False)

    def close(self):
        if not self.is_open():
            return

        self.flush_buffer()
        self.writer.release()

    def is_open(self):
        return self.writer.isOpened()



class BufferedBinaryWriter(BufferedWriter):

    def __init__(self, **kwargs):
        self.path = ""
        super(BufferedBinaryWriter, self).__init__(**kwargs)

    def open(self, path):
        self.path = path
        return super(BufferedBinaryWriter, self).open(path)

    def close(self):
        if not self.is_open():
            return
        super(BufferedBinaryWriter, self).close()

    def flush_buffer(self):
        buffer_len = len(self.buffer)
        if buffer_len < 1:
            return

        data_shape = self.buffer[0].shape
        data_type  = self.buffer[0].dtype
        array = np.empty((buffer_len,) + data_shape, dtype=data_type)
        for i in range(0, buffer_len):
            array[i, :] = self.buffer[i]

        directory = os.path.dirname(self.path)
        if not os.path.exists(directory):
            os.mkdir(directory)

        np.save(self.path, array)
        super(BufferedBinaryWriter, self).flush_buffer()
