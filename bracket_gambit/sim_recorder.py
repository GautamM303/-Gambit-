"""Video of a simulated game: the status display (3-D scene, hand camera, head camera,
board) written at a fixed frame rate. The scene/hand renders come from SimRobot's Views."""
from __future__ import annotations

import cv2


class SimRecorder:
    def __init__(self, display, video_path, fps=30):
        import imageio
        self.display = display
        self.writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
        self.fps, self.next_t, self.frames = fps, 0.0, 0
        self.path = video_path
        self.robot = None

    def attach(self, robot):
        self.robot = robot

    def use_camera(self, name):
        if self.robot is not None:
            self.robot.use_camera(name)

    def tick(self, data):
        if data.time < self.next_t:
            return
        frame = self.display.compose()
        self.writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.next_t += 1 / self.fps
        self.frames += 1

    def snapshot(self, data, path):
        cv2.imwrite(str(path), self.display.compose())

    def close(self):
        self.writer.close()
        print(f"wrote {self.path} ({self.frames} frames)", flush=True)
