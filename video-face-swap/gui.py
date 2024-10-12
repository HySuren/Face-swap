import os
import sys
import cv2
import insightface
from insightface.app import FaceAnalysis
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QIcon
import shutil
import time
import moviepy.editor as mp
import numpy as np
from gfpgan import GFPGANer
from tqdm import tqdm
import asyncio

VIDEO_FRAMES_DIRECTORY = "_tmp_frames"
PROCESSED_FRAMES_DIRECTORY = "_tmp_frames_out"

GFPGAN_MODEL_CHECKPOINT = "models/GFPGANv1.4.pth"
INSWAPPER_MODEL_CHECKPOINT = "models/inswapper_128.onnx"

class ProcessingThread(QtCore.QThread):
    update_progress = QtCore.Signal(int)
    processing_finished = QtCore.Signal()

    def __init__(self, video_path, face_path, output_path, restore):
        super().__init__()
        self.video_path = video_path
        self.face_path = face_path
        self.output_path = output_path
        self.restore = restore
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(INSWAPPER_MODEL_CHECKPOINT, download=False, download_zip=False)
        self.gfpgan = None
        if self.restore:
            self.gfpgan = GFPGANer(model_path=GFPGAN_MODEL_CHECKPOINT, upscale=1)

        # Добавляем переменную для хранения аудио
        self.audio_clip = None

    def run(self):
        asyncio.run(self.process_video())

    async def process_video(self):
        # Удаляем старые директории
        shutil.rmtree(VIDEO_FRAMES_DIRECTORY, ignore_errors=True)
        shutil.rmtree(PROCESSED_FRAMES_DIRECTORY, ignore_errors=True)

        os.makedirs(VIDEO_FRAMES_DIRECTORY, exist_ok=True)
        os.makedirs(PROCESSED_FRAMES_DIRECTORY, exist_ok=True)

        # Загружаем исходное изображение лица
        source_img = cv2.imread(self.face_path)
        source_faces = self.app.get(source_img)

        # Извлекаем аудио из оригинального видео
        await asyncio.to_thread(self.extract_audio, self.video_path)

        # Разделяем видео на кадры
        frames, video_fps = await asyncio.to_thread(self.video_to_images, self.video_path)

        # Обрабатываем кадры параллельно
        await asyncio.gather(*[self.process_frame_async(index, frame, source_faces, len(frames)) for index, frame in enumerate(frames)])

        # Собираем видео обратно
        await asyncio.to_thread(self.images_to_video, video_fps, self.output_path)

        # Удаляем временные директории
        shutil.rmtree(VIDEO_FRAMES_DIRECTORY, ignore_errors=True)
        shutil.rmtree(PROCESSED_FRAMES_DIRECTORY, ignore_errors=True)

        self.processing_finished.emit()

    async def process_frame_async(self, index, frame, source_faces, total_frames):
        await asyncio.to_thread(self.process_image, frame, source_faces)
        progress = int((index + 1) / total_frames * 100)
        self.update_progress.emit(progress)

    def extract_audio(self, video_file_path):
        # Извлечение аудио из исходного видео
        video = mp.VideoFileClip(video_file_path)
        self.audio_clip = video.audio  # Сохраняем аудио для последующего добавления

    def process_image(self, image_file_name, source_faces):
        target_img_path = os.path.join(VIDEO_FRAMES_DIRECTORY, image_file_name)
        target_img = cv2.imread(target_img_path)
        target_faces = self.app.get(target_img)

        res = target_img.copy()
        for face in target_faces:
            # Face swapping
            res = self.swapper.get(res, face, source_faces[0], paste_back=True)

            # Face restoration
            if self.restore and self.gfpgan:
                _, _, res = self.gfpgan.enhance(np.array(res, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)

        output_path = os.path.join(PROCESSED_FRAMES_DIRECTORY, f"output_{image_file_name}")
        cv2.imwrite(output_path, res)

    def video_to_images(self, video_file_path):
        cap = cv2.VideoCapture(video_file_path)
        list_files = []
        frame_count = 0
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(VIDEO_FRAMES_DIRECTORY, f"{frame_count:07d}.png")
            list_files.append(f"{frame_count:07d}.png")  # Сохраняем только имя файла
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        return list_files, original_fps

    def images_to_video(self, fps, output_file):
        images = [os.path.join(PROCESSED_FRAMES_DIRECTORY, img) for img in sorted(os.listdir(PROCESSED_FRAMES_DIRECTORY))]
        clip = mp.ImageSequenceClip(images, fps=fps)

        # Добавляем сохраненное аудио к видео
        if self.audio_clip:
            clip = clip.set_audio(self.audio_clip)

        clip.write_videofile(output_file, fps=fps)

class FaceSwapApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.video_path = ""
        self.face_path = ""
        self.output_path = ""
        self.restore = False  # Опция восстановления лица
        self.total_start_time = time.time()

    def init_ui(self):
        self.setWindowTitle("Face-Swap")
        self.setGeometry(100, 100, 500, 400)
        self.setWindowIcon(QIcon("icon.png"))

        layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel("Выберите видео и изображение для замены лиц", self)
        layout.addWidget(self.label)

        self.video_button = QtWidgets.QPushButton("Выбрать видео | Select video", self)
        self.video_button.clicked.connect(self.load_video)
        layout.addWidget(self.video_button)

        self.face_button = QtWidgets.QPushButton("Выбрать изображение | Select image", self)
        self.face_button.clicked.connect(self.load_face)
        layout.addWidget(self.face_button)

        self.output_button = QtWidgets.QPushButton("Сохранить как... | Save as...", self)
        self.output_button.clicked.connect(self.load_output)
        layout.addWidget(self.output_button)

        self.restore_checkbox = QtWidgets.QCheckBox("Восстановление лица | Face restoration", self)
        self.restore_checkbox.stateChanged.connect(self.toggle_restore)
        layout.addWidget(self.restore_checkbox)

        self.start_button = QtWidgets.QPushButton("Старт | Start", self)
        self.start_button.clicked.connect(self.start_swap)
        layout.addWidget(self.start_button)

        self.status_label = QtWidgets.QLabel("", self)
        layout.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.console_output = QtWidgets.QTextEdit(self)
        self.console_output.setReadOnly(True)
        layout.addWidget(self.console_output)

        self.setLayout(layout)

    def load_video(self):
        self.video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video files (*.mp4 *.mov)")
        if self.video_path:
            self.status_label.setText(f"Выбрано видео: {os.path.basename(self.video_path)}")

    def load_face(self):
        self.face_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Image files (*.jpg *.jpeg *.png)")
        if self.face_path:
            self.status_label.setText(f"Выбрано изображение: {os.path.basename(self.face_path)}")

    def load_output(self):
        self.output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить выходной файл", "", "Video files (*.mp4)")
        if self.output_path:
            self.status_label.setText(f"Выходной файл: {os.path.basename(self.output_path)}")

    def toggle_restore(self, state):
        self.restore = state == QtCore.Qt.Checked

    def start_swap(self):
        if not self.video_path or not self.face_path or not self.output_path:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Пожалуйста, выберите видео, изображение и выходной файл.")
            return

        self.status_label.setText("Начинается обработка...")
        self.progress_bar.setValue(0)

        # Создаем и запускаем поток обработки
        self.processing_thread = ProcessingThread(self.video_path, self.face_path, self.output_path, self.restore)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.processing_finished)

        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def processing_finished(self):
        self.status_label.setText("Обработка завершена.")
        total_time = time.time() - self.total_start_time
        self.console_output.append(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FaceSwapApp()
    window.show()
    sys.exit(app.exec())