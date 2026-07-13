import os
import sys
import time
import queue
import threading
import subprocess
import shutil
import numpy as np
from load_data import check_and_prepare_assets, detect_best_backend, PATH_VIDEO

# FRAME_SKIP: Số lượng frame bỏ qua không nhận diện để giảm tải tính toán cho thiết bị yếu.
FRAME_SKIP = 1
# KIỂM TRA KHẢ NĂNG FFMPEG + NVDEC GPU DECODE
def _check_ffmpeg_nvdec():
    """Kiểm tra xem FFmpeg có sẵn và hỗ trợ h264_cuvid (NVDEC GPU decode) không."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-decoders"], capture_output=True, text=True, timeout=5
        )
        return "h264_cuvid" in result.stdout
    except Exception:
        return False

HAS_FFMPEG_NVDEC = _check_ffmpeg_nvdec()

def _get_video_info(video_path):
    """Dùng ffprobe lấy width, height, codec của video."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name",
             "-of", "csv=p=0:s=x", str(video_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=5
        )
        # Output format: "codec_namexwidthxheight"
        parts = result.stdout.strip().split("x")
        if len(parts) == 3:
            codec, w, h = parts[0], int(parts[1]), int(parts[2])
            return w, h, codec
    except Exception:
        pass
    return None, None, None

# 1. CÁC LUỒNG XỬ LÝ (MULTI-THREADING PIPELINE)
def video_reader_worker(video_path, raw_queue: queue.Queue, stop_event: threading.Event):
    """
    Luồng đọc video: Ưu tiên FFmpeg + NVDEC GPU decode nếu khả dụng,
    nếu không thì fallback về cv2.VideoCapture (CPU decode).
    """
    video_path = str(video_path)

    # Thử dùng FFmpeg + NVDEC GPU decode
    if HAS_FFMPEG_NVDEC:
        width, height, codec = _get_video_info(video_path)
        nvdec_codecs = {"h264": "h264_cuvid", "hevc": "hevc_cuvid", "vp9": "vp9_cuvid"}
        
        if width and height and codec in nvdec_codecs:
            decoder = nvdec_codecs[codec]
            print(f"[Thread-Reader] Sử dụng FFmpeg GPU decode ({decoder}) - {width}x{height}")
            
            cmd = [
                "ffmpeg",
                "-hwaccel", "cuda", # Bật tăng tốc phần cứng CUDA
                "-c:v", decoder,    # Chỉ định chip giải mã GPU (ví dụ: h264_cuvid)
                "-i", video_path,   # File video đầu vào
                "-f", "rawvideo",   # Xuất ra định dạng video thô (không nén)
                "-pix_fmt", "bgr24",# Định dạng màu BGR24 (giống hệt định dạng của OpenCV)
                "-v", "error",      # Chỉ hiện log nếu có lỗi
                "pipe:1"            # Ghi dữ liệu trực tiếp vào stdout (RAM) thay vì lưu ra file   
            ]
            
            frame_size = width * height * 3  # BGR24 = 3 bytes per pixel
            
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    bufsize=frame_size * 2
                )
                
                while not stop_event.is_set():
                    raw_bytes = proc.stdout.read(frame_size)
                    if len(raw_bytes) < frame_size:
                        print("[Thread-Reader] FFmpeg: Đã đọc hết video.")
                        break
                    
                    frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 3))
                    
                    try:
                        raw_queue.put(frame, timeout=1.0)
                    except queue.Full:
                        continue
                
                proc.stdout.close()
                proc.terminate()
                proc.wait(timeout=3)
                stop_event.set()
                print("[Thread-Reader] Đã dừng luồng đọc (FFmpeg GPU).")
                return
                
            except Exception as e:
                print(f"[Thread-Reader] FFmpeg GPU decode thất bại: {e}, chuyển sang OpenCV...")
        else:
            print(f"[Thread-Reader] Video codec '{codec}' không hỗ trợ NVDEC, dùng OpenCV fallback.")
    else:
        print("[Thread-Reader] FFmpeg/NVDEC không khả dụng, sử dụng OpenCV CPU decode.")

    # Fallback: OpenCV CPU decode
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở video tại {video_path}")
        stop_event.set()
        return

    print(f"[Thread-Reader] Bắt đầu đọc video (OpenCV CPU decode)...")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[Thread-Reader] Đã đọc hết video hoặc dừng.")
            break

        try:
            raw_queue.put(frame, timeout=1.0)
        except queue.Full:
            continue

    cap.release()
    stop_event.set()

def video_processor_worker(raw_queue: queue.Queue, display_queue: queue.Queue, 
                            stop_event: threading.Event, backend: str, model_path):
    """
    Luồng xử lý (Inference & Tracking): 
    Chỉ thực hiện tác vụ nặng (suy luận + tracking + vẽ box) và đẩy kết quả vào display_queue.
    """
    from ultralytics import YOLO

    print(f"[Thread-Processor] Khởi tạo mô hình từ {model_path} bằng backend '{backend}'...")
    
    device = "cpu"
    if backend in ["tensorrt", "onnx_gpu", "pytorch_gpu"]:
        device = 0 # Sử dụng GPU
        
    model = YOLO(str(model_path), task="detect")

    print("[Thread-Processor] Bắt đầu xử lý nhận diện & tracking...")
    raw_fps = 0.0
    frame_idx = 0
    last_results = None

    # Biến đo hiệu năng thuần (active processing time)
    active_time = 0.0
    processed_count = 0

    while not stop_event.is_set() or not raw_queue.empty():
        # Kiểm tra nhanh stop_event để thoát sớm nếu luồng chính đã gửi tín hiệu dừng
        if stop_event.is_set() and raw_queue.empty():
            break

        try:
            # Rút ngắn timeout để phát hiện stop_event nhanh hơn
            frame = raw_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        # Đo thời gian xử lý thực sự của frame (Inference + Plotting)
        t_start = time.time()

        # Kiểm tra xem frame hiện tại có bị bỏ qua nhận diện không
        if FRAME_SKIP > 0 and frame_idx % (FRAME_SKIP + 1) != 0:
            if last_results is not None:
                # Vẽ đè bounding box từ kết quả nhận diện của frame trước lên frame hiện tại
                annotated_frame = last_results.plot(img=frame, line_width=1, font_size=8)
            else:
                annotated_frame = frame
        else:
            # Thực hiện suy luận kèm Tracking
            results = model.track(frame, device=device, persist=True, verbose=False)
            last_results = results[0]
            # Vẽ bounding box lên frame
            annotated_frame = last_results.plot(line_width=1, font_size=8)

        frame_idx += 1
        
        # Cộng dồn thời gian xử lý tích cực
        active_time += (time.time() - t_start)
        processed_count += 1

        # Cập nhật Raw Capacity FPS mỗi 10 frame một lần
        if processed_count >= 10:
            if active_time > 0:
                raw_fps = processed_count / active_time
            processed_count = 0
            active_time = 0.0

        # Đưa frame đã vẽ cùng Raw FPS vào display_queue để luồng chính hiển thị
        try:
            # Rút ngắn timeout từ 1.0 xuống 0.05s để tránh nghẽn khi luồng chính đã tắt
            display_queue.put((annotated_frame, raw_fps), timeout=0.05)
        except queue.Full:
            continue

    stop_event.set()
    print("[Thread-Processor] Đã dừng luồng xử lý.")

# 2. HÀM CHẠY CHÍNH (MAIN THREAD - GUI & RATE LIMIT)

def main():
    print("Khởi động hệ thống NVIT Object Detection & Tracking Test...")
    
    # Bước 1: Kiểm tra và chuẩn bị dữ liệu (tải tự động nếu thiếu)
    check_and_prepare_assets()

    try:
        import cv2
    except ImportError:
        print("[LỖI] Vui lòng cài đặt opencv-python trước (pip install opencv-python)")
        sys.exit(1)

    # Đọc FPS gốc của video
    cap = cv2.VideoCapture(str(PATH_VIDEO))
    if not cap.isOpened():
        print(f"[LỖI] Không thể đọc video tại {PATH_VIDEO}.")
        sys.exit(1)
    
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    if fps_original <= 0 or fps_original > 120:
        fps_original = 30.0
    cap.release()

    # Bước 2: Tự động chọn Backend tối ưu nhất
    backend, model_path = detect_best_backend()
    print(f"\n[KHỞI TẠO] Chọn backend tối ưu nhất: {backend.upper()} sử dụng {model_path.name}")

    # Khởi tạo các hàng đợi và sự kiện dừng
    raw_queue = queue.Queue(maxsize=30)
    display_queue = queue.Queue(maxsize=30)
    stop_event = threading.Event()

    # Khởi chạy luồng đọc và luồng xử lý
    reader_thread = threading.Thread(
        target=video_reader_worker, 
        args=(PATH_VIDEO, raw_queue, stop_event),
        name="ReaderThread"
    )
    processor_thread = threading.Thread(
        target=video_processor_worker, 
        args=(raw_queue, display_queue, stop_event, backend, model_path),
        name="ProcessorThread"
    )

    reader_thread.start()
    processor_thread.start()

    # LUỒNG CHÍNH: HIỂN THỊ GUI & ĐỒNG BỘ FPS (RATE LIMIT)
    ideal_frame_time = 1.0 / fps_original
    fps_start_time = time.time()
    fps_counter = 0
    display_fps = 0.0
    
    # Thời điểm hiển thị frame trước đó để căn chỉnh sleep chính xác
    last_display_time = time.time()

    print("[Main-Thread] Đang khởi chạy luồng hiển thị GUI...")
    try:
        while not stop_event.is_set() or not display_queue.empty():
            try:
                # Lấy frame và Raw FPS từ queue
                annotated_frame, raw_fps = display_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Tính toán FPS thực tế hiển thị
            fps_counter += 1
            now = time.time()
            elapsed = now - fps_start_time
            if elapsed >= 1.0:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = now

            # Ghi thông số với chữ trắng đậm (thickness=2, fontScale=0.55) cho dễ nhìn
            cv2.putText(annotated_frame, f"Backend: {backend.upper()}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Display FPS: {display_fps:.2f} / Video FPS: {fps_original:.2f}", (15, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Raw Capacity FPS: {raw_fps:.2f}", (15, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # Hiển thị frame
            cv2.imshow("NVIT Object Detection & Tracking", annotated_frame)
            
            # Nhấn phím 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            # Tối ưu hóa giới hạn tốc độ hiển thị bằng thời gian trôi qua thực tế
            display_elapsed = time.time() - last_display_time
            wait_time = ideal_frame_time - display_elapsed
            if wait_time > 0:
                time.sleep(wait_time)
                
            last_display_time = time.time()

    except KeyboardInterrupt:
        print("\n[HỦY] Người dùng yêu cầu dừng bằng Ctrl+C.")
        stop_event.set()

    cv2.destroyAllWindows()
    reader_thread.join()
    processor_thread.join()
if __name__ == "__main__":
    main()
