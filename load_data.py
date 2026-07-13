import os
import sys
import requests
from pathlib import Path

# --- CẤU HÌNH ĐƯỜNG DẪN TẢI FILE TỪ GOOGLE DRIVE ---
GOOGLE_DRIVE_IDS = {
    "model.onnx": "1fcXZdV8_Uw7zSVyvVaMPgT0IhCOcg6iX",    
    "model.pt": "1bkpA9UZkEOnsYrA_N_6R7roczhpjlAlY",      
    "video.mp4": "1W4fOxdJkZ8f0Agzi8pr0_1x0bCeEmI27"     
}

# --- CẤU HÌNH THƯ MỤC LƯU TRỮ ---
MODEL_DIR = Path("models")
DATA_DIR = Path("data")

PATH_MODEL_ENGINE = MODEL_DIR / "model.engine"
PATH_MODEL_ONNX = MODEL_DIR / "model.onnx"
PATH_MODEL_PT = MODEL_DIR / "model.pt"
PATH_VIDEO = DATA_DIR / "video.mp4"

def download_file_from_google_drive(file_id: str, destination: Path):
    """
    Tải file từ Google Drive hỗ trợ xác nhận quét virus cho file lớn.
    """
    if not file_id:
        print(f"[CẢNH BÁO] ID Google Drive cho {destination.name} trống. Bỏ qua tải tự động cho file này.")
        return False

    print(f"\n[DOWNLOAD] Bắt đầu tải {destination.name} từ Google Drive (ID: {file_id})...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    try:
        response = session.get(URL, params={'id': file_id}, stream=True)
        
        # Tìm token xác nhận nếu file dung lượng lớn
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
                
        if token:
            response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
            
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rTiến độ: {percent:.2f}% ({downloaded}/{total_size} bytes)", end="")
                    else:
                        print(f"\rĐã tải: {downloaded} bytes", end="")
        print(f"\n[THÀNH CÔNG] Đã lưu file tại: {destination}")
        return True
    except Exception as e:
        print(f"\n[LỖI] Không thể tải file: {e}")
        if destination.exists():
            destination.unlink()
        return False

def check_and_prepare_assets():
    """
    Kiểm tra sự tồn tại của các file model và video. Tiến hành tải nếu thiếu.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Kiểm tra model.engine
    if PATH_MODEL_ENGINE.exists():
        print(f"[OK] Đã tìm thấy file mô hình TensorRT cục bộ: {PATH_MODEL_ENGINE}")
    else:
        print(f"[THÔNG BÁO] Không có file TensorRT cục bộ ({PATH_MODEL_ENGINE}). Sẽ tự động dùng ONNX/PyTorch thay thế.")

    # 2. Kiểm tra model.onnx
    if not PATH_MODEL_ONNX.exists():
        print(f"[THÔNG BÁO] Không tìm thấy {PATH_MODEL_ONNX}")
        download_file_from_google_drive(GOOGLE_DRIVE_IDS["model.onnx"], PATH_MODEL_ONNX)
    else:
        print(f"[OK] Đã có file: {PATH_MODEL_ONNX}")

    # 3. Kiểm tra model.pt
    if not PATH_MODEL_PT.exists():
        print(f"[THÔNG BÁO] Không tìm thấy {PATH_MODEL_PT}")
        download_file_from_google_drive(GOOGLE_DRIVE_IDS["model.pt"], PATH_MODEL_PT)
    else:
        print(f"[OK] Đã có file: {PATH_MODEL_PT}")

    # 4. Kiểm tra video.mp4
    if not PATH_VIDEO.exists():
        print(f"[THÔNG BÁO] Không tìm thấy {PATH_VIDEO}")
        download_file_from_google_drive(GOOGLE_DRIVE_IDS["video.mp4"], PATH_VIDEO)
    else:
        print(f"[OK] Đã có file: {PATH_VIDEO}")

def detect_best_backend():
    """
    Tự động phát hiện thư viện và phần cứng để chọn backend tối ưu nhất theo thứ tự:
    1. TensorRT (.engine)
    2. ONNX Runtime GPU
    3. ONNX Runtime CPU
    4. PyTorch GPU
    5. PyTorch CPU
    """
    has_trt_lib = False
    has_onnx = False
    has_ultralytics = False
    onnx_providers = []
    has_cuda_torch = False

    # Kiểm tra thư viện TensorRT gốc
    try:
        import tensorrt
        has_trt_lib = True
    except ImportError:
        pass

    # Kiểm tra ONNX Runtime
    try:
        import onnxruntime as ort
        has_onnx = True
        onnx_providers = ort.get_available_providers()
    except ImportError:
        pass

    # Kiểm tra PyTorch / Ultralytics
    try:
        import torch
        from ultralytics import YOLO
        has_ultralytics = True
        has_cuda_torch = torch.cuda.is_available()
    except ImportError:
        pass

    print("\n" + "="*50)
    print("THÔNG TIN PHẦN CỨNG & PHẦN MỀM PHÁT HIỆN ĐƯỢC:")
    print(f"- Cài đặt thư viện TensorRT: {'Có' if has_trt_lib else 'Không'}")
    print(f"- Cài đặt ONNX Runtime: {'Có' if has_onnx else 'Không'}")
    if has_onnx:
        print(f"  Các providers khả dụng: {onnx_providers}")
    print(f"- Cài đặt Ultralytics/Torch: {'Có' if has_ultralytics else 'Không'}")
    if has_ultralytics:
        print(f"  Khả dụng CUDA (GPU) trong Torch: {'Có' if has_cuda_torch else 'Không'}")
    print("="*50)

    # 1. Kiểm tra TensorRT (.engine)
    # Cần file .engine tồn tại và có thư viện TensorRT hoặc TensorrtExecutionProvider của ONNX Runtime
    if PATH_MODEL_ENGINE.exists() and (has_trt_lib or (has_onnx and 'TensorrtExecutionProvider' in onnx_providers)):
        return "tensorrt", PATH_MODEL_ENGINE

    # 2. Kiểm tra ONNX Runtime GPU (CUDA)
    if PATH_MODEL_ONNX.exists() and has_onnx and 'CUDAExecutionProvider' in onnx_providers:
        return "onnx_gpu", PATH_MODEL_ONNX

    # 3. Kiểm tra ONNX Runtime CPU
    if PATH_MODEL_ONNX.exists() and has_onnx:
        return "onnx_cpu", PATH_MODEL_ONNX

    # 4. Kiểm tra PyTorch GPU
    if PATH_MODEL_PT.exists() and has_ultralytics and has_cuda_torch:
        return "pytorch_gpu", PATH_MODEL_PT

    # 5. Kiểm tra PyTorch CPU
    if PATH_MODEL_PT.exists() and has_ultralytics:
        return "pytorch_cpu", PATH_MODEL_PT

    # Fallbacks dự phòng chéo
    if PATH_MODEL_ONNX.exists() and has_onnx:
        return "onnx_cpu", PATH_MODEL_ONNX
    if PATH_MODEL_PT.exists() and has_ultralytics:
        return "pytorch_cpu", PATH_MODEL_PT

    print("[LỖI] Không tìm thấy file mô hình phù hợp hoặc thiếu thư viện để thực thi!")
    sys.exit(1)
