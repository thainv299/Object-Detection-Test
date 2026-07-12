# DECISIONS

I built a real-time vehicle detection and tracking pipeline that runs from a street-view video clip. The focus was on making it actually viable on a low-power edge device — not just getting it to work, but getting it to work *efficiently*.

---

## 1. My understanding of the problem

The core ask is not "build the most accurate detector." It's "build something that could realistically run on a cheap camera with a weak CPU, no beefy GPU, and possibly no internet."

That reframes the whole problem. What matters most is throughput (can I sustain the video's frame rate?), latency (does the output lag behind the input?), and resource footprint (RAM, CPU load, power draw). What matters least is squeezing out extra mAP by using a heavier model — on a constrained device, a model that's 3% more accurate but 5x slower is simply unusable.
We must accept a clear engineering trade-off: **prioritizing real-time throughput and low latency over detecting very small or distant objects.** On an edge device, attempting to detect every tiny pixel-level object requires heavy models and large input resolutions that ruin frame rates. What matters most is stable FPS, low latency, and efficient resource utilization.

## 2. How I analysed and scoped it

I chose vehicle detection (cars, motorbikes, buses) from a static traffic camera angle — a realistic edge deployment scenario.

I scoped the project to the **inference pipeline only**: read video → run detection → track objects → render results. Everything else got cut:
- Scope & boundary: In a full production surveillance system, downstream components like database storage, event logging, and cloud API streaming are standard. However, for this edge challenge, I deliberately scoped the project strictly to the **core real-time inference loop** (video ingestion → preprocessing → detection + tracking → display rendering) to focus 100% on device-level optimization.
- No heavy trackers like DeepSORT. I use the built-in ByteTrack that ships with YOLO, which is lightweight enough to not blow the CPU budget.
- Why ByteTrack for tracking: I chose ByteTrack over traditional trackers (like SORT or DeepSORT). Conventional trackers discard low-confidence bounding boxes immediately, causing frequent track fragmentation when vehicles are partially occluded or blurred. ByteTrack retains low-confidence detections and associates them using spatial continuity, allowing stable tracking through occlusions **without requiring a heavy Re-ID neural network** that would overload edge CPUs.
- I deliberately skip auto-exporting TensorRT `.engine` files during the test run. In a real production deployment on NVIDIA hardware, TensorRT is absolutely the right choice — it's the fastest inference path available. But the export process takes 5–15 minutes, and I didn't want the reviewer to sit through that on first launch. Instead, the code checks if a pre-built `.engine` file already exists and uses it automatically; otherwise it falls back to ONNX or PyTorch.

## 3. Why I made my key choices

**Model — YOLO26 Nano (Custom Trained).** YOLO26 is the latest generation from Ultralytics, and I picked the Nano variant (`yolo26n`) specifically for edge deployment. 

Instead of using the default COCO pre-trained weights, **I fully fine-tuned the model (150 epochs, `freeze=0`) on a custom dataset of approximately 90,000 images** (similar to COCO format). Due to data security and privacy policies, the raw dataset cannot be made public. 

This custom model is optimized specifically for target surveillance tasks and detects 7 classes: `person`, `car`, `truck`, `bus`, `bicycle`, `motorcycle`, and `license_plate` (a class critical for traffic management that default COCO lacks).

Key advantages of this model choice:
- **NMS-free end-to-end inference** — the one-to-one detection head produces predictions directly without Non-Maximum Suppression as a post-processing step. That removes a whole chunk of CPU work that older YOLO versions needed.
- **DFL-free box regression** — the detection head is simpler and lighter than previous generations while maintaining unbounded regression range.
- **43% faster ONNX CPU inference** than YOLO11n on Intel Xeon CPU (per the official paper), which directly translates to better FPS on edge CPUs.
- **Targeted Domain Adaptation:** Fully training on 90k custom traffic/pedestrian images ensures high localization accuracy and reliability for our specific camera angles.

*(Detailed training metrics, confusion matrices, and loss curves are located in the [training_results/] directory).*

I considered MobileNet-SSD (lighter but noticeably worse accuracy on vehicle detection) and larger YOLO variants like YOLO26s/m (better mAP but too slow for real-time on a modest CPU). The Nano variant hits the sweet spot — 40.9 mAP on COCO at just 1.7ms on T4 TensorRT.

**Inference engine — dynamic selection.** Rather than hardcoding one engine, the system auto-detects what's available and picks the best option:
1. TensorRT (`.engine`) — if the file exists locally and the runtime is installed. Fastest option on NVIDIA hardware, but I never force the reviewer to wait for an export.
2. ONNX Runtime with GPU — if CUDA provider is available.
3. ONNX Runtime with CPU — the default fallback. Much lighter than PyTorch (~50MB vs >2GB install), and it leverages AVX/SSE vectorization well.
4. PyTorch (`.pt`) — last resort.

This means the same codebase runs on a laptop with a GPU, a desktop with only a CPU, or (in principle) a Jetson Nano — without any config changes.

**Multi-threaded pipeline.** I split the work into three stages:
- A reader thread that decodes video frames into a bounded queue.
- A processor thread that runs detection + tracking and pushes annotated frames to a display queue.
- The main thread that handles GUI rendering and frame-rate synchronization.

The key insight was moving `cv2.imshow` to the main thread. On Windows, running OpenCV GUI calls from a worker thread causes the event loop to stall, which dragged FPS down to ~20–25 even though the GPU was barely loaded. Moving display to main thread fixed this completely.

## 4. How I approached the constraints

**Real-time on weak hardware — Frame Skipping.** The biggest lever I have is simply not running the model on every single frame. The `FRAME_SKIP` parameter controls this: setting it to 1 means I only run detection on every other frame, setting it to 2 means every third frame, etc.

For the skipped frames, I reuse the bounding boxes from the last detected frame and draw them onto the current frame (`last_results.plot(img=current_frame)`). At 30 FPS, a vehicle moves very little between consecutive frames, so the visual result is smooth — the boxes stay on the objects and there's no flickering.

This cuts the inference workload by 50–70% with minimal visual degradation.

**Input size.** The model receives frames resized to its native training resolution (typically 640×640 for Nano), which keeps the computation per frame predictable and minimal.

**Hardware-accelerated video decoding (FFmpeg + NVDEC).** On devices with an NVIDIA GPU, the video reader thread automatically uses FFmpeg with the `h264_cuvid` hardware decoder (NVDEC) instead of OpenCV's CPU-based `cv2.VideoCapture`. This offloads H.264 decoding entirely to the GPU's dedicated decode chip, freeing the CPU for other tasks. The system auto-detects at startup whether FFmpeg and NVDEC are available via `ffprobe`; if not, it falls back gracefully to OpenCV CPU decoding.

**TensorRT acceleration.** On NVIDIA hardware, model inference runs through TensorRT with FP16 precision for maximum throughput. A dedicated `export_engine.py` script is provided for exporting `.engine` files on the target device.

**Offline operation.** Once deployed, the entire pipeline operates 100% locally on the edge device with zero internet dependency.
## 5. How I judged whether it works

Without ground-truth labels, I evaluated on two axes:

- **Performance:** I measure "Raw Capacity FPS" — the number of frames the processor thread can handle per second based purely on active compute time (excluding queue waits). When I set `FRAME_SKIP=1`, this number should roughly double vs `FRAME_SKIP=0`. If it doesn't, something is bottlenecked. I also check that "Display FPS" stays locked to the video's native frame rate (30 FPS).
- **Visual correctness:** I watch the output. Boxes should stick to vehicles, track IDs should be mostly stable, and there shouldn't be obvious missed detections at normal viewing distance. The boxes are drawn thin (`line_width=1`) so they don't obscure the scene.

## 6. The hardest part, and remaining limitations

The hardest part was optimizing the full pipeline to achieve real-time performance while maintaining detection accuracy. Getting the multi-threaded architecture right — decoupling video decoding, model inference, and GUI rendering into separate threads with proper synchronization — required careful tuning to avoid queue blocking and FPS measurement distortion.

**Remaining limitations:**
Currently, our edge pipeline handles single-stream detection and tracking smoothly. However, if we scale to support multiple camera streams or add heavier downstream analytics (e.g., License Plate Recognition, behavior analysis), the Python GIL becomes a real bottleneck. In a production multi-stream edge deployment (like NVIDIA Jetson), we would migrate to **DeepStream SDK** or **GStreamer pipelines** for true hardware-level parallelism across multiple video sources.

## 7. If the device were even more constrained

Two things I'd do first:

1. **Region of Interest (ROI).** Don't scan the full frame. Mask out the sky, sidewalks, and buildings — in a typical traffic camera view, that eliminates ~40% of the image area, which directly cuts inference time.
2. **INT8 Quantization.** Convert the ONNX model from float32 to int8. This roughly halves inference time on CPU and cuts model memory by 4x, at the cost of a small accuracy drop that's usually acceptable for vehicle-sized objects.

If the device is ARM-based (like a Jetson Nano), I'd also switch the video decoder to GStreamer with hardware-accelerated `nvv4l2decoder`, which offloads H.264 decoding to dedicated hardware and frees the CPU entirely for inference.

## 8. What I didn't have time to do

- **Rewrite the pipeline in C++.** Python's GIL limits true parallelism. A C++ implementation with direct ONNX Runtime C API calls would eliminate interpreter overhead and allow tighter thread synchronization.
- **Lightweight Optical Flow for skipped frames.** Instead of drawing static boxes from the previous detection, use Lucas-Kanade optical flow to shift the box coordinates in the direction of motion. This would make the boxes track movement even on skipped frames, with negligible CPU cost compared to running the full model.
- **Benchmarking on actual edge hardware.** I tested on a desktop machine. Profiling on a real Jetson Nano or Raspberry Pi would reveal different bottlenecks (memory bandwidth, thermal throttling) that might change the optimal `FRAME_SKIP` value or suggest different optimizations.
