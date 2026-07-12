from ultralytics import YOLO
model = YOLO(r"models/model.pt")
model.export(
    format="engine",
    half=True,        
    dynamic=True,     
    batch=4,          
    device=0,
    workspace=4       
)