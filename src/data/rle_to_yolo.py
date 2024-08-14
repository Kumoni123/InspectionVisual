import json
import os
import numpy as np
from pycocotools import mask as mask_util
import cv2

# Decodificar RLE a polígonos
def rle_to_polygon(rle, height, width):
    mask = mask_util.decode(rle)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.reshape(-1, 2) for contour in contours]
    return polygons

# Función para convertir la segmentación COCO a formato YOLOv8
def convert_coco_to_yolo(coco_data):
    yolo_annotations = {}

    for img in coco_data["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        width = img["width"]
        height = img["height"]

        yolo_annotations[file_name] = []

        for ann in coco_data["annotations"]:
            if ann["image_id"] == img_id:
                category_id = ann["category_id"]
                rle = ann["segmentation"]
                
                polygons = rle_to_polygon(rle, height, width)
                
                for polygon in polygons:
                    # Convertir los puntos del polígono a formato YOLO
                    normalized_polygon = []
                    for point in polygon:
                        x = point[0] / width
                        y = point[1] / height
                        normalized_polygon.append(x)
                        normalized_polygon.append(y)
                    
                    # Convertir el bbox a formato YOLO
                    bbox = ann["bbox"]
                    x_center = (bbox[0] + bbox[2] / 2) / width
                    y_center = (bbox[1] + bbox[3] / 2) / height
                    bbox_width = bbox[2] / width
                    bbox_height = bbox[3] / height

                    yolo_annotation = f"{category_id} {x_center} {y_center} {bbox_width} {bbox_height} {' '.join(map(str, normalized_polygon))}"
                    yolo_annotations[file_name].append(yolo_annotation)

    return yolo_annotations

#Dirección
#direc = 'C:/Repositorio_Belyeud/Visual_Inspection/archivos/project-2-at-2024-08-11-00-24-12ea8c0d.json'
direc = 'C:/Repositorio_Belyeud/Visual_Inspection/archivos/resultv2.json'
# Leer datos del archivo JSON en formato COCO
#with open('./data/labeled/project-5-at-2024-07-08-22-27-03d581a0/result.json') as f:
with open(direc) as f:
    coco_data = json.load(f)

# Convertir los datos al formato YOLOv8
yolo_data = convert_coco_to_yolo(coco_data)

# Guardar las anotaciones en archivos de texto
#output_dir = './data/labeled/project-5-at-2024-07-08-22-27-03d581a0/labels'
output_dir = 'C:/Repositorio_Belyeud/Visual_Inspection/archivos'

os.makedirs(output_dir, exist_ok=True)

for img_file, annotations in yolo_data.items():
    img_base = os.path.splitext(os.path.basename(img_file))[0]
    txt_file = os.path.join(output_dir, f"{img_base}.txt")

    with open(txt_file, "w") as f:
        for ann in annotations:
            f.write(f"{ann}\n")

print("Conversión completa.")
