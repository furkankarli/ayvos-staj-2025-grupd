# YOLOv8-seg Semantic and Instance Segmentation Example

## Description

This pull request introduces an example script demonstrating semantic and instance segmentation using the **YOLOv8-seg** model.

- Added the `yolov8_seg_example.py` script.  
- Used a sample image (`cat.jpg`) for demonstration.  
- Performed semantic and instance segmentation on the same image.  
- Created a combined visualization saved as `output.jpg`.  

---

## What was done

- Performed instance segmentation using YOLOv8-seg.  
- Generated semantic segmentation-like masks for specific classes.  
- Visualized original, instance, and semantic segmentations side by side.  
- Ensured code styling compliance via `black`, `isort`, and `flake8` pre-commit checks.  

---

## Checks

- Passed all pre-commit hooks (`black`, `isort`, `flake8`).  
- Generated and reviewed output image (`output.jpg`).  
- Ready for merging after at least two team members review.  

---

## Notes

- The script is designed for simplicity and clarity to aid understanding of YOLOv8 segmentation capabilities.  
- The output image shows the original input, instance segmentation, and semantic segmentation results side by side.

---

Thank you for reviewing this contribution!
