import csv

import cv2
import numpy as np
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

# ------------------------------
# 1. Load SAM model
# ------------------------------
sam_checkpoint = "models/sam_vit_b.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

# ------------------------------
# 2. Load OpenCLIP model
# ------------------------------
model, _, preprocess = create_model_and_transforms("ViT-B-32", pretrained="openai")
model.eval()
model.cuda() if torch.cuda.is_available() else model.cpu()
tokenizer = get_tokenizer("ViT-B-32")

# ------------------------------
# 3. Define candidate labels
# ------------------------------
candidate_labels = [
    "a puma",
    "a snake",
    "a coyote",
    "an iguana",
    "an ocelot",
    "a squirrel",
    "a bird",
    "an otter",
    "a coati",
    "a tamandua",
    "a lizard",
    "a bat",
    "a butterfly",
    "a tayra",
    "a monkey",
    "a cougar",
    "a paca",
    "a raccoon",
    "a skunk",
    "a dog",
    "an agouti",
    "a cat",
    "a mouse",
    "a turkey",
]
text_tokens = (
    tokenizer(candidate_labels).cuda()
    if torch.cuda.is_available()
    else tokenizer(candidate_labels)
)

# ------------------------------
# 4. Start video processing
# ------------------------------
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

# ------------------------------
# 5. CSV logging
# ------------------------------
csvfile = open("video_log.csv", "w", newline="")
csvwriter = csv.writer(csvfile)
csvwriter.writerow(["frame_id", "label", "confidence", "x", "y", "w", "h"])

# ------------------------------
# 6. Frame-by-frame loop
# ------------------------------
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).contiguous()
    image_tensor = image_tensor.cuda() if torch.cuda.is_available() else image_tensor

    H, W = image_rgb.shape[:2]
    input_box = np.array([0, 0, W, H])  # entire image
    masks, scores, logits = predictor.predict(
        box=input_box[None, :], multimask_output=True
    )

    for i, mask in enumerate(masks):
        area = np.sum(mask)
        if area < 5000:  # filter small segments
            continue

        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        cropped = image_rgb[y : y + h, x : x + w]
        pil_crop = Image.fromarray(cropped)
        image_input = preprocess(pil_crop).unsqueeze(0)
        image_input = image_input.cuda() if torch.cuda.is_available() else image_input

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0)

        best_idx = similarity.argmax().item()
        best_label = candidate_labels[best_idx]
        best_score = similarity[best_idx].item()

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_text = f"{best_label} ({best_score:.2f})"
        cv2.putText(
            frame,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Log to CSV
        csvwriter.writerow([frame_id, best_label, round(best_score, 3), x, y, w, h])

    out.write(frame)
    print(f"Processed frame {frame_id}")

# ------------------------------
# 7. Cleanup
# ------------------------------
cap.release()
out.release()
csvfile.close()
print(
    "✅ Video processing complete. Output saved to output_video.mp4 and video_log.csv"
)
