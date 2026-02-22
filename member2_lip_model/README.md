# Member 2: Lip Reading Model (AI/ML)

## Deliverable
Function: lip frames → predicted text + confidence

## Files
- `src/lip_reader.py`: Main model implementation
- `requirements.txt`: Dependencies

## Usage
```python
from src.lip_reader import LipReadingModel, predict_from_lips

# Method 1: Streaming (for real-time)
model = LipReadingModel()
for frame in video_stream:
    if model.add_frame(lip_crop):
        result = model.predict()
        print(result['text'], result['confidence'])

# Method 2: Batch (for testing)
text, conf = predict_from_lips(lip_frames_list)
```
