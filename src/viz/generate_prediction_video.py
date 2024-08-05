from pathlib import Path
import os

import moviepy.video.io.ImageSequenceClip


if __name__ == "__main__":
    predictions_pth = Path(
        "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/sota_models/Revisiting-Reverse-Distillation/results/rails/rail/imgs"
    )
    fps = 3
    anomalies_predictions = sorted(
        [
            pred_pth.as_posix()
            for pred_pth in list(predictions_pth.iterdir())
            if "prediction" in str(pred_pth)
        ]
    )
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
        anomalies_predictions, fps=fps
    )
    clip.write_videofile(
        "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/sota_models/Revisiting-Reverse-Distillation/results/rails/rail/anomalies_predictions.mp4"
    )
