# Object Detection Project

This project aims to detect Mice and Cats in images using the YOLOv8 model.

## Training

The model was initially trained using the `dataset_original` which is configured via `data.yaml`.

To improve the model's performance and train with a larger dataset, the training configuration was updated to use the `dataset_enhanced` dataset, which is configured via `data_enhanced.yaml`. This enhanced dataset includes a greater number of images.

The training was executed using the `train.py` script with the following parameters:
- Dataset: `data_enhanced.yaml`
- Epochs: 2
- Image Size: 640
- Batch Size: 12
- Augmentation: Enabled

The training process completed successfully, and the results, including the trained model weights (`best.pt`, `last.pt`), plots, and metrics, are saved in the `runs/detect/mouse_and_cat_detection2` directory.

## Further Steps

You can continue to train the model with more epochs, or fine-tune other hyperparameters in `train.py` to further improve its performance. You can also evaluate the model using the `val.py` script or perform inference on new images using the `predict.py` script (if available).