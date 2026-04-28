from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config
import json
from os import path
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config',      default='config.json')
parser.add_argument('--output_dir',  default='/project/def-sponsor00/remilalonde/outputs/climatenet_run_engineered_data')
parser.add_argument('--uncertainty', action='store_true', help='Run MC Dropout uncertainty inference after training')
parser.add_argument('--n_passes',    type=int, default=30, help='Number of MC Dropout forward passes')
parser.add_argument('--save_preds',  action='store_true', help='Save predictions.nc after training')
args = parser.parse_args()

config = Config(args.config)
cgnet = CGNet(config)

train_path = '/project/def-sponsor00/shared_CN_B/climatenet_engineered'
inference_path = '/project/def-sponsor00/shared_CN_B/climatenet_engineered/test'

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


train = ClimateDatasetLabeled(path.join(train_path, 'train'), config)
test = ClimateDatasetLabeled(path.join(train_path, 'test'), config)
inference = ClimateDataset(inference_path, config)

history = cgnet.train(train, test)
metrics = cgnet.evaluate(test)


history_file = path.join(output_dir, "training_history.json")

with open(history_file, "w") as f:
    json.dump(history, f, indent=4)

print(f"Training history saved to {history_file}")


metrics_file = path.join(output_dir, "metrics.json")
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {metrics_file}")


cgnet.save_model(path.join(output_dir, 'trained_cgnet'))

if args.save_preds and not args.uncertainty:
    print("\nSaving predictions...")
    preds = cgnet.predict(inference)
    preds.to_netcdf(path.join(output_dir, 'predictions.nc'))
    print(f"Predictions saved to {output_dir}")

if args.uncertainty:
    print(f"\nRunning MC Dropout uncertainty inference ({args.n_passes} passes)...")
    mean_preds, uncertainty = cgnet.predict_with_uncertainty(inference, n_passes=args.n_passes)
    mean_preds.load().to_netcdf(path.join(output_dir, 'predictions.nc'))
    uncertainty.load().to_netcdf(path.join(output_dir, 'uncertainty.nc'))
    print(f"Predictions and uncertainty maps saved to {output_dir}")
# use a saved model with
# cgnet.load_model('trained_cgnet')

# plt.figure()
# plt.plot(history["train_losses"], label="Train Loss")
# plt.plot(history["val_losses"], label="Val Loss")
# plt.legend()
# plt.title("Loss Curve")
# plt.show()

# plt.figure()
# plt.plot(history["train_mean_ious"], label="Train mIoU")
# plt.plot(history["val_mean_ious"], label="Val mIoU")
# plt.legend()
# plt.title("Mean IoU Curve")
# plt.show()

# class_names = ["Background", "TC", "AR"]

# train_ious = np.array(history["train_ious_per_class"])
# val_ious = np.array(history["val_ious_per_class"])

# num_classes = train_ious.shape[1]

# plt.figure()

# for c in range(num_classes):
#     plt.plot(train_ious[:, c], label=f"Train {class_names[c]}")
#     plt.plot(val_ious[:, c], linestyle="--", label=f"Val {class_names[c]}")

# plt.legend()
# plt.xlabel("Epoch")
# plt.ylabel("IoU")
# plt.title("IoU per Class")
# plt.show()

#class_masks = cgnet.predict(inference) # masks with 1==TC, 2==AR
#event_masks = track_events(class_masks) # masks with event IDs

#analyze_events(event_masks, class_masks, path.join(output_dir, 'results'))
#visualize_events(event_masks, inference, 'pngs/')
