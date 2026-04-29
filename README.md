# IFT6759-Project

## Abstract
Extreme weather events such as rivers (AR) or tropical cyclones (TC) are becoming increasingly frequent due to global warming, hence the importance of developing automated methods to detect them. This work focuses on the semantic segmentation of TCs and ARs in the ClimateNet dataset, which suffers from severe class imbalance. By building upon the CGNet architecture used by KappSchwoerer et al. (2020) and the work of Lacombe et al. (2023), two approaches were explored: (1) sampling methods involving file and patch selection to reduce class imbalance, and (2) alternative architectures, notably U-Net and ConvLSTM, to better capture the temporal dimension of climate events. The second approach proved more effective, particularly as the U-Net Attention architecture achieved the best overall performance. This architecture yielded a TC IoU of 0.3765, representing a 0.027 increase over the baseline for this class, at the cost of a slight decline in AR segmentation performance. The results obtained confirm that the main challenge remains the extreme class imbalance, particularly for TCs, and that exploring more suitable architectures is more promising than data sampling methods.

Translated with DeepL.com (free version)

## How to execute
1. Install Python, we recommend using Python 3.13.x.
2. Create a virtual env and install the requirements with `pip install -r requirements.txt`.
3. Download the ClimateNet dataset with `data/climatenet/data_downloader.py`.
4. Training can be done on the cluster by locating the `.sh` bash files or done locally by running the Python training scripts directly.
5. Use the newly trained models or locate the saved `.pth` model checkpoint files.
6. Run inference on ClimateNet eval to evaluate the capabilites of your model.
7. Use the post-processing scheme to visualize the predictions of your model.

## Project Structure
```text
ConvLSTM/             # Code and checkpoints for the ConvLSTM models
baselines/            # Reproduction of the baseline models and results from Kapp-Schwoerer et al. (2020)
climatenet/           # Copy of the ClimateNet/climatenet folder in https://github.com/andregraubner/ClimateNet 
data/                 # ClimateNet dataset, feature engineering and data reduction strategies
documents/            # Submitted homework PDFs and referenced papers
hybrid/               # Hybrid ConvLSTM architectures
post-processing/      # Post-processing system + results and ERA5 dataset
results/ConvLSTM/     # ConvLSTM training results
unets/                # U-Net training, models and results               
```

## Post-processing examples
### Attention U-Net worlwide AR tracking on September 2022's hurricane season
![AR tracking september 2022](post_processing/animations/2022-09-26-29_ar_tracking.gif)
### Attention U-Net tracking hurricane Erin's evolution in the Carribeans during August 2025 
![Erin tracking](post_processing/animations/hurricane_erin_tc_tracking.gif)