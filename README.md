# IFT6759-Project

## Abstract (FR)
Que ce soient les rivières atmosphériques (AR) ou les cyclones tropicaux (TC), les événements climatiques extrêmes sont de plus en plus fréquents en raison du réchauffement climatique, d’où l’importance du développement de méthodes automatiques afin de les détecter. Ce travail se concentre sur la segmentation sémantique de TC et d’AR dans le jeu de données ClimateNet, qui possède un déséquilibre sévère de classes. En itérant sur l’architecture CGNet utilisée par KappSchwoerer et al. (2020) et sur les travaux de Lacombe et al. (2023), deux approches ont été explorées : (1) des méthodes d’échantillonnage par sélection de fichiers et de patches afin de réduire le déséquilibre des classes, et (2) des architectures alternatives, notamment les U-Net et les ConvLSTM, pour mieux capturer la dimension temporelle des événements climatiques. La seconde approche a été plus efficace, surtout avec l'obtention des meilleures performances globales par l'Attention U-Net. Cette architecture a permis d’obtenir un IoU TC de 0.3765, soit une augmentation de 0.027 par rapport à la baseline sur cette classe, au prix d’un léger recul sur la segmentation des AR. Les résultats obtenus soulignent que l’enjeu principal reste le déséquilibre extrême des classes, particulièrement sur les TC, et que l’exploration d’architectures plus adaptées est plus prometteuse que les méthodes d’échantillonnage de données.

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