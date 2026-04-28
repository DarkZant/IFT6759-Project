import numpy as np
from scipy.ndimage import label

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from PIL import Image

from typing import Tuple

from models.attention_unet import AttentionUNet
from models.train_attention_unet import ClimateNetDataset, FeatureSets
from post_processing.arco_dataset import ARCOStreamDataset



def track_climate_events(
    binary_predictions: np.ndarray, 
    min_area: int=250, 
    min_duration_hours: int=12, 
    hours_per_timestep: int=3
) -> np.ndarray:
    """
    Tracks climate events over time based on spatial overlap.
    
    Args:
        binary_predictions: Numpy array of shape (T, H, W) with boolean/binary 
                            predictions for a single class (1 for event, 0 for bg).
        min_area: Minimum pixel area to keep a connected component.
        min_duration_hours: Minimum hours an event must persist to be kept.
        hours_per_timestep: Temporal resolution of the data (ClimateNet is usually 3-hourly).
        
    Returns:
        tracked_events: Numpy array of shape (T, H, W) where pixels are labeled 
                        with unique integer Event IDs. Background is 0.
    """
    T, H, W = binary_predictions.shape
    tracked_events = np.zeros_like(binary_predictions, dtype=np.int32)
    
    current_max_id = 0
    # Dictionary to keep track of which timesteps each ID exists in
    id_lifespan = {}

    # Step 1-3: Spatial Filtering and Temporal Tracking
    for t in range(T):
        # Find connected components in the current timestep
        labeled_array, num_features = label(binary_predictions[t])
        current_t_ids = np.zeros_like(labeled_array, dtype=np.int32)

        for comp_idx in range(1, num_features + 1):
            mask = (labeled_array == comp_idx)
            area = np.sum(mask)

            # Spatial filter: Remove small components
            if area < min_area:
                continue 

            assigned_id = None
            
            # Check for overlap with the previous timestep
            if t > 0:
                # Get the IDs from the previous timestep exactly where our current component is
                overlapping_ids = tracked_events[t-1][mask]
                overlapping_ids = overlapping_ids[overlapping_ids > 0] # Filter out background (0)

                if len(overlapping_ids) > 0:
                    # If it overlaps with multiple previous events, inherit the ID with the most overlap
                    assigned_id = np.bincount(overlapping_ids).argmax()

            # If no overlap found (or it's the first timestep), assign a new ID
            if assigned_id is None:
                current_max_id += 1
                assigned_id = current_max_id

            # Apply the assigned ID to the current component
            current_t_ids[mask] = assigned_id

            # Record that this ID was seen at timestep 't'
            if assigned_id not in id_lifespan:
                id_lifespan[assigned_id] = set()
            id_lifespan[assigned_id].add(t)

        # Save the processed frame
        tracked_events[t] = current_t_ids

    # Step 4: Temporal Filtering (Duration check)
    min_timesteps_required = min_duration_hours / hours_per_timestep
    
    for event_id, timesteps in id_lifespan.items():
        if len(timesteps) < min_timesteps_required:
            # If the event didn't live long enough, erase it (set to background 0)
            tracked_events[tracked_events == event_id] = 0

    return tracked_events


def process_unet_batch(model: Module, dataloader_sequence: DataLoader, hours_per_timestep: int=3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Generate predictions and apply tracking.
    Assumes dataloader_sequence yields chronological data.
    """
    model.eval()
    all_predictions = []
    all_tmq = []
    all_timestamps = []
    
    print("Running UNet inference on the dataset...")
    with torch.no_grad():
        index = 1
        for images, labels, timestamp in dataloader_sequence:
            # Output shape: (Batch, 3, H, W)
            logits = model(images)
            
            # Get class predictions using argmax
            # Shape: (Batch, H, W)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.append(preds)

            tmq_batch = images[:, 0, :, :].numpy() 
            all_tmq.append(tmq_batch)

            all_timestamps.extend(timestamp)

            print("Processed batch ", index)
            index += 1

    normalized_tmq = np.concatenate(all_tmq, axis=0)
            
    # Concatenate all batches into a single volume (T, H, W)
    full_sequence = np.concatenate(all_predictions, axis=0)
    
    # ClimateNet typical mapping: 0=Background, 1=TC, 2=AR
    # Create binary masks for each class
    tc_binary = (full_sequence == 1).astype(np.uint8)
    ar_binary = (full_sequence == 2).astype(np.uint8)
    
    # Apply tracking independently
    print("Tracking Tropical Cyclones...")
    tc_tracked = track_climate_events(tc_binary, min_area=250, min_duration_hours=12, hours_per_timestep=hours_per_timestep)
    
    print("Tracking Atmospheric Rivers...")
    ar_tracked = track_climate_events(ar_binary, min_area=250, min_duration_hours=12, hours_per_timestep=hours_per_timestep)
    
    return tc_tracked, ar_tracked, normalized_tmq, all_timestamps


def animate_tracked_events(background_variable: np.ndarray, tracked_events: np.ndarray, timestamps: list, save_path: str="tracking_animation.gif"):
    """
    Creates an animation of the tracked events over the raw meteorological data.
    
    Args:
        background_variable: numpy array (T, H, W) of the background variable (e.g., TMQ).
        tracked_events: numpy array (T, H, W) of the integer event IDs.
        timestamps: list of timestamps corresponding to each frame.
        save_path: filename to save the animation (.mp4 or .gif).
    """
    T, H, W = background_variable.shape
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # Plot the background raw variable
    tracking_title = ("AR" if '_ar_' in save_path else "TC") + " Tracking : "

    color_palette = [
        'crimson', 'darkviolet', 'limegreen', 'gold', 'deeppink', 'darkorange', 'darkgoldenrod',  'yellow', 
        'magenta',   'orangered', 'darkgreen', 'tomato',
        'yellowgreen', 'saddlebrown', 'springgreen'
    ]
    
    # Find every ID that will ever appear in the video
    all_known_ids = np.unique(tracked_events)
    id_to_color = {}

    global_min = np.min(background_variable)
    global_max = np.max(background_variable)
    
    color_index = 0
    for uid in all_known_ids:
        if uid == 0:
            continue # Skip background
        # Permanently assign this specific ID a specific color
        id_to_color[int(uid)] = color_palette[color_index % len(color_palette)]
        color_index += 1

    def update(frame):
        ax.clear() 
        
        # 1. Redraw the background and styling from scratch
        ax.imshow(background_variable[frame], cmap='Blues', alpha=0.8, vmin=global_min, vmax=global_max)
        ax.set_title(f"{tracking_title}{timestamps[frame]}", fontsize=14, fontweight='bold')
        ax.axis('off')
        # Zoom in Golf of Mexico and Atlantic
        ax.set_xlim(530, 850)
        ax.set_ylim(400, 150)
        
        current_frame_events = tracked_events[frame]
        unique_ids = np.unique(current_frame_events)
        
        legend_handles = []

        # 2. Redraw the contours from scratch
        for event_id in unique_ids:
            if event_id == 0:
                continue 
                
            color = id_to_color[int(event_id)]
            event_mask = (current_frame_events == event_id).astype(int)
            
            # Because we wiped the canvas, we don't need to save csets anymore!
            ax.contour(event_mask, levels=[0.5], colors=[color], linewidths=2)

            legend_line = Line2D([0], [0], color=color, lw=3, label=f'ID: {int(event_id)}')
            legend_handles.append(legend_line)

        # 3. Redraw the legend from scratch
        if legend_handles:
            ax.legend(handles=legend_handles, loc='lower right', 
                      title="Active Systems", facecolor='white', framealpha=0.9,
                      fontsize=10, title_fontsize=12)

    print("Generating animation... this might take a minute.")
    
    ani = animation.FuncAnimation(fig, update, frames=T, blit=False)
    # Save the animation
    ani.save(save_path, fps=5)
    print(f"Animation saved to {save_path}")
    # Show it if you are in a Jupyter Notebook, otherwise just close it to save memory
    # plt.show() 
    plt.close()


def get_ram_tmq(dataloader: DataLoader) -> np.ndarray:
    """Extracts the raw TMQ sequence from the dataloader."""

    print("Extracting raw TMQ sequence from the dataloader...")
    dataloader.dataset.normalize = False

    all_tmq = []
    index = 0
    for images, _, _ in dataloader:
        tmq_batch = images[:, 0, :, :].numpy() 
        all_tmq.append(tmq_batch)
        if index % 10 == 0:
            print(f"Processed {index} batches...")
        index += 1

    raw_tmq_sequence = np.concatenate(all_tmq, axis=0)

    return raw_tmq_sequence


TRACKING_PATH ="post_processing/tracking/"

def save_tracked_events(tcs: np.ndarray, ars: np.ndarray, file_name: str):
    """Saves the tracked events to a .npz file to avoid running inference again."""

    np.savez(TRACKING_PATH + file_name, tc_tracked=tcs, ar_tracked=ars)
    print(f"Tracked events saved to {file_name}")


def load_tracked_events(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the tracked events from a .npz file."""

    data = np.load(TRACKING_PATH + file_name)
    tc_tracked = data['tc_tracked']
    ar_tracked = data['ar_tracked']
    print(f"Tracked events loaded from {file_name}")
    return tc_tracked, ar_tracked


def track_climatenet_eval(model: Module):
    eval_data_path = "../shared_CN_B/climatenet_engineered/test"
    features = FeatureSets()
    feature_set = features.get_features(features.CGNet)
    dataset = ClimateNetDataset(eval_data_path, feature_set)
    dataloader = DataLoader(dataset, num_workers=2)

    tc_tracked, ar_tracked, _, timestamps = process_unet_batch(model, dataloader)
    raw_tmq_sequence = get_ram_tmq(dataloader)

    save_tracked_events(tc_tracked, ar_tracked, "climatenet_eval_tracked_events.npz")

    animate_tracked_events(raw_tmq_sequence, tc_tracked, timestamps, 'climatenet_eval_tc_tracking.gif')
    animate_tracked_events(raw_tmq_sequence, ar_tracked, timestamps, 'climatenet_eval_ar_tracking.gif')


def track_era5_timeframe(model: Module):
    # Hurricane Season 2022
    # START_DATE = '2022-09-26'
    # END_DATE = '2022-09-29'

    # Hurricane Erin
    START_DATE = '2025-08-19'
    END_DATE = '2025-08-22'

    date_range = START_DATE + END_DATE[-3:]

    stream_dataset = ARCOStreamDataset(start_date=START_DATE, end_date=END_DATE)
    stream_loader = DataLoader(stream_dataset, batch_size=4, shuffle=False, num_workers=2)

    tc_tracked, ar_tracked, _, timestamps = process_unet_batch(model, stream_loader, hours_per_timestep=1)
    save_tracked_events(tc_tracked, ar_tracked, f"era5_{date_range}_tracked_events.npz")

    raw_tmq = get_ram_tmq(stream_loader)
    animate_tracked_events(raw_tmq, tc_tracked, timestamps, save_path=f"{date_range}_tc_tracking.gif")
    animate_tracked_events(raw_tmq, ar_tracked, timestamps, save_path=f"{date_range}_ar_tracking.gif")


def get_unet(n_features: int, file_path: str) -> Module:
    loaded_model = AttentionUNet(in_channels=n_features, out_channels=3)
    loaded_model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

    return loaded_model


def save_gif_as_png(frame_index: int, gif_path: str, png_path: str):
    try:
        # Open the GIF
        with Image.open(gif_path) as img:
            
            # Move the internal pointer to the specific frame
            # Note: Frames are 0-indexed, so frame 0 is the first image
            img.seek(frame_index)
            
            # Save that specific frame as a PNG
            img.save(png_path, format="PNG")
            
            print(f"Success! Frame {frame_index} saved to {png_path}")
            
    except EOFError:
        print(f"Error: Frame index {frame_index} does not exist. The GIF has fewer frames.")
    except FileNotFoundError:
        print(f"Error: Could not find the file at {gif_path}")


def save_animations_as_pngs():
    save_gif_as_png(30, "post_processing/animations/climatenet_eval_ar_tracking.gif", "post_processing/pngs/cnet_ar.png")
    save_gif_as_png(30, "post_processing/animations/climatenet_eval_tc_tracking.gif", "post_processing/pngs/cnet_tc.png")
    save_gif_as_png(48, "post_processing/animations/september_2022_ar_tracking.gif", "post_processing/pngs/era5_ar.png")
    save_gif_as_png(48, "post_processing/animations/september_2022_tc_tracking.gif", "post_processing/pngs/era5_tc.png")
    save_gif_as_png(48, "post_processing/animations/hurricane_erin_tc_tracking.gif", "post_processing/pngs/erin_tc.png")


def main():
    model = get_unet(4, "models/unet_models/att_unet_25epo_feat_cgnet.pth") 

    # track_climatenet_eval(model)
    track_era5_timeframe(model)


if __name__ == "__main__":
    main()