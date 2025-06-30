# Create a simple UI to visualize the reconstructed video
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, HTML
import numpy as np
from PIL import Image
import torch

def create_video_visualization(video_tensor, title="Reconstructed Video", fps=30, dpi=100, display_width=640, display_height=360):
    """
    Create an interactive video visualization using matplotlib.
    
    Args:
        video_tensor: Video tensor of shape [T, C, H, W]
        title: Title for the plot
        fps: Frames per second for playback
        dpi: Dots per inch for the plot
        display_width: Width of the display in pixels
        display_height: Height of the display in pixels
    Returns:
        tuple: (animation object, video_np array)
    """
    # Convert tensor from [-1, 1] to [0, 255] pixel range
    video_tensor = (video_tensor + 1) / 2  # NOTE for some reason this is slightly off of [-1, 1]
    video_tensor = (video_tensor * 255).to(torch.uint8)  # [T, C, H, W]
    video_tensor = torch.clip(video_tensor, 0, 255)  # bandaid for the [-1,1] use this logic
    
    # Permute to [T, H, W, C]
    video_tensor = video_tensor.permute(0, 2, 3, 1)
    
    # Convert to numpy
    video_np = video_tensor.cpu().numpy()
    
    # Calculate figure size to maintain 360x640 aspect ratio
    fig_width = display_width / dpi
    fig_height = display_height / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    # Create the first frame
    img = ax.imshow(video_np[0])
    
    def animate(frame_idx):
        img.set_array(video_np[frame_idx])
        return [img]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(video_np), 
        interval=1000//fps, blit=True, repeat=True
    )
    
    plt.tight_layout()
    return anim, video_np

def export_video_as_gif(video_np, output_path, fps=30):
    """
    Export video numpy array as a GIF file using PIL directly.
    
    Args:
        video_np: Video numpy array of shape [T, H, W, C]
        output_path: Path where to save the GIF file
        fps: Frames per second for the GIF
    """
    # Convert frames to PIL Images
    frames = []
    for frame in video_np:
        # Ensure frame is in uint8 format
        frame_uint8 = frame.astype(np.uint8)
        pil_image = Image.fromarray(frame_uint8)
        frames.append(pil_image)
    
    # Save as GIF with no border/padding
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//fps,  # duration in milliseconds
        loop=0,  # 0 means loop forever
        optimize=True
    )
    print(f"GIF saved to: {output_path}")