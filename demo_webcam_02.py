import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    

    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    if args.pred_only: 
        output_width = frame_width
    else: 
        output_width = frame_width * 2 + margin_width
    

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break
        
        raw_frame = cv2.flip(raw_frame,1)
        
        depth = depth_anything.infer_image(raw_frame, args.input_size)
        
        min_val = depth.min()
        max_val = depth.max()

        min_loc = np.unravel_index(np.argmin(depth, axis=None), depth.shape)
        max_loc = np.unravel_index(np.argmax(depth, axis=None), depth.shape)

        depth = (depth - min_val) / (max_val - min_val) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        
        cv2.putText(depth, f'Closest: {min_val:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(depth, f'Farthest: {max_val:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(depth, (min_loc[1], min_loc[0]), 5, (0, 255, 0), -1)  # Closest point
        cv2.circle(depth, (max_loc[1], max_loc[0]), 5, (255, 0, 255), -1)  # Farthest point

        cv2.putText(depth, f'Closest', (max_loc[1], max_loc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
        
        if args.pred_only:
            cv2.imshow('depth only ', depth)
        else:
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth])
            cv2.imshow('depth together ', combined_frame)
            
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
