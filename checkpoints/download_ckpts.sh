#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Define the URLs for the checkpoints
BASE_URL="https://huggingface.co/depth-anything"
dam2_s_url="${BASE_URL}/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
dam2_b_plus_url="${BASE_URL}/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
dam2_l_url="${BASE_URL}/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"


# Download each of the four checkpoints using wget

echo "Downloading sam2_hiera_small.pt checkpoint..."
wget $dam2_s_url || { echo "Failed to download checkpoint from $dam2_s_url"; exit 1; }

echo "Downloading sam2_hiera_base_plus.pt checkpoint..."
wget $dam2_b_plus_url || { echo "Failed to download checkpoint from $dam2_b_plus_url"; exit 1; }

echo "Downloading sam2_hiera_large.pt checkpoint..."
wget $dam2_l_url || { echo "Failed to download checkpoint from $dam2_l_url"; exit 1; }

echo "All checkpoints are downloaded successfully."
