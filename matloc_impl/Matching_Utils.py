import cv2
import numpy as np
import torch

def lift_to_3d(depth_image, feature_image, transform_matrix, im_size=(256, 256), fov = np.pi/2, device='cuda'):
    """
    Lifts a feature image to 3D using the depth image.
    Args:
        depth_image: A 2D depth image.
        feature_image: A 2D feature image.
    Returns:
        A 3D point cloud.
    """
        
    image_width = im_size[0]
    image_height = im_size[1]
    pp_w = image_width / 2.0
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov / 2.0) # since we are supplying the horizontal fov, this might need to change to pp_w
    intrinsics_matrix = torch.tensor([[focal_length, 0, pp_w], [0, focal_length, pp_h], [0, 0, 1]], dtype=torch.float32)
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    
    intrinsics_matrix = intrinsics_matrix.to(device)
    
    
    H, W = im_size
    
    #Create a grid of pixel coordinates
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H))
    x = x.to(device)
    y = y.to(device)
    
    #flatten depth image 
    depth = depth_image.flatten()
    
    #create homogenous coordiantes
    ones = torch.ones_like(torch.tensor(x.flatten())).to(device)
    pixels_homogeneous = torch.stack([x.flatten(), y.flatten(), depth], dim=0)
    pixels_homogeneous = torch.tensor(pixels_homogeneous, dtype=torch.float32)
    
    
    intrinsics_matrix_inv = torch.linalg.inv(intrinsics_matrix)
    pixels_camera_frame = intrinsics_matrix_inv @ pixels_homogeneous
    

    transform_matrix_inv = torch.linalg.inv(transform_matrix)
    pixels_world_frame = transform_matrix_inv @ torch.vstack([pixels_camera_frame, ones])
    
    pixels_world_frame /= pixels_world_frame[3, :].clone().detach().requires_grad_(True)
    
    pixels_world_frame = pixels_world_frame[:3, :]

    # Reshape the result back to the image shape
    X_world = pixels_world_frame[0, :].reshape(H, W)
    Y_world = pixels_world_frame[1, :].reshape(H, W)
    Z_world = pixels_world_frame[2, :].reshape(H, W)
    
    # append features to the final output. Final shape is N_points by 3 + C
    _, _, C = feature_image.shape
    features_flat = feature_image.reshape(-1, C)
    points_with_features = torch.concatenate([
        pixels_world_frame.T, 
        features_flat  
    ], dim=1)
    
   

    return points_with_features
    


def test_lift_to_3d():
    depth_image = torch.zeros((256, 256,1), dtype=torch.float32)
    feature_image = torch.zeros((256, 256,64), dtype=torch.float32)
    transform_matrix = torch.tensor(np.eye(4), dtype=torch.float32)
    output = lift_to_3d(depth_image, feature_image, transform_matrix)
    
    print(output.shape)
    print(output[0])
    
if __name__ == '__main__':
    test_lift_to_3d()
    