# sims.py

import numpy as np

# Function to compute the diffusion update
def apply_diffusion(field, D, delta_t, delta_x):
    # Create a copy of the field to store updates
    updated_field = np.copy(field)
    
    # Apply the diffusion equation using finite differences
    for i in range(1, field.shape[0] - 1):
        for j in range(1, field.shape[1] - 1):
            laplacian = (field[i+1, j] + field[i-1, j] + field[i, j+1] + field[i, j-1] - 4 * field[i, j]) / (delta_x**2)
            updated_field[i, j] = field[i, j] + D * delta_t * laplacian
            
    return updated_field
