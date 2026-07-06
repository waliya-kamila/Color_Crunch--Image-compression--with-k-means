import numpy as np
from PIL import Image
import random
import sys

def run_k_means():
    """
    Main function to run K-Means image compression.
    """
    print("K-Means Image Compressor")
    print("=" * 30)
    
    while True:
        print("\n1. Compress an image")
        print("2. Open a compressed image")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            compress_image()
        elif choice == '2':
            open_compressed_image()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def compress_image():
    """
    Compress an image using K-Means clustering.
    """
    try:
        # Get image path
        image_path = input("Enter the path to the image: ")
        
        # Load image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Get K value
        k = int(input("Enter the number of colors (K): "))
        
        # Reshape image for clustering
        height, width, channels = image_array.shape
        pixels = image_array.reshape(-1, channels)
        
        # Initialize centroids randomly
        centroids = pixels[random.sample(range(len(pixels)), k)]
        
        # K-Means clustering
        max_iterations = 100
        for iteration in range(max_iterations):
            # Assign pixels to nearest centroid
            distances = np.sqrt(((pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((k, channels))
            for i in range(k):
                cluster_pixels = pixels[labels == i]
                if len(cluster_pixels) > 0:
                    new_centroids[i] = cluster_pixels.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                print(f"Converged after {iteration + 1} iterations")
                break
            
            centroids = new_centroids
        
        # Reconstruct image
        compressed_pixels = centroids[labels]
        compressed_image = compressed_pixels.reshape(height, width, channels).astype(np.uint8)
        
        # Save compressed image
        output_path = image_path.rsplit('.', 1)[0] + '_compressed.' + image_path.rsplit('.', 1)[1]
        compressed_image_pil = Image.fromarray(compressed_image)
        compressed_image_pil.save(output_path)
        
        # Calculate compression statistics
        original_size = len(pixels) * channels
        compressed_size = k * channels + len(pixels)  # centroids + labels
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        print(f"\nCompression completed!")
        print(f"Original colors: {len(np.unique(pixels, axis=0))}")
        print(f"Compressed colors: {k}")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

def open_compressed_image():
    """
    Open and display a compressed image.
    """
    try:
        # Get image path
        image_path = input("Enter the path to the compressed image: ")
        
        # Load and display image
        image = Image.open(image_path)
        image.show()
        
        print(f"Image opened: {image_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_k_means() 
