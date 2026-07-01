# """Fill this in. That's the whole interface.

# Usage:
#     python predict.py some_image.jpg
# Prints ONE number from 0 to 1:
#     0 = real photo,  1 = photo of a screen (recapture / fraud)
# A hard 0 or 1 is fine if your method gives a yes/no answer.
# """

# import sys
# from PIL import Image


# def predict(image_path: str) -> float:
#     img = Image.open(image_path).convert("RGB")
#     # TODO: run your detector and return how likely the image is a photo-of-a-screen.
#     # It can be a trained model, a classic CV / image-processing method, frequency
#     # analysis, or any algorithm you like -- your choice.
#     raise NotImplementedError("return a fraud score in [0, 1]")


# if __name__ == "__main__":
#     print(predict(sys.argv[1]))








import sys
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')



class MobileNetDetector:
    def __init__(self, model_path='mobilenet_model.pth'):
        """Initialize the MobileNet detector"""
        self.device = torch.device('cpu')
        
        # Load model architecture
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Load trained weights
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f" Loaded model from {model_path}", file=sys.stderr)
            except Exception as e:
                print(f" Error loading model: {e}", file=sys.stderr)
                print("Using fallback detection", file=sys.stderr)
                self.model = None
        else:
            print(f" Model file '{model_path}' not found", file=sys.stderr)
            print("Using fallback detection", file=sys.stderr)
            self.model = None
        
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Predict if image is a screen recapture"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(image_tensor)
                probability = torch.sigmoid(output).item()
            
            return probability
            
        except Exception as e:
            print(f"Error during prediction: {e}", file=sys.stderr)
            return 0.5

# FALLBACK: FEATURE-BASED DETECTION (if model not available)


def detect_screen_features(image_path):
    """Fallback feature-based detection when model is not available"""
    try:
        import cv2
    except ImportError:
        return 0.5
    
    img = cv2.imread(image_path)
    if img is None:
        return 0.5
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    
    # Frequency analysis
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    magnitude_log = np.log(magnitude + 1)
    
    h, w = 256, 256
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    low_mask = dist <= 25
    low_energy = np.mean(magnitude_log[low_mask]) if np.any(low_mask) else 0
    
    high_mask = dist > 60
    high_energy = np.mean(magnitude_log[high_mask]) if np.any(high_mask) else 0
    
    freq_ratio = high_energy / (low_energy + 1e-6)
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    
    # Color analysis
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_std = np.std(hsv[:, :, 0])
        color_uniformity = 1.0 / (1.0 + hue_std / 10)
    else:
        color_uniformity = 0.5
    
    # Combine features
    score = (
        0.35 * min(freq_ratio / 3.0, 1.0) +
        0.30 * min(edge_density * 5, 1.0) +
        0.35 * min(color_uniformity, 1.0)
    )
    
    return np.clip(score, 0, 1)
# MAIN PREDICTION FUNCTION

# Global detector instance
_detector = None

def predict(image_path):
    """
    Predict if image is a screen recapture.
    Returns float between 0 and 1 (1 = screen, 0 = real)
    """
    global _detector
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found", file=sys.stderr)
        return 0.5
    
    # Check file extension
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    if not image_path.lower().endswith(valid_extensions):
        print(f"Warning: Unsupported file format", file=sys.stderr)
        return 0.5
    
    # Initialize detector if not already done
    if _detector is None:
        _detector = MobileNetDetector()
    
    # Use MobileNet if available
    if _detector.model is not None:
        try:
            score = _detector.predict(image_path)
            return score
        except Exception as e:
            print(f"MobileNet error: {e}, using fallback", file=sys.stderr)
    
    # Fallback to feature-based detection
    return detect_screen_features(image_path)


def evaluate_on_dataset(dataset_path='dataset'):
    """Evaluate the model on your dataset"""
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please create dataset/real/ and dataset/screen/ folders", file=sys.stderr)
        return None
    
    results = {'real': [], 'screen': []}
    file_names = {'real': [], 'screen': []}
    
    # Test real images
    real_dir = os.path.join(dataset_path, 'real')
    if os.path.exists(real_dir):
        for fname in os.listdir(real_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(real_dir, fname)
                score = predict(path)
                results['real'].append(score)
                file_names['real'].append(fname)
    
    # Test screen images
    screen_dir = os.path.join(dataset_path, 'screen')
    if os.path.exists(screen_dir):
        for fname in os.listdir(screen_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(screen_dir, fname)
                score = predict(path)
                results['screen'].append(score)
                file_names['screen'].append(fname)
    
    # Calculate statistics
    if results['real'] and results['screen']:
        real_correct = sum(1 for s in results['real'] if s < 0.5)
        screen_correct = sum(1 for s in results['screen'] if s >= 0.5)
        
        real_total = len(results['real'])
        screen_total = len(results['screen'])
        total = real_total + screen_total
        correct = real_correct + screen_correct
        accuracy = correct / total if total > 0 else 0
        
        real_avg = sum(results['real']) / real_total if real_total > 0 else 0
        screen_avg = sum(results['screen']) / screen_total if screen_total > 0 else 0
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\n REAL IMAGES: {real_total}")
        print(f"   Correctly identified: {real_correct}/{real_total} ({real_correct/real_total*100:.1f}%)")
        print(f"   Average score: {real_avg:.4f} (should be < 0.5)")
        
        print(f"\n SCREEN IMAGES: {screen_total}")
        print(f"   Correctly identified: {screen_correct}/{screen_total} ({screen_correct/screen_total*100:.1f}%)")
        print(f"   Average score: {screen_avg:.4f} (should be > 0.5)")
        
        print(f"\n{'='*60}")
        print(f" OVERALL ACCURACY: {accuracy*100:.2f}%")
        print(f"{'='*60}")
        
        # Show sample predictions
        print("\n Sample Predictions:")
        print("  REAL images (first 3):")
        for i in range(min(3, len(results['real']))):
            status = "yes" if results['real'][i] < 0.5 else "no"
            print(f"    {file_names['real'][i][:30]:30s} {results['real'][i]:.4f} {status}")
        
        print("  SCREEN images (first 3):")
        for i in range(min(3, len(results['screen']))):
            status = "yes" if results['screen'][i] >= 0.5 else "no"
            print(f"    {file_names['screen'][i][:30]:30s} {results['screen'][i]:.4f} {status}")
        
        return accuracy
    else:
        print("Dataset not found or empty", file=sys.stderr)
        return None

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py image.jpg")
        print("   or: python predict.py --evaluate")
        print("   or: python predict.py --help")
        print("\nReturns: score (0-1) where 1 = photo of screen, 0 = real photo")
        sys.exit(1)
   
    if sys.argv[1] == "--evaluate":
        evaluate_on_dataset()
        sys.exit(0)
    

    if sys.argv[1] in ["-h", "--help", "help"]:
        print("="*60)
        print("SCREEN RECAPTURE DETECTION - MobileNet")
        print("="*60)
        print("\nUsage:")
        print("  python predict.py <image_path>")
        print("  python predict.py --evaluate")
        print("  python predict.py --help")
        print("\nOutput:")
        print("  A float between 0 and 1")
        print("  < 0.5: Image is likely a REAL photo")
        print("  > 0.5: Image is likely a PHOTO OF A SCREEN")
        print("\nSupported Formats:")
        print("  JPEG, PNG (including transparency), BMP, TIFF")
        print("\nExamples:")
        print("  python predict.py photo.jpg")
        print("  python predict.py screenshot.png")
        print("  python predict.py --evaluate  # Test on your dataset")
        print("="*60)
        sys.exit(0)
    
   
    image_path = sys.argv[1]
    
    # Measure latency
    start_time = time.time()
    result = predict(image_path)
    latency = (time.time() - start_time) * 1000
    
    print(f"{result:.4f}")
    

    print(f"Latency: {latency:.1f}ms", file=sys.stderr)
    print(f"Image: {image_path}", file=sys.stderr)
    print(f"Prediction: {'SCREEN' if result >= 0.5 else 'REAL'}", file=sys.stderr)