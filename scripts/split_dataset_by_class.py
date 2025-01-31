import os
import shutil
import random
from scipy.io import loadmat

# Paths
source_dir = "dataset/jpg"  # Folder containing the images
output_dir = "dataset"  # Destination folder for splits
labels_path = "dataset/imagelabels.mat"  # Path to imagelabels.mat

# Create base output directories
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "validation")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load imagelabels.mat
labels = loadmat(labels_path)['labels'].flatten()  # Class labels (1-indexed)

class_names = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", 
    "wild geranium", "tiger lily", "moon orchid", "bird of paradise", "monkshood", 
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle", 
    "yellow iris", "globe flower", "purple coneflower", "peruvian lily", 
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", 
    "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", 
    "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", 
    "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", 
    "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", 
    "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower", 
    "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", 
    "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", 
    "gaura", "geranium", "orange dahlia", "pink-yellow dahlia?", "cautleya spicata", 
    "japanese anemone", "black-eyed susan", "silverbush", "californian poppy", 
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", 
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", 
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani", 
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", 
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", 
    "bee balm", "pink quill", "foxglove", "bougainvillea", "camellia", 
    "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", 
    "blackberry lily"
]

# Get all image indices
all_indices = list(range(1, len(labels) + 1))  # MATLAB indices are 1-based

# Shuffle and split indices into 80/10/10 for train/validation/test
random.seed(42)  # Ensure reproducibility
random.shuffle(all_indices)

train_split = int(0.8 * len(all_indices))
val_split = int(0.9 * len(all_indices))

train_indices = all_indices[:train_split]
val_indices = all_indices[train_split:val_split]
test_indices = all_indices[val_split:]

# Helper function to copy images to their respective class folders
def copy_images_by_class(indices, destination):
    for idx in indices:
        class_label = labels[idx - 1]  # Class label for the image (1-indexed in MATLAB)
        #class_dir = os.path.join(destination, f"class_{class_label:03d}")
        class_dir = os.path.join(destination, class_names[class_label - 1])

        os.makedirs(class_dir, exist_ok=True)  # Create class directory if it doesn't exist

        filename = f"image_{idx:05d}.jpg"  # Format index as image_XXXX.jpg
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(class_dir, filename)
        if os.path.exists(src_path):  # Check if file exists
            shutil.copy(src_path, dst_path)
        else:
            print(f"File not found: {src_path}")

# Organize images into train, validation, and test folders by class
copy_images_by_class(train_indices, train_dir)
copy_images_by_class(val_indices, val_dir)
copy_images_by_class(test_indices, test_dir)

print("Dataset split and organized by class completed!")
