#!C:\Users\shima\Documents\Maryan\Code\.venv\Scripts\python.exe
# The csa dataset contains measurements from 147 exposures of a 36‑dye sensor 
# array to volatile chemical toxicants under two concentration conditions.
#  Each observation is a 36×3 matrix of red/green/blue values, stored in two
#  lists (PEL and IDLH) with a class label vector y indicating 21 different 
# analytes
# search.r-project.org
# . This dataset (from Zhong & Suslick, 2015) shows how colour matrices can be 
# organised and labelled for classification tasks.
      # class labels (1–21)
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_csa_matrix(matrix, title=None):
    """
    Given a 36×3 matrix (rows=dye spots, columns=R,G,B),
    reshape to 6×6×3 and display as an image.
    """
    # Reshape: 36 rows → 6×6 grid; 3 columns stay as RGB channels.
    img = matrix.reshape((6, 6, 3))
    # Values may be floats in 0–1 or 0–255; clip to [0,1] for display.
    img_norm = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img_norm)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

# Read the labels (assumes the CSV has one column of labels)
labels = pd.read_csv("Datasets/csa_labels.csv").iloc[:, 0].to_numpy()

# Read PEL and IDLH JSON files.
with open("Datasets/csa_PEL.json", "r") as f:
    pel_json = json.load(f)

with open("Datasets/csa_IDLH.json", "r") as f:
    idlh_json = json.load(f)

# Convert each JSON matrix (list of lists) to a NumPy array.
# Each entry in pel_matrices/idlh_matrices is a 36×3 array.
pel_matrices = [np.array(m) for m in pel_json]
idlh_matrices = [np.array(m) for m in idlh_json]

# Example: visualize the first PEL sample
# visualize_csa_matrix(pel_matrices[1], title='PEL sample 2')
visualize_csa_matrix(idlh_matrices[1], title='IDLH sample 2')


#Plot individual dye responses by channel.
import pandas as pd
import matplotlib.pyplot as plt

# Convert a single matrix into a DataFrame for plotting
df = pd.DataFrame(np.squeeze(pel_matrices[1]), columns=['R','G','B'])
df.index = np.arange(1, 37)  # label dyes 1–36

df.plot(kind='bar', figsize=(10,4))
plt.xlabel('Dye spot')
plt.ylabel('Colour change')
plt.title('Colour response for one sample (R, G, B)')
plt.show()


# # If you want to flatten the 36×3 matrices into 108‑element feature vectors:
# pel_vectors  = np.array([m.flatten() for m in pel_matrices])
# idlh_vectors = np.array([m.flatten() for m in idlh_matrices])

# # Example: combine both concentration levels and duplicate labels accordingly.
# X = np.vstack([pel_vectors, idlh_vectors])
# y = np.hstack([labels, labels])  # each label is duplicated for PEL and IDLH

