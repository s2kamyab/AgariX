import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== USER SETTINGS =====
CSV_PATH = "clicked_circle_colors.csv"   # your CSV file
R_COL, G_COL, B_COL = "mean_R","mean_G","mean_B"   # names of RGB columns in the CSV
N_COLS = 48  # <-- set number of columns in the grid (try 31 or 48 etc.)
DOT_SIZE = 150  # marker size for circles
# =========================
#############################################################################################
# 1) Load RGB data Ecoli_mg all antibiotics
df = pd.read_csv(CSV_PATH)
df_partial = df[:384]
# df_partial = df[384:665]
# Extract RGB as numpy array (shape: [N, 3])
colors = df_partial[[R_COL, G_COL, B_COL]].values.astype(float)

# Normalize if values are 0–255
if colors.max() > 1.0:
    colors = colors / 255.0

n_samples = colors.shape[0]

# 2) Compute grid size
assert n_samples % N_COLS == 0, "N_COLS must divide number of samples"
n_rows = n_samples // N_COLS

# Row-wise mapping: 1st row in CSV -> top row of grid
# x increases left→right, y increases bottom→top (we'll flip later)
x = np.tile(np.arange(N_COLS), n_rows)
y = np.repeat(np.arange(n_rows), N_COLS)

# Flip y so first row of CSV appears at the TOP
# y = (n_rows - 1) - y

# 3) Plot as colored circles
plt.figure(figsize=(N_COLS/3, n_rows/3))  # adjust size as you like
plt.scatter(x, y,
            c=colors,
            s=DOT_SIZE,
            marker='o',
            edgecolors='black',
            linewidths=0.5)
mean_color_over_dye_r = np.zeros((8, 4))
mean_color_over_dye_g = np.zeros((8, 4))
mean_color_over_dye_b = np.zeros((8, 4))
doz_list = [0, 1, 2, 4, 8, 16, 32, 64]
bact_list = ['Ecoli_mg', 'E_asb', "E_coli_ef", "s_typh"]
antibiotic_list = ['Ampicillin', 'Chloramphenicol', 'Kanamycin', 'Tetracycline']
dataset_swell = []
for i in range(4 ):
    for k  in range(8):
        t_r = np.mean(colors[i*12:12*(i+1)][:,0])
        t_g = np.mean(colors[i*12:12*(i+1)][:,1])
        t_b = np.mean(colors[i*12:12*(i+1)][:,2])
        # mean_color_over_dye_r[k, i] = np.mean([t_r], axis=1)#.append(np.mean(colors[i*12:12*(i+1)], axis=0))
        # mean_color_over_dye_g[k, i] = np.mean([t_g], axis=1)
        # mean_color_over_dye_b[k, i] = np.mean([t_b], axis=1)
        dataset_swell.append({'mean_R': t_r, 'mean_G': t_g,
                               'mean_B': t_b, 'x': i, 'y': k, \
                               'dye': "Ecoli_mg", 'doz': doz_list[k] , 'antibiotic': antibiotic_list[i]})

dataset_swell_df = pd.DataFrame(dataset_swell)
dataset_swell_df.to_csv('mean_color_over_dye_Ecoli_mg.csv', index=False)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()   # just in case; can comment out if orientation flips
plt.axis('off')
plt.tight_layout()
plt.show()
a=2
#############################################################################################
# 1) Load RGB data Ecoli_asb Ampicillin
N_COLS = 12 
df = pd.read_csv(CSV_PATH)
df_partial = df[384: 384 + 96]
# df_partial = df[384:665]
# Extract RGB as numpy array (shape: [N, 3])
colors = df_partial[[R_COL, G_COL, B_COL]].values.astype(float)

# Normalize if values are 0–255
if colors.max() > 1.0:
    colors = colors / 255.0

n_samples = colors.shape[0]

# 2) Compute grid size
assert n_samples % N_COLS == 0, "N_COLS must divide number of samples"
n_rows = n_samples // N_COLS

# Row-wise mapping: 1st row in CSV -> top row of grid
# x increases left→right, y increases bottom→top (we'll flip later)
x = np.tile(np.arange(N_COLS), n_rows)
y = np.repeat(np.arange(n_rows), N_COLS)

# Flip y so first row of CSV appears at the TOP
# y = (n_rows - 1) - y

# 3) Plot as colored circles
plt.figure(figsize=(N_COLS/3, n_rows/3))  # adjust size as you like
plt.scatter(x, y,
            c=colors,
            s=DOT_SIZE,
            marker='o',
            edgecolors='black',
            linewidths=0.5)
mean_color_over_dye = np.zeros((8, 4))
doz_list = [0, 1, 2, 4, 8, 16, 32, 64]
bact_list = ['Ecoli_mg', 'E_asb', "E_coli_ef", "s_typh"]
antibiotic_list = ['Ampicillin', 'Chloramphenicol', 'Kanamycin', 'Tetracycline']
dataset_swell = []
for i in range(1 ):
    for k  in range(8):
        t_r = np.mean(colors[i*12:12*(i+1)][:,0])
        t_g = np.mean(colors[i*12:12*(i+1)][:,1])
        t_b = np.mean(colors[i*12:12*(i+1)][:,2])
        # mean_color_over_dye_r[k, i] = np.mean([t_r], axis=1)#.append(np.mean(colors[i*12:12*(i+1)], axis=0))
        # mean_color_over_dye_g[k, i] = np.mean([t_g], axis=1)
        # mean_color_over_dye_b[k, i] = np.mean([t_b], axis=1)
        dataset_swell.append({'mean_R': t_r, 'mean_G': t_g,
                               'mean_B': t_b, 
                                  'x': i, 'y': k, \
                               'dye': "Ecoli_asb", 'doz': doz_list[k] , 'antibiotic': antibiotic_list[0]})

dataset_swell_df = pd.DataFrame(dataset_swell)
dataset_swell_df.to_csv('mean_color_over_dye_Ecoli_asb_ampicillin.csv', index=False)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()   # just in case; can comment out if orientation flips
plt.axis('off')
plt.tight_layout()
plt.show()
a=2
#############################################################################################
# 1) Load RGB data Ecoli_asb Chloramphenicol
N_COLS = 11 # my mistake! in the selection phase I took 11 columns instead of 12 
df = pd.read_csv(CSV_PATH)
df_partial = df[384+96: 384 + 96 + 88]
# df_partial = df[384:665]
# Extract RGB as numpy array (shape: [N, 3])
colors = df_partial[[R_COL, G_COL, B_COL]].values.astype(float)

# Normalize if values are 0–255
if colors.max() > 1.0:
    colors = colors / 255.0

n_samples = colors.shape[0]

# 2) Compute grid size
assert n_samples % N_COLS == 0, "N_COLS must divide number of samples"
n_rows = n_samples // N_COLS

# Row-wise mapping: 1st row in CSV -> top row of grid
# x increases left→right, y increases bottom→top (we'll flip later)
x = np.tile(np.arange(N_COLS), n_rows)
y = np.repeat(np.arange(n_rows), N_COLS)

# Flip y so first row of CSV appears at the TOP
# y = (n_rows - 1) - y

# 3) Plot as colored circles
plt.figure(figsize=(N_COLS/3, n_rows/3))  # adjust size as you like
plt.scatter(x, y,
            c=colors,
            s=DOT_SIZE,
            marker='o',
            edgecolors='black',
            linewidths=0.5)
mean_color_over_dye = np.zeros((8, 4))
doz_list = [0, 1, 2, 4, 8, 16, 32, 64]
bact_list = ['Ecoli_mg', 'E_asb', "E_coli_ef", "s_typh"]
antibiotic_list = ['Ampicillin', 'Chloramphenicol', 'Kanamycin', 'Tetracycline']
dataset_swell = []
for i in range(1 ):
    for k  in range(8):
        t_r = np.mean(colors[i*12:12*(i+1)][:,0])
        t_g = np.mean(colors[i*12:12*(i+1)][:,1])
        t_b = np.mean(colors[i*12:12*(i+1)][:,2])
        # mean_color_over_dye_r[k, i] = np.mean([t_r], axis=1)#.append(np.mean(colors[i*12:12*(i+1)], axis=0))
        # mean_color_over_dye_g[k, i] = np.mean([t_g], axis=1)
        # mean_color_over_dye_b[k, i] = np.mean([t_b], axis=1)
        dataset_swell.append({'mean_R': t_r, 'mean_G': t_g,
                               'mean_B': t_b, 'x': i, 'y': k, \
                               'dye': "Ecoli_asb", 'doz': doz_list[k] , 'antibiotic': antibiotic_list[1]})

dataset_swell_df = pd.DataFrame(dataset_swell)
dataset_swell_df.to_csv('mean_color_over_dye_Ecoli_asb_chloramphenicol.csv', index=False)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()   # just in case; can comment out if orientation flips
plt.axis('off')
plt.tight_layout()
plt.show()
a=2
#############################################################################################
# 1) Load RGB data Ecoli_asb 'Kanamycin', 'Tetracycline'
N_COLS = 24
df = pd.read_csv(CSV_PATH)
df_partial = df[384 + 96 + 88: 384 + 96 + 88 + 192]
# df_partial = df[384:665]
# Extract RGB as numpy array (shape: [N, 3])
colors = df_partial[[R_COL, G_COL, B_COL]].values.astype(float)

# Normalize if values are 0–255
if colors.max() > 1.0:
    colors = colors / 255.0

n_samples = colors.shape[0]

# 2) Compute grid size
assert n_samples % N_COLS == 0, "N_COLS must divide number of samples"
n_rows = n_samples // N_COLS

# Row-wise mapping: 1st row in CSV -> top row of grid
# x increases left→right, y increases bottom→top (we'll flip later)
x = np.tile(np.arange(N_COLS), n_rows)
y = np.repeat(np.arange(n_rows), N_COLS)

# Flip y so first row of CSV appears at the TOP
# y = (n_rows - 1) - y

# 3) Plot as colored circles
plt.figure(figsize=(N_COLS/3, n_rows/3))  # adjust size as you like
plt.scatter(x, y,
            c=colors,
            s=DOT_SIZE,
            marker='o',
            edgecolors='black',
            linewidths=0.5)
mean_color_over_dye = np.zeros((8, 4))
doz_list = [0, 1, 2, 4, 8, 16, 32, 64]
bact_list = ['Ecoli_mg', 'E_asb', "E_coli_ef", "s_typh"]
antibiotic_list = ['Ampicillin', 'Chloramphenicol', 'Kanamycin', 'Tetracycline']
dataset_swell = []
for i in range(2 ):
    for k  in range(8):
        t_r = np.mean(colors[i*12:12*(i+1)][:,0])
        t_g = np.mean(colors[i*12:12*(i+1)][:,1])
        t_b = np.mean(colors[i*12:12*(i+1)][:,2])
        # mean_color_over_dye_r[k, i] = np.mean([t_r], axis=1)#.append(np.mean(colors[i*12:12*(i+1)], axis=0))
        # mean_color_over_dye_g[k, i] = np.mean([t_g], axis=1)
        # mean_color_over_dye_b[k, i] = np.mean([t_b], axis=1)
        dataset_swell.append({'mean_R': t_r, 'mean_G': t_g,
                               'mean_B': t_b, 'x': i, 'y': k, \
                               'dye': "Ecoli_asb", 'doz': doz_list[k] , 'antibiotic': antibiotic_list[i+2]})

dataset_swell_df = pd.DataFrame(dataset_swell)
dataset_swell_df.to_csv('mean_color_over_dye_Ecoli_asb_Kanamycin_tetracycline.csv', index=False)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()   # just in case; can comment out if orientation flips
plt.axis('off')
plt.tight_layout()
plt.show()
a=2
###########################################################################################
# 1) Load RGB data Ecoli_EF all antibiotics
N_COLS = 48
df = pd.read_csv(CSV_PATH)
df_partial = df[384 + 96 + 88 + 192: 384 + 96 + 88 + 192 + 384]
# df_partial = df[384:665]
# Extract RGB as numpy array (shape: [N, 3])
colors = df_partial[[R_COL, G_COL, B_COL]].values.astype(float)

# Normalize if values are 0–255
if colors.max() > 1.0:
    colors = colors / 255.0

n_samples = colors.shape[0]

# 2) Compute grid size
assert n_samples % N_COLS == 0, "N_COLS must divide number of samples"
n_rows = n_samples // N_COLS

# Row-wise mapping: 1st row in CSV -> top row of grid
# x increases left→right, y increases bottom→top (we'll flip later)
x = np.tile(np.arange(N_COLS), n_rows)
y = np.repeat(np.arange(n_rows), N_COLS)

# Flip y so first row of CSV appears at the TOP
# y = (n_rows - 1) - y

# 3) Plot as colored circles
plt.figure(figsize=(N_COLS/3, n_rows/3))  # adjust size as you like
plt.scatter(x, y,
            c=colors,
            s=DOT_SIZE,
            marker='o',
            edgecolors='black',
            linewidths=0.5)
mean_color_over_dye = np.zeros((8, 4))
doz_list = [0, 1, 2, 4, 8, 16, 32, 64]
bact_list = ['Ecoli_mg', 'E_asb', "E_coli_ef", "s_typh"]
antibiotic_list = ['Ampicillin', 'Chloramphenicol', 'Kanamycin', 'Tetracycline']
dataset_swell = []
for i in range(4 ):
    for k  in range(8):
        t_r = np.mean(colors[i*12:12*(i+1)][:,0])
        t_g = np.mean(colors[i*12:12*(i+1)][:,1])
        t_b = np.mean(colors[i*12:12*(i+1)][:,2])
        # mean_color_over_dye_r[k, i] = np.mean([t_r], axis=1)#.append(np.mean(colors[i*12:12*(i+1)], axis=0))
        # mean_color_over_dye_g[k, i] = np.mean([t_g], axis=1)
        # mean_color_over_dye_b[k, i] = np.mean([t_b], axis=1)
        dataset_swell.append({'mean_R': t_r, 'mean_G': t_g,
                               'mean_B': t_b, 'x': i, 'y': k, \
                               'dye': "Ecoli_ef", 'doz': doz_list[k] , 'antibiotic': antibiotic_list[i]})

dataset_swell_df = pd.DataFrame(dataset_swell)
dataset_swell_df.to_csv('mean_color_over_dye_Ecoli_ef.csv', index=False)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()   # just in case; can comment out if orientation flips
plt.axis('off')
plt.tight_layout()
plt.show()
a=2
#############################################################################################
# 1) Load RGB data s_typh  all antibiotics ### only 7 dozes available
df = pd.read_csv(CSV_PATH)
N_COLS = 48
df_partial = df[384 + 96 + 88 + 192 + 384: 384 + 96 + 88 + 192 + 384 + 336]
# Extract RGB as numpy array (shape: [N, 3])
colors = df_partial[[R_COL, G_COL, B_COL]].values.astype(float)

# Normalize if values are 0–255
if colors.max() > 1.0:
    colors = colors / 255.0

n_samples = colors.shape[0]

# 2) Compute grid size
assert n_samples % N_COLS == 0, "N_COLS must divide number of samples"
n_rows = n_samples // N_COLS

# Row-wise mapping: 1st row in CSV -> top row of grid
# x increases left→right, y increases bottom→top (we'll flip later)
x = np.tile(np.arange(N_COLS), n_rows)
y = np.repeat(np.arange(n_rows), N_COLS)

# Flip y so first row of CSV appears at the TOP
# y = (n_rows - 1) - y

# 3) Plot as colored circles
plt.figure(figsize=(N_COLS/3, n_rows/3))  # adjust size as you like
plt.scatter(x, y,
            c=colors,
            s=DOT_SIZE,
            marker='o',
            edgecolors='black',
            linewidths=0.5)
mean_color_over_dye = np.zeros((7, 4))
doz_list = [0, 1, 2, 4, 8, 16, 32, 64]
bact_list = ['Ecoli_mg', 'E_asb', "E_coli_ef", "s_typh"]
antibiotic_list = ['Ampicillin', 'Chloramphenicol', 'Kanamycin', 'Tetracycline']
dataset_swell = []
for i in range(4 ):
    for k  in range(7):
        t_r = np.mean(colors[i*12:12*(i+1)][:,0])
        t_g = np.mean(colors[i*12:12*(i+1)][:,1])
        t_b = np.mean(colors[i*12:12*(i+1)][:,2])
        # mean_color_over_dye_r[k, i] = np.mean([t_r], axis=1)#.append(np.mean(colors[i*12:12*(i+1)], axis=0))
        # mean_color_over_dye_g[k, i] = np.mean([t_g], axis=1)
        # mean_color_over_dye_b[k, i] = np.mean([t_b], axis=1)
        dataset_swell.append({'mean_R': t_r, 'mean_G': t_g,
                               'mean_B': t_b, 'x': i, 'y': k, \
                               'dye': "s_typh", 'doz': doz_list[k] , 'antibiotic': antibiotic_list[i]})

dataset_swell_df = pd.DataFrame(dataset_swell)
dataset_swell_df.to_csv('mean_color_over_dye_s_typh.csv', index=False)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()   # just in case; can comment out if orientation flips
plt.axis('off')
plt.tight_layout()
plt.show()
a=2
##############################################################################################