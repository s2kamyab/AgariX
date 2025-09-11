import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the CSV files
df_isolates = pd.read_csv("Datasets/Antibiotic_resistance/pcbi.1006258.s010.csv")
df_genes = pd.read_csv("Datasets/Antibiotic_resistance/pcbi.1006258.s011.csv")
df_preds = pd.read_csv("Datasets/Antibiotic_resistance/pcbi.1006258.s012.csv")

# Strip whitespaces from column names
df_isolates.columns = df_isolates.columns.str.strip()
df_preds.columns = df_preds.columns.str.strip()
df_genes.columns = df_genes.columns.str.strip()

# Merge predictions with isolate features
df_merged = pd.merge(df_preds, df_isolates, left_on="Isolates", right_on="Isolate", how="left")

# ----------- Visualization ------------
def plot_prediction_vs_observation(data, antibiotic_name):
    data_abx = data[data["Antibiotic"] == antibiotic_name]
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data_abx, x="Observation", hue="Prediction")
    plt.title(f"Observed vs Predicted - {antibiotic_name}")
    plt.xlabel("Observed Resistance")
    plt.ylabel("Count")
    plt.legend(title="Prediction")
    plt.tight_layout()
    plt.show()

# Example plot for CTX (change to AMP, CIP, etc. as needed)
plot_prediction_vs_observation(df_merged, "CTX")

# ----------- Simple Model Example ------------
def model_antibiotic_resistance(df_isolates, antibiotic):
    df_model = df_isolates[["Isolate", "Year", antibiotic]].dropna()
    df_model = df_model[df_model[antibiotic].isin(["S", "I", "R"])]  # filter valid labels
    
    label_encoder = LabelEncoder()
    df_model["Label"] = label_encoder.fit_transform(df_model[antibiotic])

    X = df_model[["Year"]]  # replace with richer features as needed
    y = df_model["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix for {antibiotic}")
    plt.show()

# Example model for CTX
model_antibiotic_resistance(df_isolates, "CTX")




# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load your data
df = pd.read_csv("Datasets/Antibiotic_resistance/pcbi.1006258.s010.csv")
df.columns = df.columns.str.strip()

# Identify antibiotic columns (exclude metadata)
antibiotic_cols = [
    col for col in df.columns 
    if col not in ["ENA.Accession.Number", "Isolate", "Lane.accession", "Sequecning Status", "Year"]
]

# Store accuracy results
results = []

for abx in antibiotic_cols:
    df_abx = df[["Isolate", "Year", abx]].dropna()
    df_abx = df_abx[df_abx[abx].isin(["S", "I", "R"])]  # filter labels
    
    if len(df_abx) < 20:  # skip small datasets
        continue
    
    le = LabelEncoder()
    df_abx["Label"] = le.fit_transform(df_abx[abx])

    X = df_abx[["Year"]]  # simple feature
    y = df_abx["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        "Antibiotic": abx,
        "Accuracy": acc,
        "Samples": len(df_abx)
    })

# Convert to DataFrame and visualize
df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_results, x="Accuracy", y="Antibiotic", palette="mako")
plt.title("Model Accuracy per Antibiotic (Using Year as Feature)")
plt.tight_layout()
plt.show()

# Optional: print table
print(df_results)

