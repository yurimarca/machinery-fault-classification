import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")  # Use non-GUI backend


# Data Ingestion

def load_filtered_dfs(data_path, filter_label):
    col_names = [
        'tachometer', 'underhang_axial', 'underhang_radiale', 'underhang_tangential',
        'overhang_axial', 'overhang_radiale', 'overhang_tangential', 'microphone'
    ]
    
    filtered_dfs = []
    
    for dirname, _, filenames in os.walk(data_path):
        filenames = sorted(filenames, key=lambda f: float(f.replace('.csv', '')))
        
        for filename in filenames:
            file_addr = os.path.join(dirname, filename)
            if file_addr.endswith('.csv'):
                label = "-".join(file_addr.split('.csv')[0].split("/")[-3:-1])
                if filter_label in label:
                    df = pd.read_csv(file_addr, names=col_names)
                    filtered_dfs.append(df)
    
    return filtered_dfs

# Feature Engineering

def downSampler(data, b):
    return data.groupby(data.index // b).mean().reset_index(drop=True)

def rolling_mean_data(df, window_size=100):
    df_copy = df.copy()
    df_copy[df_copy.select_dtypes(include=[np.number]).columns] = (
        df_copy[df_copy.select_dtypes(include=[np.number]).columns]
        .rolling(window=window_size, min_periods=1).mean()
    )
    return df_copy

def augment_features(df, window_size=100, downsample_factor=2500):
    downsampled_df = downSampler(df, downsample_factor).astype(np.float32)
    rolling_df = rolling_mean_data(downsampled_df, window_size).astype(np.float32)

    df.drop(columns=df.columns, inplace=True)  # Free memory before concatenation

    comb_df = pd.concat([downsampled_df, rolling_df.add_suffix('_rolling')], axis=1)
    
    return comb_df.dropna().reset_index(drop=True)

# PyTorch Dataset

class MachineryDataset(Dataset):
    def __init__(self, data, label_column='label'):
        self.labels = data[label_column].values.astype(np.float32)  # Ensure labels are float for BCELoss
        self.features = data.drop(columns=[label_column, 'time'], errors='ignore').values.astype(np.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y


class TimeSeriesMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout_prob=0.3):
        super(TimeSeriesMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_dim, 1)) 
        layers.append(nn.Sigmoid()) 

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Training Function


def train_model(model, train_loader, val_loader, epochs=50, lr=0.0005):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
        
        train_accuracy = correct_train / total_train
        
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs).squeeze()
                val_loss = criterion(outputs, targets)
                epoch_val_loss += val_loss.item()
                
                preds = (outputs > 0.5).float()
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)
        
        val_accuracy = correct_val / total_val
        
        history['train_loss'].append(epoch_train_loss / len(train_loader))
        history['val_loss'].append(epoch_val_loss / len(val_loader))
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {epoch_val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    return history

# Plot Function

def plot_metrics(history, fig_folder="../figures"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label="Train Loss", marker='o')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label="Validation Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss Over Epochs")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/training_loss_plot.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], label="Train Accuracy", marker='o')
    plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], label="Validation Accuracy", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Validation Accuracy Over Epochs")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/training_accuracy_plot.png")
    plt.close()


def plot_tsne(data, label_column='label', output_file="../figurestsne_plot.png", subset_size=1000):
    """
    Generates and saves a t-SNE visualization of the dataset.

    Parameters:
    - data: pandas DataFrame containing features and labels
    - label_column: column name for labels (default: 'label')
    - output_file: filename to save the t-SNE plot
    - subset_size: number of samples to use for faster computation
    """

    # Drop label column for feature extraction
    features = data.drop(columns=[label_column], errors='ignore')
    labels = data[label_column]

    # Sample a subset of data for efficiency
    subset_indices = np.random.choice(len(features), subset_size, replace=False)
    subset_features = features.iloc[subset_indices]
    subset_labels = labels.iloc[subset_indices]

    # Scale features
    scaler = StandardScaler()
    scaled_subset = scaler.fit_transform(subset_features)

    # Apply t-SNE directly
    tsne = TSNE(n_components=2, random_state=42, perplexity=20, learning_rate=100)
    tsne_results = tsne.fit_transform(scaled_subset)

    # Convert to DataFrame for visualization
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['label'] = subset_labels.values

    # Plot t-SNE
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue=tsne_df['label'].astype(str), palette='coolwarm', alpha=0.6)
    plt.title('t-SNE Visualization of Machinery Data (16 Features)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Label')
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"t-SNE plot saved to {output_file}")


def save_correlation_matrix(data, output_file="../figures/correlation_matrix.png"):
    """
    Generates and saves a correlation matrix heatmap of the dataset.

    Parameters:
    - data: pandas DataFrame containing numeric features
    - output_file: filename to save the correlation matrix heatmap
    """

    # Compute the correlation matrix
    corr_matrix = data.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Correlation matrix saved to {output_file}")


def evaluate_model(model, test_loader, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze() 
            preds = (outputs > threshold).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    unique_classes = np.unique(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    auc_roc = None
    if len(unique_classes) > 1:  # Check if at least two classes exist
        auc_roc = roc_auc_score(all_labels, all_preds)

    metrics = {
        "Accuracy": accuracy,
        "F1-score": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC-ROC": auc_roc
    }

    # Print only if value is not None
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")

    return metrics


def time_series_split(all_data, train_ratio=0.7, val_ratio=0.15):
    # Separate normal and anomaly data
    normal_data = all_data[all_data["label"] == 0]
    anomaly_data = all_data[all_data["label"] == 1]

    # Compute split indices
    def split_data(df, train_ratio, val_ratio):
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        return train_df, val_df, test_df
    
    def check_class_balance(df, dataset_name):
        class_counts = df["label"].value_counts()
        print(f"Class distribution in {dataset_name} set:")
        print(class_counts)
        print("-" * 40)

    # Split normal and anomaly data separately
    train_normal, val_normal, test_normal = split_data(normal_data, train_ratio, val_ratio)
    train_anomaly, val_anomaly, test_anomaly = split_data(anomaly_data, train_ratio, val_ratio)

    # Concatenate to form the final train, val, and test sets
    train_data = pd.concat([train_normal, train_anomaly]).sort_index()
    val_data = pd.concat([val_normal, val_anomaly]).sort_index()
    test_data = pd.concat([test_normal, test_anomaly]).sort_index()

    check_class_balance(train_data, "Train")
    check_class_balance(val_data, "Validation")
    check_class_balance(test_data, "Test")

    return train_data, val_data, test_data

import matplotlib.pyplot as plt
import seaborn as sns

def plot_evaluation_results(metrics, output_file="../figures/evaluation_results.png"):
    """
    Generates a bar plot of model evaluation metrics and saves it to an image file.

    Parameters:
    - metrics: Dictionary containing evaluation metrics (Accuracy, F1-score, Precision, Recall, AUC-ROC).
    - output_file: Filename to save the plot.
    """
    # Remove None values (e.g., if AUC-ROC wasn't computed)
    metrics = {k: v for k, v in metrics.items() if v is not None}

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
    plt.ylim(0, 1)  # Ensure scale is between 0-1
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.title("Model Evaluation Metrics")

    # Add text labels above bars
    for i, (key, value) in enumerate(metrics.items()):
        plt.text(i, value + 0.02, f"{value:.4f}", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Evaluation results plot saved to {output_file}")



# Main Execution

if __name__ == "__main__":
    data_path = "../data"
    hidden_dim = 3
    n_layers = 2

    # Create folder figures f it doesn't exist
    fig_folder = "../figures"

    os.makedirs(fig_folder, exist_ok=True)
    
    # Load normal and imbalance data
    normal_dfs = load_filtered_dfs(data_path, "normal")
    imbalance_dfs = load_filtered_dfs(data_path, "imbalance-6g")
    
    normal_df = pd.concat([augment_features(df) for df in normal_dfs], ignore_index=True)
    imbalance_df = pd.concat([augment_features(df) for df in imbalance_dfs], ignore_index=True)
    
    del normal_dfs, imbalance_dfs
    
    # Label the data
    normal_df["label"] = 0
    imbalance_df["label"] = 1
    
    all_data = pd.concat([normal_df, imbalance_df], ignore_index=True)

    save_correlation_matrix(all_data)

    plot_tsne(all_data, label_column='label', output_file="../figures/tsne_visualization.png")
    
    # Apply the new split method
    train_data, val_data, test_data = time_series_split(all_data)

    scaler = StandardScaler()
    train_data.iloc[:, :-1] = scaler.fit_transform(train_data.iloc[:, :-1])
    val_data.iloc[:, :-1] = scaler.transform(val_data.iloc[:, :-1])
    test_data.iloc[:, :-1] = scaler.transform(test_data.iloc[:, :-1])

   
    # Prepare data for PyTorch
    train_dataset = MachineryDataset(train_data)
    val_dataset = MachineryDataset(val_data)
    test_dataset = MachineryDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = TimeSeriesMLP(input_dim=train_dataset.features.shape[1], hidden_dim=hidden_dim, n_layers=n_layers)
    
    # Train the model
    history = train_model(model, train_loader, val_loader)

    plot_metrics(history)
    
    # Prepare test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the trained model on test data
    test_metrics = evaluate_model(model, test_loader)

    print(test_metrics)
    plot_evaluation_results(test_metrics, output_file="../figures/evaluation_plot.png")

