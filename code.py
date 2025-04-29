# Import required libraries
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib  # For saving the model
import traceback

dataset_path = "/home/hamza_ali/work/ml_cep/Animal"

VALID_EXT = ('.wav', '.flac', '.mp3', '.ogg')

# 1. Data Preparation
def load_data(dataset_path):
    features, labels = [], []
    for animal in os.listdir(dataset_path):
        animal_dir = os.path.join(dataset_path, animal)
        if not os.path.isdir(animal_dir):
            continue
        for fname in os.listdir(animal_dir):
            if not fname.lower().endswith(VALID_EXT):
                continue
            file_path = os.path.join(animal_dir, fname)
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                features.append(mfcc)
                labels.append(animal)
    return np.array(features), np.array(labels)

# 2. Feature Extraction
def extract_mfcc(file_path, n_mfcc=13, max_pad_len=400):
    try:
        audio, sr = librosa.load(file_path, sr=None)  # Keep native sampling rate
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Pad/Crop MFCCs to fixed size
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0), (0,pad_width)), mode='constant')
        
        return mfccs.flatten()  # Flatten to 1D array
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        #traceback.print_exc()
        return None

# 3. Training Pipeline
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, 
                                  random_state=42,
                                  class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# 4. Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Path to your audio dataset
    DATASET_PATH = "/home/hamza_ali/work/ml_cep/Animal"
    
    # Load and preprocess data
    X, y = load_data(DATASET_PATH)
    
    # Remove None values from failed feature extractions
    valid_indices = [i for i, x in enumerate(X) if x is not None]
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    classes = np.unique(y)
    evaluate_model(model, X_test, y_test)
    
    # Save model for later use
    joblib.dump(model, 'animal_classifier_rf.pkl')
    
    # Example prediction on new audio
    def predict_animal(audio_path, model):
        features = extract_mfcc(audio_path)
        if features is not None:
            features = features.reshape(1, -1)
            proba = model.predict_proba(features)[0]
            for animal, p in zip(model.classes_, proba):
                print(f"{animal}: {p:.2%}")
            return model.predict(features)[0]
        return None
    
    #the sound that we want to test
    test_file = "/home/hamza_ali/work/ml_cep/test_file/Kus_110.wav"
    print(f"\nPrediction for {test_file}: {predict_animal(test_file, model)}")
