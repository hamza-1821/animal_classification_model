import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import time

dataset_path = "/home/hamza_ali/work/ml_cep/Animal"
VALID_EXT = ('.wav', '.flac', '.mp3', '.ogg')

def extract_mfcc(file_path, n_mfcc=13):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.concatenate([
            np.mean(mfccs, axis=1), np.var(mfccs, axis=1),
            np.mean(delta_mfccs, axis=1), np.var(delta_mfccs, axis=1),
            np.mean(delta_delta_mfccs, axis=1), np.var(delta_delta_mfccs, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

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

def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                               param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    start = time.time()
    X, y = load_data(dataset_path)
    print(f"Feature extraction: {time.time() - start:.2f} seconds")
    valid_indices = [i for i, x in enumerate(X) if x is not None]
    X, y = X[valid_indices], y[valid_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start = time.time()
    model = train_model(X_train_scaled, y_train)
    print(f"Training: {time.time() - start:.2f} seconds")
    evaluate_model(model, X_test_scaled, y_test)
    
    joblib.dump(model, 'animal_classifier_rf.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    def predict_animal(audio_path, model, scaler):
        features = extract_mfcc(audio_path)
        if features is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
            return model.predict(features_scaled)[0]
        return None
    
    test_file = "/home/hamza_ali/work/ml_cep/test_file/Dog_50.wav"
    model = joblib.load('animal_classifier_rf.pkl')
    scaler = joblib.load('scaler.pkl')
    print(f"Prediction for {test_file}: {predict_animal(test_file, model, scaler)}")
