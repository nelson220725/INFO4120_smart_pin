import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class AudioPositionClassifier:
    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = ["sitting_laptop", "standing_door", "standing_laptop"]
        
    def build_model(self):
        inputs = keras.Input(shape=self.input_shape)

        x = layers.Conv1D(32, kernel_size = 3, activation = "relu", padding = "same")(inputs)
        x = layers.MaxPooling1D(pool_size = 2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv1D(64, kernel_size = 3, activation = "relu", padding = "same")(x)
        x = layers.MaxPooling1D(pool_size = 2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv1D(128, kernel_size = 3, activation = "relu", padding = "same")(x)
        x = layers.MaxPooling1D(pool_size = 2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation = "softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def load_data(
        self,
        data_path,
        file_pattern = None,
        target_shape = None,
    ):
        class_to_label = {"sitting_laptop": 0, "standing_door": 1, "standing_laptop": 2}
        
        data = []
        labels = []
        files_loaded = 0
        class_counts = {class_name: 0 for class_name in class_to_label.keys()}
        shapes = {}

        if file_pattern:
            all_numpy_files = glob.glob(os.path.join(data_path, file_pattern))
        else:
            all_numpy_files = glob.glob(os.path.join(data_path, "*.npy"))

        print(f"Found {len(all_numpy_files)} numpy files")

        if len(all_numpy_files) == 0:
            raise ValueError(
                f"No files found in {data_path}"
            )
        
        for file in all_numpy_files:
            try:
                arr = np.load(file)
                shape = arr.shape
                if shape in shapes:
                    shapes[shape] += 1
                else:
                    shapes[shape] = 1
            except Exception as e:
                print(f"Error loading {os.path.basename(file)}: {e}")

        if target_shape is None:
            if len(shapes) > 1:
                most_common_shape = max(shapes.items(), key=lambda x: x[1])[0]
                target_shape = most_common_shape
            else:
                target_shape = list(shapes.keys())[0]

        print(f"Using target shape: {target_shape}")

        for file in all_numpy_files:
            filename = os.path.basename(file)

            label = None
            for class_name, class_label in class_to_label.items():
                if class_name in filename:
                    label = class_label
                    class_counts[class_name] += 1
                    break

            if label is None:
                print(f"Skipping {filename} - couldn't determine class")
                continue

            try:
                audio_features = np.load(file)

                if files_loaded == 0:
                    print(f"First file shape: {audio_features.shape}")
                    print(f"Data type: {audio_features.dtype}")

                if audio_features.shape != target_shape:
                    print(
                        f"Reshaping {filename} from {audio_features.shape} to {target_shape}"
                    )

                    # Reshape logic for 2D arrays
                    if len(target_shape) == 2: 
                        reshaped = np.zeros(target_shape, dtype = audio_features.dtype)
                        min_rows = min(audio_features.shape[0], target_shape[0])
                        min_cols = min(audio_features.shape[1], target_shape[1])
                        reshaped[:min_rows, :min_cols] = audio_features[
                            :min_rows, :min_cols
                        ]
                        audio_features = reshaped

                data.append(audio_features)
                labels.append(label)
                files_loaded += 1
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        if files_loaded == 0:
            raise ValueError(
                "No files, check file naming and format."
            )

        X = np.array(data)
        y = np.array(labels)
        
        return X, y

    def train_with_cross_validation(
        self, 
        X, 
        y, 
        n_splits = 3, 
        epochs = 30, 
        batch_size = 2, 
        stratified = True,
        random_state = 42,
        early_stopping = True
    ):

        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_samples_per_class = min(class_counts)
        
        if min_samples_per_class < n_splits:
            print(f"Not enough samples per class for {n_splits}-fold cross-validation!")
            print(f"Minimum class has only {min_samples_per_class} samples")
            print(f"Reducing to {min_samples_per_class}-fold cross-validation")
            n_splits = min_samples_per_class
        
        # Configure cross-validation strategy
        if stratified:
            kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
        else:
            kf = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)
        
        # Store results
        fold_results = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "histories": [],
            "models": [],
            "predictions": [],
            "confusion_matrices": []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\nFold {fold+1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"  Training samples: {len(X_train)}")
            print(f"  Validation samples: {len(X_val)}")
            
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            val_classes, val_counts = np.unique(y_val, return_counts=True)
            
            print("  Training class distribution:", dict(zip(train_classes, train_counts)))
            print("  Validation class distribution:", dict(zip(val_classes, val_counts)))
            
            # Create model for each fold
            self.model = None
            self.build_model()
            
            callbacks = []
            if early_stopping:
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        patience = 10, 
                        restore_best_weights = True, 
                        monitor = "val_accuracy"
                    )
                )
                callbacks.append(
                    keras.callbacks.ReduceLROnPlateau(
                        factor = 0.5, 
                        patience = 5, 
                        min_lr = 1e-5, 
                        monitor = "val_accuracy"
                    )
                )
            
            history = self.model.fit(
                X_train,
                y_train,
                validation_data = (X_val, y_val),
                epochs = epochs,
                batch_size = batch_size,
                callbacks = callbacks,
                verbose = 1
            )
            
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            train_loss, train_accuracy = self.model.evaluate(X_train, y_train, verbose=0)
            
            print(f"  Fold {fold+1} - Training accuracy: {train_accuracy:.4f}")
            print(f"  Fold {fold+1} - Validation accuracy: {val_accuracy:.4f}")
            
            y_pred = np.argmax(self.model.predict(X_val), axis = 1)
            cm = confusion_matrix(y_val, y_pred)
            
            fold_results["train_loss"].append(train_loss)
            fold_results["train_accuracy"].append(train_accuracy)
            fold_results["val_loss"].append(val_loss)
            fold_results["val_accuracy"].append(val_accuracy)
            fold_results["histories"].append(history.history)
            fold_results["models"].append(self.model)
            fold_results["predictions"].append(y_pred)
            fold_results["confusion_matrices"].append(cm)
        
        avg_train_acc = np.mean(fold_results["train_accuracy"])
        avg_val_acc = np.mean(fold_results["val_accuracy"])
        std_val_acc = np.std(fold_results["val_accuracy"])
        
        print("\nCross-Validation Results:")
        print(f"  Average Training Accuracy: {avg_train_acc:.4f}")
        print(f"  Average Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
        
        self.history = history
        
        return fold_results
    
    def plot_cross_validation_results(self, results):

        n_folds = len(results["val_accuracy"])
        
        plt.figure(figsize = (12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, n_folds+1), results["train_accuracy"], label = "Training", alpha=0.7)
        plt.bar(range(1, n_folds+1), results["val_accuracy"], label = "Validation", alpha=0.7)
        plt.axhline(y = np.mean(results["val_accuracy"]), color = 'r', linestyle='--', 
                   label = f'Mean Val Acc: {np.mean(results["val_accuracy"]):.4f}')
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.title("Accuracy by Fold")
        plt.xticks(range(1, n_folds+1))
        plt.ylim(0, 1.1)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        avg_cm = np.mean(results["confusion_matrices"], axis=0)
        sns.heatmap(
            avg_cm,
            annot = True,
            fmt = ".1f",
            cmap = "Blues",
            xticklabels = self.class_names,
            yticklabels = self.class_names,
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Average Confusion Matrix")
        
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        for i, history in enumerate(results["histories"]):
            plt.plot(history["accuracy"], label = f"Fold {i+1}")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        for i, history in enumerate(results["histories"]):
            plt.plot(history["val_accuracy"], label = f"Fold {i+1}")
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        for i, history in enumerate(results["histories"]):
            plt.plot(history["loss"], label = f"Fold {i+1}")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        for i, history in enumerate(results["histories"]):
            plt.plot(history["val_loss"], label = f"Fold {i+1}")
        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis = 1)

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model

    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot = True,
            fmt = "d",
            cmap = "Blues",
            xticklabels = self.class_names,
            yticklabels = self.class_names,
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        print(classification_report(y_test, y_pred, target_names=self.class_names))


if __name__ == "__main__":
    input_shape = (2400, 2843)
    classifier = AudioPositionClassifier(input_shape=input_shape)
    classifier.build_model()
    classifier.model.summary()

    data_dir = "/Users/nzhan/Downloads/Cornell/INFO4120/final/data"
    
    X, y = classifier.load_data(data_dir)
    
    cv_results = classifier.train_with_cross_validation(
        X, y, 
        n_splits = 10,          
        epochs = 30,
        batch_size = 2,
        stratified = True,     
        random_state = 42,
        early_stopping = True
    )
    
    classifier.plot_cross_validation_results(cv_results)