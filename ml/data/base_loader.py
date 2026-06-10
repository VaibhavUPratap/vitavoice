import abc
from sklearn.model_selection import train_test_split
import numpy as np

class BaseDatasetLoader(abc.ABC):
    """
    Abstract Base Class for swapping clinical datasets in VitaVoice.
    Ensures a unified interface for loading data, features, and model training.
    """
    
    @abc.abstractmethod
    def load_samples(self):
        """
        Loads the samples, features, and labels from the source dataset.
        Returns:
            X (np.ndarray): Fused or tabular features.
            y (np.ndarray): Labels (0 = healthy, 1 = pathological).
            metadata (list): List of dictionaries containing sample identifiers/metadata.
        """
        pass
        
    def get_train_test_splits(self, test_size=0.2, random_state=42):
        """
        Prepares standard stratified train/test splits for model evaluation.
        """
        X, y, metadata = self.load_samples()
        
        # Ensure splits are stratified to keep the same proportion of target status
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, metadata, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'meta_train': meta_train,
            'meta_test': meta_test
        }
