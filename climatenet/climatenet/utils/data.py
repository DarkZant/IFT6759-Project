from torch.utils.data import Dataset
from os import listdir, path
import xarray as xr
from climatenet.utils.utils import Config

SKIP_FILES = {
    "data-2000-04-17-01-1_5.nc",
    "data-2008-10-03-01-1_0.nc",
}

class ClimateDataset(Dataset):
    '''
    The basic Climate Dataset class. 

    Parameters
    ----------
    path : str
        The path to the directory containing the dataset (in form of .nc files)
    config : Config
        The model configuration. This allows to automatically infer the fields we are interested in 
        and their normalisation statistics

    Attributes
    ----------
    path : str
        Stores the Dataset path
    fields : dict
        Stores a dictionary mapping from variable names to normalisation statistics
    files : [str]
        Stores a sorted list of all the nc files in the Dataset
    length : int
        Stores the amount of nc files in the Dataset
    '''
  
    def __init__(self, path: str, config: Config):
        self.path: str = path
        self.fields: dict = config.fields
        
        self.files: [str] = [f for f in sorted(listdir(self.path)) if f[-3:] == ".nc" and f not in SKIP_FILES]
        self.length: int = len(self.files)
      
    def __len__(self):
        return self.length

    def normalize(self, features: xr.DataArray):
        for variable_name, stats in self.fields.items():   
            var = features.sel(variable=variable_name).values
            var -= stats['mean']
            var /= stats['std']

    def get_features(self, dataset: xr.Dataset):
        features = dataset[list(self.fields)].to_array()
        self.normalize(features)
        return features.transpose('time', 'variable', 'lat', 'lon')

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        try:
            dataset = xr.load_dataset(file_path)
        except Exception as e:
            print(f"[WARNING] Corrupted file skipped: {file_path}")

            # Choisir un autre index valide
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)
        return self.get_features(dataset)
    @staticmethod
    def collate(batch):
        return xr.concat(batch, dim='time')

class ClimateDatasetLabeled(ClimateDataset):
    '''
    The labeled Climate Dataset class. 
    Corresponds to the normal Climate Dataset, but returns labels as well and batches accordingly
    '''

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx])

        try:
            dataset = xr.load_dataset(file_path)
        except Exception:
            print(f"[WARNING] Corrupted file skipped: {file_path}")

            # passer au fichier suivant
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)

        return self.get_features(dataset), dataset['LABELS']

    @staticmethod 
    def collate(batch):
        data, labels = map(list, zip(*batch))
        return xr.concat(data, dim='time'), xr.concat(labels, dim='time')