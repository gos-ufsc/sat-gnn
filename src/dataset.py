from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        """IMPLEMENT YOUR DATASET HERE"""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError
