import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iTinyImageNet200, iCUB200, iCARS
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        # shuffle perturbs the mapping from original class IDs to new indices (labels are nominal)
        self._setup_data(dataset_name, shuffle, seed)

        # Initialize per-task class counts for incremental learning; len(self._increments) is #tasks
        # e.g., [10,10,10,10,8]
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)    # total number of tasks

    def get_task_size(self, task):
        return self._increments[task]   # classes in the specified task

    def get_total_classnum(self):
        return len(self._class_order)   # total number of classes

    # Fetch data for given classes with optional per-class shot, random sampling (m_rate), and appended data
    def get_dataset(self, indices, source, mode, shot=None, appendent=None, ret_data=False, m_rate=None):
        """
        m_rate: ratio of dropped samples; if None, no random sampling
        shot: samples per class in training; if None, use all samples
        """
        # Select source split
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # Build transform pipeline by mode: train/flip/test
        if mode == "train":     # transforms.Compose chains multiple transforms
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.0),*self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        # indices holds class IDs to extract
        for idx in indices:
            # Optional random sampling
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            # Optional per-class shot limitation
            if shot is None:
                data.append(class_data)
                targets.append(class_targets)
            else:
                data.append(class_data[:shot])
                targets.append(class_targets[:shot])

        # Optionally append extra data
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
        # Concatenate arrays
        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    # Like get_dataset but additionally splits out a validation subset per class
    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        """
        This function prepares:
        self._train_data, self._train_targets
        self._test_data, self._test_targets
        self._train_trsf (train transforms)
        self._test_trsf (test transforms)
        self._common_trsf (common transforms)
        """
        idata = _get_idata(dataset_name)  # dataset class
        idata.download_data()   # download if needed

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path  # whether using local paths or downloaded data

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]  # class list in train set
        if shuffle:     # shuffle class order
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order   # possibly shuffled class order
        logging.info(self._class_order)

        # Map indices
        # Remap nominal labels to the new order
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        # Find indices where low_range <= y < high_range
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        # Return samples and labels for the class range
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        # Randomly sample training samples; m_rate is the drop ratio
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(0, len(idxes), size=int((1 - m_rate) * len(idxes)))
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:   # No random sampling
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DataManager2(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        # shuffle perturbs the mapping from original class IDs to new indices (labels are nominal)
        self._setup_data(dataset_name, shuffle, seed)

        # Initialize per-task class counts for incremental learning; len(self._increments) is #tasks
        # e.g., [10,10,10,10,8]
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)    # total number of tasks

    def get_task_size(self, task):
        return self._increments[task]   # classes in the specified task

    def get_total_classnum(self):
        return len(self._class_order)   # total number of classes

    # Fetch data for given classes with optional per-class shot, random sampling (m_rate), and appended data
    def get_dataset(self, indices, source, mode, shot=None, appendent=None, ret_data=False, m_rate=None):
        """
        m_rate: ratio of dropped samples; if None, no random sampling
        shot: samples per class in training; if None, use all samples
        """
        # Select source split
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # Build transform pipeline
        if mode == "train":     # transforms.Compose chains multiple transforms
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.0),*self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))
        trsf2 = transforms.Compose([*self._test_trsf, *self._common_trsf])
        data, targets = [], []
        # indices holds class IDs to extract
        for idx in indices:
            # Optional random sampling
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            # Optional per-class shot limitation
            if shot is None:
                data.append(class_data)
                targets.append(class_targets)
            else:
                data.append(class_data[:shot])
                targets.append(class_targets[:shot])

        # Optionally append extra data
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
        # Concatenate arrays
        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset2(data, targets, trsf, trsf2, self.use_path)
        else:
            return DummyDataset2(data, targets, trsf, trsf2, self.use_path)


    def _setup_data(self, dataset_name, shuffle, seed):
        """
        This function prepares:
        self._train_data, self._train_targets
        self._test_data, self._test_targets
        self._train_trsf (train transforms)
        self._test_trsf (test transforms)
        self._common_trsf (common transforms)
        """
        idata = _get_idata(dataset_name)  # dataset class
        idata.download_data()   # download if needed

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path  # whether using local paths or downloaded data

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]  # class list in train set
        if shuffle:     # shuffle class order
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order   # possibly shuffled class order
        logging.info(self._class_order)

        # Map indices
        # Remap nominal labels to the new order
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        # Find indices where low_range <= y < high_range
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        # Return samples and labels for the class range
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        # Randomly sample training samples; m_rate is the drop ratio
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(0, len(idxes), size=int((1 - m_rate) * len(idxes)))
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:   # No random sampling
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


"""
When training neural networks in PyTorch, these dataset classes can be paired with DataLoader
to batch-load and preprocess images for the model.
"""
class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            # Image.fromarray converts numpy arrays to PIL format, compatible with torchvision transforms
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label

class DummyDataset2(Dataset):
    def __init__(self, images, labels, trsf, trsf2, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.trsf2 = trsf2
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            # Image.fromarray converts numpy arrays to PIL format, compatible with torchvision transforms
            image = self.trsf(Image.fromarray(self.images[idx]))
            image_wo_aug = self.trsf2(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, image_wo_aug, label

def _map_new_class_index(y, order):
    # Map each label in y to its index in the given order
    return np.array(list(map(lambda x: order.index(x), y)))
    """
    Equivalent to:
    result = []
    for x in y:
        index = order.index(x)
        result.append(index)
    return np.array(result)
    """


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "tinyimagenet200":
        return iTinyImageNet200()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == 'cub200':
        return iCUB200()
    elif name == 'cars':
        return iCARS()
    elif name == 'flowers':
        return iFlowers()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
