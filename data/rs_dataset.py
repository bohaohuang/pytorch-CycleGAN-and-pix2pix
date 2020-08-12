"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
import random
import toolman as tm
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from PIL import Image


class RSDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # A_dir = r'/data/users/bh163/data/mrs/synthinel_v205_random/building/ps512_pd0_ol0/'
        # B_dir = r'/hdd/mrs/inria/ps512_pd0_ol0/'
        # B_dir = r'/data/users/bh163/data/mrs/deepglobe/14p_pd0_ol0'
        # B_dir = r'/data/users/bh163/data/mrs/inria/ps512_pd0_ol0'
        # A_dir = r'/data/users/bh163/data/mrs/deepglobe/14p_pd0_ol0'
        self.dir_A = os.path.join(opt.a_dir, f'{opt.phase}A{opt.a_appendix}.txt')
        # self.dir_B = os.path.join(opt.b_dir, f'{opt.phase}B.txt')
        self.dir_B = os.path.join(opt.b_dir, 'city_files', opt.city_name, f'{opt.phase}B.txt')

        self.A_paths = tm.misc_utils.load_file(self.dir_A)
        self.B_paths = tm.misc_utils.load_file(self.dir_B)

        self.A_paths = [os.path.join(opt.a_dir, 'patches', a.strip().split(' ')[0]) for a in self.A_paths]
        self.B_paths = [os.path.join(opt.b_dir, 'patches', b.strip().split(' ')[0]) for b in self.B_paths]

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)
