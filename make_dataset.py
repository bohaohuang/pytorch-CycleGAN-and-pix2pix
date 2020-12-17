"""

"""


# Built-in
import os

# Libs
import skimage.transform
import numpy as np
import toolman as tm
from tqdm import tqdm

# Own modules


def get_rgb_path(fold_name, subdir):
    if len(subdir) > 0:
        base_dir = f'/hdd/style_transfer/target/{subdir}/{fold_name}'
    else:
        base_dir = f'/hdd/style_transfer/target/{fold_name}'
    return os.path.join(base_dir, 'test_latest', 'images')


def create_mrs_dataset(model_name, lbl_path, vis=True, subdir='', up_factor=None, dest_dir=r'/hdd/mrs', rgb_path=None):
    if rgb_path is None:
        rgb_path = get_rgb_path(model_name, subdir)

    rgb_files = tm.misc_utils.get_files(rgb_path, '*fake.png')
    lbl_files = tm.misc_utils.get_files(lbl_path, '*.png')
    rgb_list = [tm.misc_utils.get_file_name_no_extension(a)[:-5] for a in rgb_files]
    lbl_files = [a for a in lbl_files if tm.misc_utils.get_file_name_no_extension(a).replace('_BGT', '_RGB') in rgb_list]
    assert len(rgb_files) == len(lbl_files)

    dest_dir = os.path.join(dest_dir, model_name)
    patch_dir = os.path.join(dest_dir, 'patches')
    tm.misc_utils.make_dir_if_not_exist(dest_dir)
    tm.misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(dest_dir, 'file_list_train.txt'), 'w+')

    for rgb_file, lbl_file in tqdm(zip(rgb_files, lbl_files), total=len(rgb_files)):
        rgb = tm.misc_utils.load_file(rgb_file)
        if up_factor is not None:
            rgb = skimage.transform.resize(rgb, (rgb.shape[0]*up_factor, rgb.shape[1]*up_factor), anti_aliasing=True,
                                           preserve_range=True).astype(np.uint8)
        tm.misc_utils.save_file(os.path.join(patch_dir, os.path.basename(rgb_file)), rgb)
        os.system(f'cp {lbl_file} {patch_dir}')
        record_file_train.write('{} {}\n'.format(os.path.basename(rgb_file), os.path.basename(lbl_file)))

        if vis:
            lbl = tm.misc_utils.load_file(lbl_file)
            tm.vis_utils.compare_figures([rgb, lbl], (1, 2), (12, 5))

    record_file_train.close()


def make_city_dataset(ds_name, city_name, dest_dir=r'/hdd/mrs', vis=False, seed=0, val_per=0.14, epoch_num=30):
    rgb_path = os.path.join(f'/hdd/style_transfer/{ds_name}/{city_name}/test_{epoch_num}/images')
    lbl_path = r'/hdd/mrs/synthinel_v104/building/ps512_pd0_ol0/patches'
    np.random.seed(seed)
    if ds_name == 'inria':
        city_name = city_name.split('_')[0]

    rgb_files = tm.misc_utils.get_files(rgb_path, '*fake.png')
    lbl_files = tm.misc_utils.get_files(lbl_path, '*.png')
    assert len(rgb_files) == len(lbl_files)

    dest_dir = os.path.join(dest_dir, f'{ds_name}_syn')
    patch_dir = os.path.join(dest_dir, 'patches')
    tm.misc_utils.make_dir_if_not_exist(dest_dir)
    tm.misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(dest_dir, f'file_list_train_{city_name}.txt'), 'w+')
    record_file_valid = open(os.path.join(dest_dir, f'file_list_valid_{city_name}.txt'), 'w+')

    total_len = len(rgb_files)
    valid_len = int(np.floor(total_len * val_per))
    rand_idx = np.random.permutation(np.arange(total_len))
    for idx in tqdm(rand_idx):
        rgb_file, lbl_file = rgb_files[idx], lbl_files[idx]
        rgb = tm.misc_utils.load_file(rgb_file)
        rgb = skimage.transform.resize(rgb, (rgb.shape[0]*2, rgb.shape[1]*2), anti_aliasing=True,
                                       preserve_range=True).astype(np.uint8)
        rgb_file_save_name = tm.misc_utils.get_file_name_no_extension(rgb_file)[:-4] + city_name + '.png'
        lbl_file_save_name = tm.misc_utils.get_file_name_no_extension(lbl_file) + '_' + city_name + '.png'
        tm.misc_utils.save_file(os.path.join(patch_dir, rgb_file_save_name), rgb)
        os.system(f'cp {lbl_file} {os.path.join(patch_dir, lbl_file_save_name)}')
        if idx <= valid_len:
            record_file_valid.write('{} {}\n'.format(os.path.basename(rgb_file_save_name),
                                                     os.path.basename(lbl_file_save_name)))
        else:
            record_file_train.write('{} {}\n'.format(os.path.basename(rgb_file_save_name),
                                                     os.path.basename(lbl_file_save_name)))

        if vis:
            lbl = tm.misc_utils.load_file(lbl_file)
            tm.vis_utils.compare_figures([rgb, lbl], (1, 2), (12, 5))


def merge_city_dataset(ds_name, dest_dir=r'/hdd/mrs'):
    if ds_name == 'inria':
        city_names = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    else:
        city_names = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']

    dest_dir = os.path.join(dest_dir, f'{ds_name}_syn')
    record_file_train = os.path.join(dest_dir, 'file_list_train.txt')
    record_file_valid = os.path.join(dest_dir, 'file_list_valid.txt')

    train_files, valid_files = [], []
    for city_name in city_names:
        record_file_train_city = os.path.join(dest_dir, f'file_list_train_{city_name}.txt')
        record_file_valid_city = os.path.join(dest_dir, f'file_list_valid_{city_name}.txt')

        train_files.extend(tm.misc_utils.load_file(record_file_train_city))
        valid_files.extend(tm.misc_utils.load_file(record_file_valid_city))

    tm.misc_utils.save_file(record_file_train, train_files)
    tm.misc_utils.save_file(record_file_valid, valid_files)


def make_cycada_filelist():
    import toolman as tm
    data_dir = r'/hdd/mrs/synthinel_v211_random_li0_2/building/ps512_pd0_ol0'
    file_list = os.path.join(data_dir, 'file_list_train.txt')
    file_list = tm.misc_utils.load_file(file_list)
    file_list_train, file_list_valid, file_list_test = tm.misc_utils.random_split(file_list, (0.6, 0.2, 0.2))
    tm.misc_utils.save_file(os.path.join(data_dir, 'trainA_base.txt'), file_list_train)
    tm.misc_utils.save_file(os.path.join(data_dir, 'validA_base.txt'), file_list_valid)
    tm.misc_utils.save_file(os.path.join(data_dir, 'testA_base,txt'), file_list_test)


if __name__ == '__main__':
    # make_city_dataset('inria', 'vienna_syn104', vis=True)
    # merge_city_dataset('inria')

    make_cycada_filelist()

    # create_mrs_dataset(
    #     'syn205_random_dg_Khartoum',
    #     r'/hdd/mrs/synthinel_v205_random/building/ps512_pd0_ol0/patches',
    #     subdir='dg_city',
    #     vis=False
    # )
