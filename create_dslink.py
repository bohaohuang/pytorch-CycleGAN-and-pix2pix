"""

"""


# Built-in
import os

# Libs
import toolman as tm
from tqdm import tqdm

# Own modules


def inria_link_ds(dest_dir=r'/hdd/style_transfer/inria'):
    inria_dir = r'/hdd/mrs/inria/ps512_pd0_ol0'
    inria_patches = os.path.join(inria_dir, 'patches')
    inria_file = os.path.join(inria_dir, 'testB.txt')

    patch_names = tm.misc_utils.load_file(inria_file)
    patch_names = [a.split(' ')[0].strip() for a in patch_names]

    tm.misc_utils.make_dir_if_not_exist(dest_dir)

    for p in tqdm(patch_names):
        source_file = os.path.join(inria_patches, p)
        target_file = os.path.join(dest_dir, p)
        os.system(f'ln -s {source_file} {target_file}')


def dg_link_ds(dest_dir=r'/hdd/style_transfer/dg'):
    dg_dir = r'/hdd/mrs/deepglobe/14p_pd0_ol0'
    dg_patches = os.path.join(dg_dir, 'patches')
    dg_file = os.path.join(dg_dir, 'testB.txt')

    patch_names = tm.misc_utils.load_file(dg_file)
    patch_names = [a.split(' ')[0].strip() for a in patch_names]

    tm.misc_utils.make_dir_if_not_exist(dest_dir)

    for p in tqdm(patch_names):
        source_file = os.path.join(dg_patches, p)
        target_file = os.path.join(dest_dir, p)
        os.system(f'ln -s {source_file} {target_file}')


def synthetic_origin_link_ds(ds_name='synthinel_v205_random', dest_dir=r'/hdd/style_transfer', dest_name=None,
                             file_list_name='file_list_train.txt'):
    if dest_name is None:
        dest_dir = os.path.join(dest_dir, ds_name)
    else:
        dest_dir = os.path.join(dest_dir, dest_name)

    tm.misc_utils.make_dir_if_not_exist(dest_dir)

    source_dir = os.path.join(r'/hdd/mrs', ds_name, 'building', 'ps512_pd0_ol0')
    source_patches = os.path.join(source_dir, 'patches')
    source_file = os.path.join(source_dir, file_list_name)

    # dest_dir = r'/hdd/style_transfer/dg'
    # dg_dir = r'/hdd/mrs/deepglobe/14p_pd0_ol0'
    # dg_patches = os.path.join(dg_dir, 'patches')
    # dg_file = os.path.join(dg_dir, 'testB.txt')

    patch_names = tm.misc_utils.load_file(source_file)
    patch_names = [a.split(' ')[0].strip() for a in patch_names]

    tm.misc_utils.make_dir_if_not_exist(dest_dir)

    for p in tqdm(patch_names):
        source_file = os.path.join(source_patches, p)
        target_file = os.path.join(dest_dir, p)
        os.system(f'ln -s {source_file} {target_file}')


def syn_base(ds_name='synthinel_v205_random', dest_name='syn205_base'):
    ds_dir = os.path.join(r'/hdd/style_transfer', ds_name)
    dest_dir = os.path.join(r'/hdd/style_transfer', 'source', dest_name)

    tm.misc_utils.make_dir_if_not_exist(dest_dir)

    patches = tm.misc_utils.get_files(ds_dir, '*.jpg')
    patches = [a for a in patches if 'la60_saa300_li5.0' in a]

    for source_file in tqdm(patches):
        target_file = os.path.join(dest_dir, os.path.basename(source_file))
        os.system(f'ln -s {source_file} {target_file}')


def ds_city_files(ds_name='inria'):
    if ds_name == 'inria':
        ds_dir = r'/hdd/mrs/inria/ps512_pd0_ol0'
        city_names = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    else:
        ds_dir = r'/hdd/mrs/deepglobe/14p_pd0_ol0'
        city_names = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']
    train_files = tm.misc_utils.load_file(os.path.join(ds_dir, 'file_list_train.txt'))
    valid_files = tm.misc_utils.load_file(os.path.join(ds_dir, 'file_list_valid.txt'))

    for city_name in city_names:
        train_city_files = [a for a in train_files if city_name in a]
        valid_city_files = [a for a in valid_files if city_name in a]
        valid_len = len(valid_city_files)
        valid_city_files, test_city_files = valid_city_files[:valid_len//2], valid_city_files[valid_len//2:]

        save_dir = os.path.join(ds_dir, 'city_files', city_name)
        tm.misc_utils.make_dir_if_not_exist(save_dir)

        train_file_name = os.path.join(save_dir, 'trainB.txt')
        tm.misc_utils.save_file(train_file_name, train_city_files)
        train_file_name = os.path.join(save_dir, 'validB.txt')
        tm.misc_utils.save_file(train_file_name, valid_city_files)
        train_file_name = os.path.join(save_dir, 'testB.txt')
        tm.misc_utils.save_file(train_file_name, test_city_files)


if __name__ == '__main__':
    syn_base()
