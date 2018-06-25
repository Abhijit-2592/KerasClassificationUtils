# @Author: abhijit
# @Date:   25-06-2018
# @Email:  balaabhijit5@gmail.com
# @Last modified by:   abhijit
# @Last modified time: 25-06-2018
import os
import glob
import random
import shutil
import argparse
from argparse import RawTextHelpFormatter


def Train_valid_folder_split(input_dir, validation_split, test_split=0.0, output_dir=None, img_type='jpg', seed=5, copy=False):
    r"""Randomnly splits the total data into train and valid and puts in /train /valid folders. Either copies or creates symlink. Default symlink.

    NOTE
    For this function to work the input directory must follow the following hierarchy:
    The input directory must contain only sub_directories with names corresponding to their class names
    For eg: input_dir = flavour3 must contain subdirectories : with_dent and without_dent and the images placed inside
    the respective subdirectories.

    Keyword arguments
    input_dir -- str: path to the parent directory with sub directories (no default)
    validation_split -- float : (0,1) the fraction of validation data to be generated. (no default)
    output_dir -- str: (optional) (default None)
                        if given, the /train and /valid directories are created there
                        else, a directory called /data_split is created and /train and /valid are put there
    img_type -- str :(CASE SENSITIVE) the extension of the file without '.' for eg 'jpg','png' etc. (default 'jpg').
    seed -- int : to fix the random seed for shuffling the master data (default 5).\
    copy --boolean : Set true to copy (default False). By default it creates a symlink

    NOTE
    input directory contains sub directories pertaining to different categories
    optional to give output directory
    shuffles the datasplit everytime
    initialize the seed to get consistent result
    validation split is given as a float eg 0.3 for 30% validation split

    Output:
    No output

    """
    random.seed(seed)
    if test_split == 0.0:

        folders = os.listdir(input_dir)
        if output_dir is None:
            os.mkdir(input_dir + '/' + 'data_split')
            output_dir = input_dir + '/' + 'data_split'

        os.mkdir(output_dir + '/train')
        os.mkdir(output_dir + '/valid')

        for folder in folders:
            file_types = ("/*.jpg", "/*.jpeg", "/*.JPEG", "/*.png", "/*.PNG", "/*.bmp", "/*.BMP")
            images = []
            for file_type in file_types:
                images.extend(glob.glob(input_dir + '/' + folder + file_type))
            random.shuffle(images)
            valid = images[:int((len(images)*validation_split)+1)]
            train = images[int((len(images)*validation_split)+1):]
            os.mkdir(output_dir + '/train' + '/' + folder)
            os.mkdir(output_dir + '/valid' + '/' + folder)
            for train_image in train:
                if copy:
                    shutil.copy(train_image, output_dir + '/train' + '/' + folder)
                else:
                    os.symlink(train_image, output_dir + '/train' + '/' + folder + '/' + os.path.basename(train_image))
            for valid_image in valid:
                if copy:
                    shutil.copy(valid_image, output_dir + '/valid' + '/' + folder)
                else:
                    os.symlink(valid_image, output_dir + '/valid' + '/' + folder + '/' + os.path.basename(valid_image))

        print("The dataset is successfully split into /train and /valid subdirectories and it is present in {} directory".format(output_dir))

    else:
        folders = os.listdir(input_dir)
        if output_dir is None:
            os.mkdir(input_dir + '/' + 'data_split')
            output_dir = input_dir + '/' + 'data_split'

        os.mkdir(output_dir + '/train')
        os.mkdir(output_dir + '/valid')
        os.mkdir(output_dir + '/test')
        for folder in folders:
            file_types = ("/*.jpg", "/*.jpeg", "/*.JPEG", "/*.png", "/*.PNG", "/*.bmp", "/*.BMP")
            images = []
            for file_type in file_types:
                images.extend(glob.glob(input_dir + '/' + folder + file_type))
            random.shuffle(images)
            valid = images[:int((len(images)*validation_split)+1)]
            test = images[int((len(images)*validation_split)+1):int((len(images)*validation_split)+1) + int((len(images)*test_split)+1)]
            train = images[int((len(images)*validation_split)+1) + int((len(images)*test_split)+1):]
            os.mkdir(output_dir + '/train' + '/' + folder)
            os.mkdir(output_dir + '/valid' + '/' + folder)
            os.mkdir(output_dir + '/test' + '/' + folder)
            for train_image in train:
                if copy:
                    shutil.copy(train_image, output_dir + '/train' + '/' + folder)
                else:
                    os.symlink(train_image, output_dir + '/train' + '/' + folder + '/' + os.path.basename(train_image))
            for valid_image in valid:
                if copy:
                    shutil.copy(valid_image, output_dir + '/valid' + '/' + folder)
                else:
                    os.symlink(valid_image, output_dir + '/valid' + '/' + folder + '/' + os.path.basename(valid_image))

            for test_image in test:
                if copy:
                    shutil.copy(test_image, output_dir + '/test' + '/' + folder)
                else:
                    os.symlink(test_image, output_dir + '/test' + '/' + folder + '/' + os.path.basename(test_image))

        print("The dataset is successfully split into /train, /valid, /test subdirectories and it is present in {} directory".format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the data to training and validation sets \n First 2 arguments are Compulsary!!!", formatter_class=RawTextHelpFormatter)
    parser.add_argument("-if", "--input_dir",
                        help="path to input folder containing subdirectories with category name(COMPULSARY!!!)", type=str, metavar='', required=True)
    parser.add_argument("-vs", "--validation_split",
                        help="a float number representing proportion of validation split(COMPULSARY!!!)", type=float, metavar='', required=True)
    parser.add_argument("-ts", "--test_split", help="a optional float nunber representing proportion of test split (OPTIONAL)",
                        type=float, metavar='', default=0.0)
    parser.add_argument("-of", "--output_dir",
                        help="path to output folder(OPTIONAL). \n If not given it creates another dir called 'data_split' inside input_dir(OPTIONAL)",
                        type=str, metavar='', default=None)
    parser.add_argument("--seed", help="Seed to initialize the random shuffling(OPTIONAL)", type=int, metavar='', default=5)
    parser.add_argument("--copy", help="Boolean Give True if you want to copy instead of symlink(OPTIONAL)",
                        type=bool, metavar='', choices=[True, False], default=False)
    args = parser.parse_args()
    Train_valid_folder_split(input_dir=args.input_dir,
                             validation_split=args.validation_split,
                             test_split=args.test_split,
                             output_dir=args.output_dir,
                             seed=args.seed,
                             copy=args.copy)
