
import os
import patoolib


def do_zip(label: str, save_folder: str, top_level_folder=None):
    ''' Creates a labeled zip archive of all .py files in top_level_folder, thus freezing the state of the code base.
    label: Name of the zip file to be created, preferable identifiable at later times, e.g. ISO datetime.
    save_folder: Absolute path where the zip archive should be stored
    top_level_folder: The "entry point" of the scan for .py files. If not provided, this function will try to find the <repo>/code folder on its own
    '''
    if top_level_folder is None:
        surrounding_path = os.path.realpath(__file__)
        surrounding_path = surrounding_path[:surrounding_path.rfind(os.path.sep)]
        assert surrounding_path.endswith("utils"), "Zipper tool has bad directory !"
        # now we know that we are in code/utils, so we just have to cut away utils
        top_level_folder = surrounding_path[:surrounding_path.rfind("/utils")]

    files_to_zip = [os.path.join(dp, f) for dp, dn, filenames in os.walk(top_level_folder) \
    for f in filenames if os.path.splitext(f)[1] == '.py' and not os.path.splitext(f)[0].endswith("__init__")]

    patoolib.create_archive(os.path.join(save_folder, label + ".zip"), files_to_zip)
