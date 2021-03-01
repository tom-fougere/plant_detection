from manage_dataset import *
from shutil import rmtree

MY_PATH = 'tom/fougere/my_projects/'
SECOND_PATH = 'tom/fougere/my_projects/test/'


def test_make_directory():
    print('')

    # Call the tested function
    dir_first_path = make_directory(MY_PATH)

    # List of real directories
    list_dir_first_path = [x[0] for x in os.walk('tom')]
    print(list_dir_first_path)

    assert(len(list_dir_first_path) == 3)
    assert list_dir_first_path[0] == 'tom'
    assert list_dir_first_path[1] == 'tom\\fougere'
    assert list_dir_first_path[2] == 'tom\\fougere\\my_projects'
    assert(len(dir_first_path) == 3)
    assert dir_first_path[0] == './tom'
    assert dir_first_path[1] == './tom/fougere'
    assert dir_first_path[2] == './tom/fougere/my_projects'

    # Call the tested function with a new sub-dir
    dir_second_path = make_directory(SECOND_PATH)

    list_dir_second_path = [x[0] for x in os.walk('tom')]

    assert(len(list_dir_second_path) == 4)
    assert list_dir_second_path[0] == 'tom'
    assert list_dir_second_path[1] == 'tom\\fougere'
    assert list_dir_second_path[2] == 'tom\\fougere\\my_projects'
    assert list_dir_second_path[3] == 'tom\\fougere\\my_projects\\test'
    assert(len(dir_second_path) == 1)
    assert dir_second_path[0] == './tom/fougere/my_projects/test'

    # Delete tested directory
    rmtree('tom')
