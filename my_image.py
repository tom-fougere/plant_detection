import re


class MergedImage:
    def __init__(self):
        self.nb_lines = 0
        self.nb_columns = 0
        self.images = dict()

    def add_image(self, image, axis):

        # Update the number of lines or columns
        if axis == 0 or self.nb_lines == 0:
            self.nb_lines += 1
        if axis == 1 or self.nb_columns == 0:
            self.nb_columns += 1

        self.images['l' + str(self.nb_lines - 1) + 'c' + str(self.nb_columns - 1)] = image

    def add_images(self, images: list, axis):

        for img in images:
            self.add_image(img, axis=axis)

    def add_merged_image(self, merged_image, axis):

        # Add the images in the current mergedImage
        for key, value in merged_image.images.items():
            index_c = key.find('c')
            cur_line = key[1:index_c]
            cur_column = key[index_c + 1:]

            if axis == 0:
                new_column = cur_column
                new_line = int(cur_line) + self.nb_lines
            else:
                new_column = int(cur_column) + self.nb_columns
                new_line = cur_line
            new_key = 'l' + str(new_line) + 'c' + str(new_column)

            self.images[new_key] = value

        # Update the number of lines/columns
        if axis == 0:
            self.nb_lines += merged_image.nb_lines
        else:
            self.nb_columns += merged_image.nb_columns

        print('end')



