import re


class MergedImage:
    def __init__(self):
        self.nb_lines = 0
        self.nb_columns = 0
        self.images = dict()

    def add_image(self, image, axis):

        self.images['l' + str(self.nb_lines) + 'c' + str(self.nb_columns)] = image

        # Update the number of lines or columns
        if axis == 0:
            self.nb_lines += 1
        else:
            self.nb_columns += 1

    def add_images(self, images: list, axis):

        for img in images:
            self.add_image(img, axis=axis)

    def add_merged_image(self, merged_image, axis):

        for key, value in merged_image.images.items():
            index_c = key.find('c')
            column = key[1:index_c]
            line = key[index_c + 1:]

            self.add_image(value, axis=axis)



