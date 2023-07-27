def compute_max_min(graspings):

    mean, std = None, None
    
    mean = np.mean(images, axis=0)
    mean = np.mean(mean, axis=0)
    mean = np.mean(mean, axis=0)
    print(mean.shape)
    
    """
    std = np.std(images, axis=0)
    std = np.std(std, axis=0)
    std = np.std(std, axis=0)
    print(std.shape)
    """
    
    std = np.array([np.std(images[:,:,:,0]),np.std(images[:,:,:,1]),np.std(images[:,:,:,2])])
    
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return mean, std



class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(-1, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # Rescale the given images:                                            #
        #   - from (self._data_min, self._data_max)                            #
        #   - to (self.min, self.max)                                          #
        ########################################################################

        # normalization of the image max and min values
        images_transformed = (images-self._data_min)/(self._data_max - self._data_min)*(self.max-self.min) + self.min

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images_transformed