from numba import jit

@jit(nopython=False)
def data_loop(filepath):
    folder_lst = []
    for folder in os.listdir(filepath): # we should parallelize these for loops (but need to preserve order of folder_lst and tif_lst)
        tif_lst = []
        for tif in os.listdir(filepath + folder):
            # Read the tif as a numpy array
            raw_frame = rio.open(filepath + folder + '/' + tif).read(1).astype('int')

            # Mask missing data
            missing = np.where(raw_frame < 0, -1, raw_frame)

            # Mask the array
            frame = np.where(missing > 0, 1, missing) # we also need to deal with missing values

            # Add channel dimension
            frame = np.expand_dims(frame, axis=-1)
            # Add frame dimension
            frame = np.expand_dims(frame, axis=0)
            # Add sample dimension
            frame = np.expand_dims(frame, axis=0)

            tif_lst.append(frame)
        # Concatenate rasters
        folder_lst.append(np.concatenate(tif_lst, axis=0))

    # Concatenate frames
    dataset = np.concatenate(folder_lst, axis=1)
    return dataset
