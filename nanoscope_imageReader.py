import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re


# file_path = r"C:\Users\Andrea\Desktop\VU_postdoc\Lab\20220406_tests_DNP10_tip_CII_CTRL\Chromosome_100x.017"
# file_path = r"C:\Users\Andrea\Desktop\VU_postdoc\Lab\20220624_SNL10_CI_CTRL_trials_newAPTES_plasma\Chromosome_100x_SNLA_noAPTES.006"
def nanoscope_imageReader(file_path):
    with open(file_path, 'rb') as path:
        # print(path)
        header_info = str(path.read())
        index = header_info.find('Samps/line: ')
        img_px_size = int(re.findall("\d+", header_info[index:])[0])
        # all the data of the images begin at position 40960
        path.seek(40960)  # where the binary data begins
        # converting the data into the correct format
        all_img_data = np.fromfile(path, dtype=np.int16)
        # found the image data, total of 5x the pixel x pixel size of an image cause it contains all the 5 channel that gets recorded
        # print(len(all_img_data))
        # isolate the data of the channel n.1 (height)
        height_data = img_px_size**2
        height_img = all_img_data[:height_data]
        height_img_retrace = all_img_data[height_data: 2 * height_data]
        height_img.shape = (img_px_size, img_px_size)
        height_img_retrace.shape = (img_px_size, img_px_size)
        df = pd.DataFrame(height_img)
        df_retrace = pd.DataFrame(height_img_retrace)
        #  rotate the dataframe i.e. the image
        df = df.iloc[::-1]
        df_retrace = df_retrace.iloc[::-1]
        # print(df.head())
        path.seek(0)
    return df, df_retrace

# df = nanoscope_imageReader(file_path)[0]
# df_retrace = nanoscope_imageReader(file_path)[1]
#
# # correcting for stripes and convolution effects by superimposing trace and retrace
# abs_diff = np.abs(df - df_retrace)
# img_sum = df + df_retrace
# corrected = (img_sum - abs_diff) / 2
# # #  Plotting
# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(df, cmap='afmhot')
# ax[0].set_title('raw image (Trace)')
# ax[1].imshow(df_retrace, cmap='afmhot')
# ax[1].set_title('raw image (Retrace)')
# ax[2].imshow(corrected, cmap='afmhot')
# ax[2].set_title('raw image (Corrected)')
# plt.show()