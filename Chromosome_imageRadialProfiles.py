import pandas as pd
import numpy as np
import math
from nanoscope_imageReader import nanoscope_imageReader
import matplotlib.pyplot as plt
from skimage import measure, draw
import re
from scipy.optimize import curve_fit
import seaborn as sns



img_path = "Chromosome/Chrms_PABuffer_OlympusTip.007_Image_header.txt"
mask_path = "Chromosome/Chrms_PABuffer_OlympusTip.007_Crossprofiles_maskSelection.txt"

# loading the mask and converting it to boolean
mask = np.loadtxt(mask_path).astype(bool)
img = np.loadtxt(img_path)
# extracting information from the image header
with open(img_path, 'r') as doc:
    img_info = doc.read()
    # Both width and height are expressed in um in the header
    index = img_info.find("Width: ")
    img_width = float(re.findall("\d+.\d+", img_info[index:])[0])
    index = img_info.find("Height: ")
    img_height = float(re.findall("\d+.\d+", img_info[index:])[0])

# in the case the image is squared, pixel size in um
pixel_size = img_height / img.shape[0]
# print("Pixel size: ", pixel_size)
roi_img = np.copy(img)
roi_img[~mask] = 0  # Set pixels outside the mask to 0 (or any other value you prefer)

# Plotting the original image and the image selection (mask)
fig, ax = plt.subplots(1, 2)
ax[0].set_title("Original Image")
ax[0].imshow(img, cmap="afmhot")
ax[1].set_title("Mask selection")
ax[1].imshow(roi_img, cmap="afmhot")
plt.savefig("OriginalvsSelection.png", dpi=400)
plt.show()

# interested only in the points under the mask
selection = roi_img > 0
# Finding the coordinates of the centroid of the selection
# rows are the y coordinates while columns are the x coords
x_c, y_c = measure.centroid(selection)
y_c = round(y_c)
x_c = round(x_c)
# Finding contours of the selection
boundary = measure.find_contours(selection)
# rows are the y coordinates while columns are the x coords
y_boundary = (list(boundary)[0][:, 0])
x_boundary = (list(boundary)[0][:, 1])


# fig, ax = plt.subplots()
# plt.ion()
# ax.imshow(roi_img, zorder=1)
# ax.scatter(y_c, x_c, c='r', zorder=3)
df_lines = pd.DataFrame()
# the height value along the profiles
intensities = [[] for _ in range(len(y_boundary))]
# the coordinate along the profiles
profile_coords = [[] for _ in range(len(y_boundary))]
profiles_df = pd.DataFrame()

fig = plt.figure(figsize=(10,10))
ax = fig.subplots(2, 2)
ax[0, 0].imshow(roi_img)
ax[0, 0].scatter(x_c, y_c)
for i in range(len(y_boundary)):
    # ith point along the boundary
    x_b, y_b = y_boundary[i], x_boundary[i]
    # storing the intensity (height) from centroid to point on boundary
    intensities[i] = measure.profile_line(roi_img, (x_c, y_c), (x_b, y_b), reduce_func=np.max)
    # profile length
    plength = np.sqrt((x_c - x_b)**2 + (y_c - y_b)**2) * pixel_size
    # building the array for the height profile x coordinate
    profile_coords[i] = np.linspace(0, plength, len(intensities[i]))
    # storing height and spatial coordinate in dataframe
    df = pd.DataFrame({str(i) + "coord_um": profile_coords[i],
                       str(i) + "height_nm": intensities[i] * 1e9})
    profiles_df = pd.concat([profiles_df, df], axis=1)
    # Plotting the height profile on the image
    ax[0, 0].plot([y_c, y_b], [x_c, x_b])

# Storing the values in .csv for future usage
profiles_df.to_csv("RadialHeightProfiles.csv", sep=';')
# Reading the df
df = pd.read_csv("RadialHeightProfiles.csv", sep=';', index_col=0)
vals = pd.DataFrame()


# Correcting for the zeroes in the height
# (i.e., when the trace cuts outside the chromosome)
for i in range(1, df.shape[1], 2):
    df.iloc[:, i].replace(0, np.nan, inplace=True)
    # print(df.iloc[:, i].info())
    # Find the first occurrence of NaN
    first_nan_index = df.iloc[:, i].isnull().idxmax()
    if first_nan_index != 0:
        # replace with nan all the values following the first nan
        df.iloc[first_nan_index + 1:, i] = np.nan
# Plotting one representative height profile
ax[0,1].plot(df.iloc[:,0], df.iloc[:,1])
ax[0,1].set_xlabel("Distance (um)")
ax[0,1].set_ylabel("Height (nm)")
ax[0,1].set_title("Representative radial height profile")

# fig, ax = plt.subplots()
# Calculating the size of the starting window (3 points) and the
# minimum window size increase (distance between two consecutive points)
start_win_size = df.iloc[:, 0][2] - df.iloc[:, 0][0]
delta_win_size = df.iloc[:, 0][1] - df.iloc[:, 0][0]
print("Minimum window size increase (nm): ", delta_win_size)
# Rolling sigma (std) on each column
max_winsize = 60
min_size = 3
for size in range(min_size, max_winsize, 1):
    # Rolling std for all the columns in the df
    sigma_values = df.rolling(window=size, axis=0).std()
    # Calculating teh average for all the profiles
    mean_sigma = sigma_values.mean(skipna=True).values
    curr_df = pd.DataFrame({
        "sigma": mean_sigma[1::2],
        "profile": range(1, len(mean_sigma)//2 + 1, 1),
        "window": size,
        "window_size_nm": start_win_size
    })
    vals = pd.concat([vals, curr_df])
    start_win_size += delta_win_size


# Fitting until window size < specific value (um)
fit_df = vals[vals["window_size_nm"] <= .25]
x = fit_df["window_size_nm"].unique()
y = fit_df.groupby("window")["sigma"].mean()
# plotting the resulting std for each single radial profile
for profile in vals["profile"].unique():
    ax[1, 0].scatter(vals[vals["profile"]==profile]["window_size_nm"], vals[vals["profile"]==profile]["sigma"], s=3)
# plotting the fit
# ax[1, 0].scatter(x, y, s=5, c='lime', marker='D', label='fit')
# changing each color for each trace
ax[1,0].set_xscale("log")
ax[1,0].set_yscale("log")
ax[1, 0].set_xlabel("Length scale (nm)")
ax[1, 0].set_ylabel("Std dev.")
ax[1,0].set_title("Roughness scaling for each height profile")
ax[1,0].legend()
# plt.show()
def log_fit(x, m, q):
    y = q*x**m
    return y

p, cov = curve_fit(log_fit, x, y)
fit = log_fit(x, p[0], p[1])
print(f"Exponent: {p[0]}")
exponent = p[0]

# Plotting
# ax.scatter(y_c, x_c, c='r', zorder=3)
ax[1,1].errorbar(x=vals["window_size_nm"].unique(), y=vals.groupby("window_size_nm")["sigma"].mean(),
            yerr=vals.groupby("window")["sigma"].sem(), marker='o', markersize=5,
            linestyle="--", label="rolling_std", zorder=1)

ax[1, 1].plot(x, fit, c="r", label="fit", zorder=2)
ax[1, 1].set_xscale("log")
ax[1, 1].set_yscale("log")
ax[1, 1].text(.2, 20, "$\\alpha$: " + str(np.round(exponent,2)), fontsize="medium")
ax[1, 1].set_xlabel("Length scale (nm)")
ax[1, 1].set_ylabel("Std dev.")
ax[1, 1].set_title("Roughness scaling for entire chromosome")
ax[1, 1].legend()
plt.tight_layout()
plt.savefig("RoughnessScalingAnalysis.png", dpi=400)
plt.show()