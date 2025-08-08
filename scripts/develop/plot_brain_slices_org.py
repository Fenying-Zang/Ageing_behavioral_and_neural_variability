
#%% plot different 
# import brain atlas and brain regions objects
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
import os
from utils import config
import matplotlib.pyplot as plt

datapath = config.datapath
figpath = config.figpath
ba = AllenAtlas()
br = BrainRegions() 

#%%
fig, axs = plt.subplots(1, 1, figsize=(15, 10))

# plot coronal slice at ap = -2000 um
position_list = [-3111, -1983, 965, 1697]
for position in position_list:

    ap_para= position
    ap = ap_para / 1e6
    # Allen mapping
    ba.plot_cslice(ap, volume='annotation', mapping='Allen', ax=axs)
    # _ = axs.set_title('Allen')
    axs.set_axis_off()
    axs.set_facecolor('0')
    # # Beryl mapping
    # ba.plot_cslice(ap, volume='annotation', mapping='Beryl', ax=axs[1])
    # _ = axs[1].set_title('Beryl')
    # # Cosmos mapping
    # ba.plot_cslice(ap, volume='annotation', mapping='Cosmos', ax=axs[2])
    # _ = axs[2].set_title('Cosmos')

    fig.savefig(os.path.join(figpath, f"brain_slice_coronal_{str(ap_para)}.png"),dpi=1500)

#%%
import matplotlib.pyplot as plt
position_list = [-3111, -1983, 965, 1697]

for position in position_list[0:1]:
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    ap_para= position
    ap = ap_para / 1e6
    # 绘图：例如绘制 annotation 切片
    axs = ba.plot_cslice(
        ap,
        # ml_coordinate=0.5,
        volume='boundary',  # 或 'boundary', 'image', etc.
        mapping='Allen',
        alpha=1.0              # 可选，确保不透明显示图像本身
    )

    # 去除坐标轴（可选，让图更干净）
    axs.axis('off')

    # 获取对应 figure 并保存为透明背景矢量图
    fig = axs.figure
    fig.savefig(os.path.join(figpath, 
        f"sagittal_slice_test_{position}.svg"),      # 文件名（可改为 .pdf, .png）
        format="svg",              # 文件格式
        transparent=True,          # 背景透明
        bbox_inches='tight',       # 裁剪空白边缘
        pad_inches=0               # 去掉额外边距
    )


#%%
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

position_list = [-3111, -1983, 965, 1697]

# 构建一个带透明背景的 colormap
cmap = cm.get_cmap('bone').copy()  # 或 'gray'
cmap.set_under((1, 1, 1, 0))        # 背景透明

for position in position_list[0:1]:
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    ap = position / 1e6

    axs = ba.plot_cslice(
        ap,
        volume='annotation',
        mapping='Allen',
        cmap=cmap,
        vmin=0.01  # 非零区域显示，0 视为 under → 透明
    )

    axs.axis('off')
    axs.set_facecolor((0, 0, 0, 0))
    fig.patch.set_alpha(0)

    filename = os.path.join(figpath, f"slice_{position}.pdf")
    fig.savefig(
        filename,
        format='pdf',
        transparent=True,
        bbox_inches='tight',
        pad_inches=0,
        dpi=600  # 高分辨率避免锯齿过重
    )
    plt.close(fig)


#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from matplotlib.patches import Polygon

def plot_annotation_slice_colored_allen(im, ba, save_path=None, dpi=600):
    if im.ndim == 3:
        im = im[:, :, 0]

    fig, ax = plt.subplots(figsize=(15, 10))

    # 建立标准颜色映射：region ID → normalized RGB
    rgb_normalized = ba.regions.rgb / 255.0
    region_color_map = {rid: tuple(rgb_normalized[i]) for i, rid in enumerate(ba.regions.id)}

    region_ids_in_slice = np.unique(im)
    region_ids_in_slice = region_ids_in_slice[region_ids_in_slice != 0]

    for rid in region_ids_in_slice:
        mask = (im == rid)
        contours = measure.find_contours(mask.astype(float), level=0.5)
        color = region_color_map.get(rid, (0.7, 0.7, 0.7))  # fallback 灰色

        for contour in contours:
            poly = Polygon(
                xy=np.stack([contour[:, 1], im.shape[0] - contour[:, 0]], axis=1),
                facecolor=color,
                edgecolor='black',
                linewidth=0.25,
                antialiased=True
            )
            ax.add_patch(poly)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    if save_path:
        fmt = save_path.split('.')[-1]
        fig.savefig(save_path, format=fmt, dpi=dpi,
                    bbox_inches='tight', pad_inches=0, transparent=True)
        print(f"Saved to {save_path}")

    return fig, ax

from iblatlas.atlas import AllenAtlas
from matplotlib import pyplot as plt
from skimage import measure
from matplotlib.patches import Polygon
import numpy as np

ba = AllenAtlas()
ap = 0.0  # bregma coronal

im = ba.slice(ap, axis=1, volume='annotation', mapping='Allen')
print("✅ shape:", im.shape, "dtype:", im.dtype)

# 使用上面我们写的函数
plot_annotation_slice_colored_allen(im, ba, save_path='allen_slice_colored.pdf')

# %%
import numpy as np
import nrrd
from iblatlas.atlas import AllenAtlas
from matplotlib.patches import Polygon
from skimage import measure
import matplotlib.pyplot as plt

def save_colored_coronal_slice(ap=0.0, save_path="allen_coronal_slice.pdf", dpi=600):
    """
    保存一张 Allen coronal 彩色区域图（矢量、透明背景、标准颜色）。
    参数：
        ap: float, AP 坐标（米），例如 0.0、-0.001、0.002
        save_path: str, 输出文件路径（支持 .pdf 或 .png）
        dpi: int, PNG 输出分辨率（对 PDF 无影响）
    """
    ba = AllenAtlas(res_um=25)
    annotation_data, _ = nrrd.read(ba._annotation_path)

    ap_index = ba.bc.y2i(ap)
    im = annotation_data[:, ap_index, :]  # coronal slice: (ML, DV)

    # 获取 Allen 标准颜色映射
    rgb_norm = ba.regions.rgb / 255.0
    region_color_map = {rid: tuple(rgb_norm[i]) for i, rid in enumerate(ba.regions.id)}

    fig, ax = plt.subplots(figsize=(15, 10))
    region_ids = np.unique(im)
    region_ids = region_ids[region_ids != 0]

    for rid in region_ids:
        mask = (im == rid)
        contours = measure.find_contours(mask.astype(float), level=0.5)
        color = region_color_map.get(rid, (0.7, 0.7, 0.7))  # fallback 灰色
        for contour in contours:
            poly = Polygon(
                xy=np.stack([contour[:, 1], im.shape[0] - contour[:, 0]], axis=1),
                facecolor=color,
                edgecolor='black',
                linewidth=0.25
            )
            ax.add_patch(poly)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    fmt = save_path.split('.')[-1]
    fig.savefig(save_path, format=fmt, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"✅ Saved: {save_path}")
save_colored_coronal_slice(ap=0.0, save_path="coronal_bregma.pdf")

# %%
import numpy as np
import nrrd
from iblatlas.atlas import AllenAtlas
from matplotlib.patches import Polygon
from skimage import measure
import matplotlib.pyplot as plt

# from brainbox.atlas import plot_atlas

# Coordinates of slices in mm
ML = -0.5
AP = 1
DV = -2

# Generate some mock data
# ba = atlas.AllenAtlas(25)
ba = AllenAtlas()
all_regions = ba.regions.acronym
regions = np.random.choice(all_regions, size=500, replace=False)  # pick 500 random regions
values = np.random.uniform(-1, 1, 500)  # generate 500 random values
f, axs = plt.subplots(1, 1, figsize=(20, 10))
plot_atlas(regions, values, ML, AP, DV, color_palette="RdBu_r", minmax=[-1, 1], axs=axs)
# %%
from iblatlas.plots import plot_scalar_on_slice
fig, ax = plt.subplots(figsize=(15, 10))
plot_scalar_on_slice(regions, values, coord=-1000, slice='coronal', mapping='Allen', hemisphere='both',
                         background='boundary', cmap='viridis', clevels=None, show_cbar=False, empty_color='silver',
                         brain_atlas=None, ax=None, vector=False, slice_files=None)
# %%
from iblatlas.plots import plot_swanson

fig = plot_swanson(ax=ax)
plt.show()
# %% plot Swanson fig with our ROIs
#https://int-brain-lab.github.io/iblenv/notebooks_external/atlas_swanson_flatmap.html
import numpy as np
from iblatlas.plots import plot_swanson_vector
from iblatlas.atlas import BrainRegions

br = BrainRegions()

# Plot Swanson map will default colors and acronyms
plot_swanson_vector(br=br, annotate=True, annotate_n=len(),
                        annotate_order='top', annotate_list=None, mask=None, mask_color='w', fontsize=4)


# %%
