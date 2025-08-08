#%%
from brainrender import Scene
from brainrender.actors import Slice

position_list = [-3111, -1983, 965, 1697]

for pos in position_list:
    ap = pos / 1e6  # 转换为米
    scene = Scene(atlas_name="allen_mouse_25um", title=f"ap_{pos}um")

    slice_actor = Slice(atlas=scene.atlas, plane='coronal', coord=ap)
    scene.add(slice_actor)

    scene.screenshot(name=f"coronal_ap_{pos}um", transparent=True)
    scene.close()  # 释放资源

# %%
from brainbox.atlas import plot_atlas
f, axs = plt.subplots(2, 3, figsize=(20, 10))
plot_atlas(regions, values, ML, AP, DV, color_palette="RdBu_r", minmax=[-1, 1], axs=axs[0])