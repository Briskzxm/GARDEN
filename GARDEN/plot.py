import random
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
import os

def edge_check_vaild(e,tree,r,err):
    xp = e[0]
    xq = e[1]
    L = np.sqrt(np.dot(xq-xp,xq-xp))
    if L > 2*r:
        return False, -1
    vec = (xq-xp)/L# the vector from p to q
    normal = np.array([vec[1], -vec[0]])
    c1 = (xp + xq) / 2 + normal * np.sqrt(r**2-(L/2)**2)
    c2 = (xp + xq) / 2 - normal * np.sqrt(r**2-(L/2)**2)
    c = np.array([c1, c2])
    count = tree.query_radius(c, r=r+err, return_distance=False, count_only=True, sort_results=False)
    if count[0] <= 2:
        return True, c[0]
    elif count[1] <= 2:
        return True, c[1]
    else:
        return False, -1
    
def boundary_extract(points, alpha, err=10e-3):
    """
    Here, parameter err was place, because there are errors when calculating distance
    meanwhile, this err was different for different scaling 2D point cloud
    so, a parameter was placed here to considering the calculation errors
    """
    R = 1 / alpha
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    tri = Delaunay(pts)
    s = tri.simplices
    N = s.shape[0]
    i = 0
    edges = []
    centers = []
    while i <= N - 1:
        if s[i, 0] == -1:
            i = i + 1
            continue
        p3 = s[i]
        e1 = np.array([points[p3[0], :], points[p3[1], :]])
        e2 = np.array([points[p3[1], :], points[p3[2], :]])
        e3 = np.array([points[p3[0], :], points[p3[2], :]])
        e = [e1, e2, e3]
        for j in range(3):
            flag, center = edge_check_vaild(e[j], tree, R, err)
            if flag:
                edges.append(e[j])
                centers.append(center)
        nb = tri.neighbors[i]
        nb_valid = nb[nb != -1]
        #nb_valid_num = nb_valid.shape[0]
        #s[nb_valid] = -1
        i = i + 1
    return edges, centers 

def Set_palette(clu_num):
    vega_10 = list(map(colors.to_hex, cm.tab10.colors))
    vega_10_scanpy = vega_10.copy()
    # vega_10_scanpy[3] = '#ad494a'
    # vega_10_scanpy[2] = '#279e68'  # green
    # vega_10_scanpy[4] = '#aa40fc'  # purple
    vega_10_scanpy[8] = '#b5bd61'  # kakhi

    vega_20 = list(map(colors.to_hex, cm.tab20.colors))

    # reorderd, some removed, some added
    vega_20_scanpy = [
        # dark without grey:
        *vega_20[0:14:2],
        *vega_20[16::2],
        # light without grey:
        *vega_20[1:15:2],
        *vega_20[17::2],
        # manual additions:
        '#ad494a',
        '#8c6d31',
    ]
    vega_20_scanpy[2] = vega_10_scanpy[2]
    vega_20_scanpy[4] = vega_10_scanpy[4]
    vega_20_scanpy[7] = vega_10_scanpy[8]  # kakhi shifted by missing grey

    # set palette
    if clu_num <= 10:
        palette = vega_10_scanpy
        cmap = 'tab10'
    else:
        # palette = vega_20_scanpy
        palette = 'tab20'
        cmap = 'tab20'
    return palette, cmap

def show_edge(edges, ax, label='SCSP', z=None, color='red', linewidth=1, alpha=0.8):
    import networkx as nx
    edf = np.array(edges)
    visit = pd.DataFrame(np.zeros(len(edf)), columns=['visited'], dtype=bool)
    edf1 = pd.DataFrame(edf[:,0], columns=['x1', 'y1'])
    edf2 = pd.DataFrame(edf[:, 1], columns=['x2', 'y2'])
    edfs = pd.concat([edf1, edf2, visit], axis=1)
    edfs = edfs.drop_duplicates()
    G = nx.Graph()
    edges0 = [((edfs.iloc[i, 0], edfs.iloc[i, 1]),
               (edfs.iloc[i, 2], edfs.iloc[i, 3])) for i in range(len(edfs))]
    G.add_edges_from(edges0)

    # draw the circle
    lines = nx.cycle_basis(G)
    for line in lines:
        x = [p[0] for p in line]# add the first vertex to generate circle
        x.append(x[0])
        y = [p[1] for p in line]
        y.append(y[0])
        if z is None:
            ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, linestyle='-', label=label)
        else:
            ax.plot(x, y, [z] * (len(x)), color=color, alpha=alpha, linewidth=linewidth, linestyle='-', label=label)
        label = None
        G.remove_nodes_from(line)

    # draw existing edges
    exist_edges = list(G.edges())
    for edge in exist_edges:
        x = [edge[0][0], edge[1][0]]
        y = [edge[0][1], edge[1][1]]
        if z is None:
            ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, linestyle='-', label=label)
        else:
            ax.plot(x, y, [z] * (len(x)), color=color, alpha=alpha, linewidth=linewidth, linestyle='-', label=label)
    ret_lines = []
    ret_lines.extend(lines)
    ret_lines.extend(exist_edges)
    return ret_lines

def retain_ones(bool_array, retain_percent=0.5):
            bool_array = list(bool_array)
            ones_count = bool_array.count(True)
            ones_to_retain = int(ones_count * retain_percent)
            ones_indices = [i for i, x in enumerate(bool_array) if x]
            random.shuffle(ones_indices)
            retained_indices = ones_indices[:ones_to_retain]
            for i in range(len(bool_array)):
                if i in retained_indices:
                    bool_array[i] = True
                else:
                    bool_array[i] = False
            return bool_array

def Show_3D_landscape(adata,palette=None,folder=os.getcwd(), cell_types=None, single_show = False ,alpha=100, save=False, name="landscape.png", legend=True):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    
    print("=========3DLandscape Plotting==========")

    coords = adata.obsm['spatial']
    x, y = coords[:, 0], coords[:, 1]
    
    w, h = int(np.max(x)) + 100, int(np.max(y)) + 100  # 使用最大值设置绘图尺寸
    
    # 检查细胞类型
    if 'domain' in adata.obs:
        idents = adata.obs['domain'].astype('category')
    else:
        raise ValueError("The adata object does not contain 'domain' in .obs.")
    
    full_ct = idents.cat.categories.tolist()
    print(f"Full Cell Types: {full_ct}")
    
    if cell_types is None:
        cell_types = full_ct
    else:
        cell_types = list(set(cell_types).intersection(full_ct))
    
    print(f"Cell Types: {cell_types}")
    
    # 绘制细胞类型的3D点
    fig = plt.figure(figsize=(16,8))  # Scaling down the values for figsize
    ax1 = plt.axes(projection='3d')
    if not palette :
        palette, cmap = Set_palette(len(full_ct))
    spot3D_layer = {}
    index = 0
    z = 1
    palette =  [  '#71c33a','#FF8C00', '#FFE4E1', '#00868B', '#808000','#1CE6FF', '#FF90C9', '#e14aec','#004EB0', '#8FB0FF','#BA0900']
    for cell_type in cell_types:
            color = palette[index]
            print(color)
            selection = idents == cell_type
            if(single_show):
                if cell_type in single_show:  
                    z = z + 1          
                    spot3D = pd.DataFrame({
                        'X': x[selection],
                        'Y': y[selection],
                        'Z': z  # 初始化Z轴
                    })
                    
                else:
                    spot3D = pd.DataFrame({
                        'X': x[selection],
                        'Y': y[selection],
                        'Z': 0  # 初始化Z轴
                    })

            else: 
                 z = z+ 1
                 spot3D = pd.DataFrame({
                        'X': x[selection],
                        'Y': y[selection],
                        'Z': z  # 初始化Z轴
                    })
            index = index +1
            pts = spot3D[['Y', 'X']].values
            edges, _ = boundary_extract(pts, alpha / np.max(pts))
            show_edge(edges, ax1, z=0, color=color, linewidth=2.5, alpha=1)

            #show_edge(edges, ax1, z=0, color=color, linewidth=2.5, alpha=1)
            ax1.scatter3D(spot3D['Y'], spot3D['X'], spot3D['Z'], color=color, label=cell_type, s=5, alpha=0.9)
            # 保存的细胞比例
            retain_percent = 0.05
            bool_array = [True] * len(spot3D)
            filtered_bool_array = retain_ones(bool_array, retain_percent=retain_percent)
            # 根据布尔数组过滤出要绘制的点
            spot3D_filtered = spot3D[filtered_bool_array]
            # 每个点添加垂直线到Z轴
            for _, row in spot3D_filtered.iterrows():
                ax1.plot([row['Y'], row['Y']], [row['X'], row['X']], [row['Z'], 0], color=color, alpha=0.8, linewidth=1.25)
           

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.axis('off')

    if save:
        figname = f"{folder}/{name}"
        print(f"Saving 3DLandscape to {figname}")
        fig.savefig(figname, dpi=400)
    
    print("=========3DLandscape Finished==========")
    return