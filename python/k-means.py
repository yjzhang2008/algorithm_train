'''
# k-means
案例演示：已知城市的坐标，在合适的位置布局K个能源中心
'''

from matplotlib import  pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict
import random
import re
import math
#import pandas as pd
from matplotlib import font_manager



# 城市坐标信息
coordination_source = """
{name:'兰州', geoCoord:[103.73, 36.03]},
{name:'嘉峪关', geoCoord:[98.17, 39.47]},
{name:'西宁', geoCoord:[101.74, 36.56]},
{name:'成都', geoCoord:[104.06, 30.67]},
{name:'石家庄', geoCoord:[114.48, 38.03]},
{name:'拉萨', geoCoord:[102.73, 25.04]},
{name:'贵阳', geoCoord:[106.71, 26.57]},
{name:'武汉', geoCoord:[114.31, 30.52]},
{name:'郑州', geoCoord:[113.65, 34.76]},
{name:'济南', geoCoord:[117, 36.65]},
{name:'南京', geoCoord:[118.78, 32.04]},
{name:'合肥', geoCoord:[117.27, 31.86]},
{name:'杭州', geoCoord:[120.19, 30.26]},
{name:'南昌', geoCoord:[115.89, 28.68]},
{name:'福州', geoCoord:[119.3, 26.08]},
{name:'广州', geoCoord:[113.23, 23.16]},
{name:'长沙', geoCoord:[113, 28.21]},
{name:'海口', geoCoord:[110.35, 20.02]},
{name:'沈阳', geoCoord:[123.38, 41.8]},
{name:'长春', geoCoord:[125.35, 43.88]},
{name:'哈尔滨', geoCoord:[126.63, 45.75]},
{name:'太原', geoCoord:[112.53, 37.87]},
{name:'西安', geoCoord:[108.95, 34.27]},
{name:'台湾', geoCoord:[121.30, 25.03]},
{name:'北京', geoCoord:[116.46, 39.92]},
{name:'上海', geoCoord:[121.48, 31.22]},
{name:'重庆', geoCoord:[106.54, 29.59]},
{name:'天津', geoCoord:[117.2, 39.13]},
{name:'呼和浩特', geoCoord:[111.65, 40.82]},
{name:'南宁', geoCoord:[108.33, 22.84]},
{name:'西藏', geoCoord:[91.11, 29.97]},
{name:'银川', geoCoord:[106.27, 38.47]},
{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
{name:'香港', geoCoord:[114.17, 22.28]},
{name:'澳门', geoCoord:[113.54, 22.19]}
"""

# 数据处理
def get_city_location(coordination_source):
    city_location = {}
    for line in coordination_source.split('\n'):
        if not line:
            continue
        pattern = re.compile('{name:\'(\w+)\'\,\s*geoCoord\:\[(\d+\.?\d*)\,\s+(\d+\.?\d*)\]')
        city_info = re.findall(pattern, line)
        #print(city_info)
        if not city_info:
            continue
        name, longitude, lantitude = city_info[0]
        city_location[name] = (float(longitude), float(lantitude))
    return city_location

def get_center_location(all_x, all_y):
    x = random.uniform(min(all_x), max(all_x))
    y = random.uniform(min(all_y), max(all_y))
    return x, y


def get_geo_distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lon1, lat1 = origin
    lon2, lat2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def update_center_location(center_location, centers, update_completed, threshold):
    for c in center_location:
        if update_completed[c]:
            continue
        neightbors = centers[c]
        if neightbors:
            # 根据离中心点最近的城市，算出平均值作为备用中心点
            neightbor_center = np.mean(neightbors, axis=0)
            #print(neightbors)
            #print(center_location[c], neightbor_center)
            # 如果备用中心点与原中心点的距离大于阈值，则把备用中心点作为新的中心点，否侧该中心点的搜索到此结束
            if get_geo_distance(center_location[c], neightbor_center) > threshold:
                center_location[c] = neightbor_center
            else:
                update_completed[c] = True
        else:
            print('Warning:center[{}]:{} have no neightbors!'.format(c, center_location[c]))
            update_completed[c] = True
    return center_location, update_completed

def kmeans(city_location, K, threshold=5):
    # 初始中心坐标取得
    center_location = {}
    all_x = []
    all_y = []
    for x, y in city_location.values():
        all_x.append(x)
        all_y.append(y)
    for i in range(K):
        center_location[i+1] = get_center_location(all_x, all_y)

    # 中心坐标更新结束标志
    update_completed = {}
    for i in center_location.keys():
        update_completed[i] = False

    while list(update_completed.values()).count(False) > 0:
        centers = defaultdict(list)
        # 遍历所有城市坐标，把它放进离它最近的中心list中去
        for x, y in zip(all_x, all_y):
            # 找出离该城市最近的中心点
            center_k, center_d = min([(i, get_geo_distance((x, y), center_location[i])) for i in center_location.keys()],  key=lambda t:t[1])
            # 把该城市归入找到的最近中心点的列表中去
            centers[center_k].append((x, y))

        # 更新中心点的坐标
        center_location, update_completed = update_center_location(center_location, centers, update_completed, threshold)
        #print('--', update_completed.values(), list(update_completed.values()).count(False))
    return center_location

def draw_cities(city_location, color=None):
    city_graph = nx.Graph()
    city_graph.add_nodes_from(city_location)
    nx.draw(city_graph, city_location, node_color=color, node_size=30, with_labels=True, font_size=14)

if __name__ == '__main__':
    # 城市坐标字典取得
    city_location = get_city_location(coordination_source)
    print(city_location)

    # k-means
    center_location = kmeans(city_location, 5)
    print(center_location)
    center_location = {'能源站-{}'.format(k): x for k, x in center_location.items()}
    print(center_location)

    # 图表展示
    plt.rcParams['font.sans-serif'] = ['PingFang']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.figure(figsize=(18, 10), dpi=60)
    draw_cities(city_location, 'green')
    draw_cities(center_location, 'red')
    plt.show()


