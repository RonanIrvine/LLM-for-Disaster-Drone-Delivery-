from z3 import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from shapely.geometry import Point
import geopandas as gpd

# Parameters
n = 10  # Number of delivery nodes (excluding depot at 0)
K = 5   # Number of UAVs
T = 6   # Number of trips
N = n + 1  # Total nodes (including depot)
Q = 3   # Payload capacity
CAP = 5000  # Battery capacity
alpha = 50  # Energy consumption rate (units/km)
safety_threshold = 500  # 10% of CAP
max_nodes = 2  # Allow up to 2 delivery nodes per trip

# Generate coordinates for Upper Auckland
np.random.seed(42)
coords = {0: (174.75, -36.75)}  # Depot at approximate center (Albany, Auckland)
for i in range(1, N):
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-10, 10)
    lon = 174.65 + (x + 10) * 0.01  # Scale [-10, 10] to [174.65, 174.85]
    lat = -36.85 + (y + 10) * 0.01  # Scale [-10, 10] to [-36.85, -36.65]
    coords[i] = (lon, lat)

# Compute Haversine distances (in km)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

d = [[0.0 if i == j else haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
      for j in range(N)] for i in range(N)]

# Print max distance
max_distance = max(max(row) for row in d)
print(f"Max distance between any two nodes: {max_distance:.2f} km (energy cost: {alpha * max_distance:.2f})")

# Demands (max 10 units per node)
z = [5, 6, 7, 8, 9, 10, 9, 8, 7, 6]  # Total = 75

# Z3 Solver
s = Solver()

# Variables
x = [[[[Bool(f"x_{k}_{i}_{j}_{t}") for t in range(T)] for j in range(N)] for i in range(N)] for k in range(K)]
c = [[[Int(f"c_{k}_{i}_{t}") for t in range(T)] for i in range(N)] for k in range(K)]
v = [[[Bool(f"v_{k}_{i}_{t}") for t in range(T)] for i in range(N)] for k in range(K)]
b = [[[Real(f"b_{k}_{i}_{t}") for t in range(T)] for i in range(N)] for k in range(K)]
b_end = [[Real(f"b_end_{k}_{t}_0") for t in range(T)] for k in range(K)]

# Constraints
for k in range(K):
    for t in range(T):
        for i in range(N):
            s.add(x[k][i][i][t] == False)
        leave_depot = Sum([If(x[k][0][j][t], 1, 0) for j in range(1, N)])
        return_depot = Sum([If(x[k][j][0][t], 1, 0) for j in range(1, N)])
        s.add(leave_depot == return_depot)
        s.add(leave_depot <= 1)
        for i in range(1, N):
            incoming = Sum([If(x[k][j][i][t], 1, 0) for j in range(N)])
            outgoing = Sum([If(x[k][i][j][t], 1, 0) for j in range(N)])
            s.add(incoming == outgoing)
            s.add(v[k][i][t] == (incoming > 0))
        s.add(v[k][0][t] == (leave_depot > 0))
        s.add(Sum([If(v[k][i][t], 1, 0) for i in range(1, N)]) <= max_nodes)
        s.add(Sum([c[k][i][t] for i in range(1, N)]) <= Q)
        for i in range(N):
            s.add(c[k][i][t] >= 0)
            s.add(c[k][i][t] <= Q * If(v[k][i][t], 1, 0))
            if i != 0:
                s.add(c[k][i][t] >= If(v[k][i][t], 1, 0))
        s.add(b[k][0][t] == CAP)
        s.add(b_end[k][t] >= safety_threshold)
        s.add(b_end[k][t] <= CAP)
        for i in range(N):
            s.add(b[k][i][t] >= safety_threshold)
            s.add(b[k][i][t] <= CAP)
            for j in range(N):
                if i != j:
                    if j == 0:
                        s.add(Implies(x[k][i][j][t], b_end[k][t] == b[k][i][t] - alpha * d[i][j]))
                    else:
                        s.add(Implies(x[k][i][j][t], b[k][j][t] == b[k][i][t] - alpha * d[i][j]))
        for i in range(1, N):
            s.add(Implies(Not(v[k][i][t]), b[k][i][t] == CAP))
        s.add(Sum([If(x[k][i][j][t], alpha * d[i][j], 0) for i in range(N) for j in range(N)]) <= CAP)

for i in range(1, N):
    s.add(Sum([c[k][i][t] for k in range(K) for t in range(T)]) >= z[i-1])

if s.check() != sat:
    print("No feasible solution found.")
    exit()

# Process solution
m = s.model()
G = nx.DiGraph()
G.add_nodes_from(range(N))
pos = coords
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Track total deliveries and routes per UAV
total_deliveries = {i: 0 for i in range(1, N)}
uav_routes = {k+1: [] for k in range(K)}

# Create GeoDataFrame for nodes
points = [Point(coords[i]) for i in range(N)]
gdf = gpd.GeoDataFrame({'node': range(N)}, geometry=points, crs="EPSG:4326")

# Plot with Cartopy
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
# Use Stadia Maps (requires API key for high-volume use)
stamen_terrain = cimgt.StadiaMapsTiles(style='alidade_smooth', api_key='YOUR_API_KEY')  # Replace with your Stadia Maps API key
ax.add_image(stamen_terrain, 12)

for k in range(K):
    for t in range(T):
        edges = [(i, j) for i in range(N) for j in range(N) if m.evaluate(x[k][i][j][t], model_completion=True)]
        route = [0]
        current = 0
        visited = set([0])
        while True:
            next_node = None
            for i, j in edges:
                if i == current and j not in visited:
                    next_node = j
                    break
            if next_node is None:
                break
            route.append(next_node)
            visited.add(next_node)
            current = next_node
        if route[-1] != 0 and len(route) > 1:
            route.append(0)
            edges.append((route[-1], 0))
        route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
        G.add_edges_from(route_edges)
        
        deliveries = {}
        for i in range(N):
            val = m.evaluate(c[k][i][t], model_completion=True)
            if val is not None:
                val_num = val.as_long()
                if val_num > 0:
                    deliveries[i] = val_num
                    if i != 0:
                        total_deliveries[i] += val_num
        
        battery_levels = []
        for i in route:
            if i == 0 and len(battery_levels) == 0:
                val = m.evaluate(b[k][0][t], model_completion=True)
                label = f"Depot (start)"
            elif i == 0:
                val = m.evaluate(b_end[k][t], model_completion=True)
                label = f"Depot (return)"
            else:
                val = m.evaluate(b[k][i][t], model_completion=True)
                label = f"Node {i}"
            if val is not None:
                val_num = float(val.as_decimal(5).rstrip('?'))
                battery_levels.append((label, val_num / CAP * 100))
        
        total_distance = sum(d[route[i]][route[i+1]] for i in range(len(route)-1)) if len(route) > 1 else 0
        if route != [0]:
            print(f"UAV {k+1}, Trip {t+1}: Route = {route}, Deliveries = {deliveries}, Battery Levels (%) = {battery_levels}, Total Distance = {total_distance:.2f} km")
            uav_routes[k+1].append(f"T{t+1}: {route}")

        if route_edges:
            nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color=colors[k % len(colors)], width=2)

# Plot nodes
gdf[gdf['node'] == 0].plot(ax=ax, color='lightblue', markersize=100, label='Depot', transform=ccrs.PlateCarree())
gdf[gdf['node'] != 0].plot(ax=ax, color='lightgreen', markersize=50, label='Delivery Nodes', transform=ccrs.PlateCarree())

# Annotate nodes with battery levels
for k in range(K):
    for t in range(T):
        for i in range(N):
            if m.evaluate(v[k][i][t], model_completion=True):
                val = m.evaluate(b[k][i][t] if i != 0 else b_end[k][t], model_completion=True)
                if val is not None:
                    val_num = float(val.as_decimal(2).rstrip('?'))
                    percentage = (val_num / CAP) * 100
                    ax.text(coords[i][0], coords[i][1] + 0.005 + 0.002 * t, f'B{k+1}T{t+1}:{percentage:.1f}%', fontsize=8, color=colors[k % len(colors)], transform=ccrs.PlateCarree())

# Plot node labels using plt.text
for i in range(N):
    ax.text(coords[i][0], coords[i][1] - 0.005, str(i), fontsize=10, color='black', transform=ccrs.PlateCarree())

# Side legend
legend_handles = [mpatches.Patch(color=colors[k % len(colors)], label=f'UAV {k+1}') for k in range(K)]
plt.legend(handles=legend_handles, title="UAV Colors", loc='center right', bbox_to_anchor=(1.15, 0.5))

# Textbox for routes
textbox_content = "\n".join([f"UAV {k}: " + ", ".join(routes) for k, routes in uav_routes.items() if routes])
x_vals = [x for x, _ in coords.values()]
y_vals = [y for _, y in coords.values()]
min_x, max_x = min(x_vals), max(x_vals)
min_y, max_y = min(y_vals), max(y_vals)
plt.text(min_x, min_y - 0.02, textbox_content, fontsize=8,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), verticalalignment='top', transform=ccrs.PlateCarree())

# Set plot limits
ax.set_xlim(min_x - 0.01, max_x + 0.01)
ax.set_ylim(min_y - 0.03, max_y + 0.01)

# Add gridlines
ax.gridlines(draw_labels=True, linestyle='--')

# Print total deliveries
print("\nTotal Deliveries per Node:")
for i in range(1, N):
    print(f"Node {i}: {total_deliveries[i]} units (Required: {z[i-1]})")

plt.title("UAV Routes over Upper Auckland with Battery Levels (%)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()