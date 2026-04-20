import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import itertools

# ==========================================
# 1. 页面基本配置
# ==========================================
st.set_page_config(page_title="2034 Network Optimization", layout="wide")

st.title("🇭🇺2024-2034 Hungary's electronic waste recycling network optimization.Dynamic addressing simulator based on P-Graph core algorithm.")
st.markdown("The evolution of national recycling hubs was observed by adjusting fixed costs and transportation unit prices.")

# =========================================
# 2. 侧边栏：核心参数控制 (UI 调整)
# =========================================
st.sidebar.header("⚙️ Economic Parameters")

# 1. Fixed Cost (对应原 Government Subsidy 位置)
fixed_cost = st.sidebar.slider("Fixed Cost (HUF)", 
                               min_value=5000000, 
                               max_value=50000000, 
                               value=15000000, 
                               step=1000000)

# 2. Unit Trans Cost (对应原 Logistics 位置)
unit_trans_cost = st.sidebar.slider("Unit Trans Cost (HUF/t*km)", 
                                    min_value=50, 
                                    max_value=400, 
                                    value=226)

st.sidebar.divider()

# 3. 年份滑块
target_year = st.sidebar.slider("Projection Year", 2024, 2034, 2034)

# 4. 枢纽数量 K
num_hubs = st.sidebar.select_slider("Fixed K-Hubs (Number of Centers)", 
                                    options=[1, 2, 3, 4, 5, 6, 7, 8], 
                                    value=4)

# ==========================================
# 3. 动态增长与数据准备
# ==========================================
@st.cache_data
def get_original_data():
    data = {
        'City': ['Budapest', 'Gyor', 'Miskolc', 'Szeged', 'Pecs', 'Debrecen', 'Dunakeszi', 'Kecskemet', 'Szombathely', 'Szekesfehervar'],
        'Lat': [47.49, 47.68, 48.10, 46.25, 46.07, 47.53, 47.63, 46.90, 47.23, 47.19],
        'Lon': [19.04, 17.63, 20.78, 20.14, 18.23, 21.62, 19.13, 19.69, 16.62, 18.41],
        'Waste_2024': [45000, 18000, 21000, 12000, 9500, 15000, 8000, 11000, 9000, 7500]
    }
    return pd.DataFrame(data)

df_base = get_original_data()
growth_rate = 0.0385  # 模拟年增长率
df = df_base.copy()
df['Current_Waste'] = df['Waste_2024'] * ((1 + growth_rate) ** (target_year - 2024))

# ==========================================
# 4. P-Median 核心优化算法
# ==========================================
def solve_p_median(df_in, k, f_cost, t_unit_cost):
    sources = df_in[['Lat', 'Lon']].values
    dist_matrix = cdist(sources, sources) * 111 # 粗略公里转换
    potential_indices = list(range(len(df_in)))
    
    best_total_cost = float('inf')
    best_combo = []
    
    # 遍历 K 个枢纽的所有组合
    for combo in itertools.combinations(potential_indices, k):
        min_dists = np.min(dist_matrix[:, combo], axis=1)
        total_t_cost = np.sum(min_dists * df_in['Current_Waste'] * t_unit_cost)
        total_f_cost = k * f_cost
        
        if (total_t_cost + total_f_cost) < best_total_cost:
            best_total_cost = total_t_cost + total_f_cost
            best_combo = combo
            
    return df_in.iloc[list(best_combo)], best_total_cost, best_combo

# 执行计算
selected_hubs, final_cost, best_idx_list = solve_p_median(df, num_hubs, fixed_cost, unit_trans_cost)

# ==========================================
# 5. UI 渲染与绘图
# ==========================================
c_plot, c_stats = st.columns([3, 1])

with c_plot:
    st.subheader(f"{target_year} Optimized Recycling Network (K={num_hubs})")
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.grid(True, color='#F0F0F0', alpha=0.5, zorder=0)
    
    # 计算分配连线
    sources_coords = df[['Lat', 'Lon']].values
    hub_coords = selected_hubs[['Lat', 'Lon']].values
    assignments = np.argmin(cdist(sources_coords, hub_coords), axis=1)

    # 绘制分配线
    for i in range(len(df)):
        hub_row = selected_hubs.iloc[assignments[i]]
        ax.plot([df.iloc[i]['Lon'], hub_row['Lon']], 
                [df.iloc[i]['Lat'], hub_row['Lat']], 
                color='gray', linestyle='-', linewidth=0.7, alpha=0.3, zorder=1)

    # 绘制普通节点
    ax.scatter(df['Lon'], df['Lat'], s=df['Current_Waste']*0.06, 
               c='lightgray', alpha=0.5, label='Source Nodes', zorder=2)

    # 绘制枢纽 (多色星形)
    colors = plt.cm.Set1(np.linspace(0, 1, len(selected_hubs)))
    for i, (idx, color) in enumerate(zip(best_idx_list, colors)):
        hub_row = df.iloc[idx]
        ax.scatter(hub_row['Lon'], hub_row['Lat'], s=600, 
                   color=color, marker='*', edgecolors='black', label=f"Hub: {hub_row['City']}", zorder=5)
        ax.text(hub_row['Lon']+0.12, hub_row['Lat']+0.08, 
                hub_row['City'], fontsize=12, fontweight='bold', zorder=6)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='lower left', frameon=True)
    st.pyplot(fig)

with c_stats:
    st.write("### 📊 System Summary")
    st.metric("Forecast year", f"{target_year}")
    st.metric("Total estimated waste", f"{int(df['Current_Waste'].sum()):,} t")
    st.metric("Total cost (M HUF)", f"{final_cost/1e6:.2f}")
    
    st.divider()
    st.write("### 📍 Selected hubs:")
    hub_names = selected_hubs['City'].tolist()
    for name in hub_names:
        if name == "Szeged":
            st.success(f"🌟 **{name}**")
        else:
            st.info(f"📍 {name}")
