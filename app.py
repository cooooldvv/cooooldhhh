import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
import time

# ================= 1. 页面配置 =================
st.set_page_config(layout="wide", page_title="城市房价与租房智能分析看板")

st.markdown("""
<style>
    .stSidebar { background-color: #f0f2f6; }
    .main-header { font-size: 36px; font-weight: bold; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-header">🏙️ 城市房价与租房市场智能分析看板</div>', unsafe_allow_html=True)


# ================= 2. 双保险数据引擎 (实时爬虫 + 完美容错回退) =================
@st.cache_data(ttl=3600)  # 设置缓存1小时，避免每次筛选都重新爬取
def load_and_process_data():
    st.info("🔄 系统正在爬取链家 / 贝壳今日真实房源数据，请稍候... (约需 5-10 秒)")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    data_list = []

    # ============= 真实爬虫逻辑 =============
    try:
        # 为了稳定且不过度触发反爬，我们同时抓取“二手房”和“租房”频道
        # 北京链家二手房真实数据 (抓取前2页展示)
        for page in range(1, 3):
            url = f'https://bj.lianjia.com/ershoufang/pg{page}/'
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            house_list = soup.find_all('div', class_='info clear')
            for house in house_list:
                try:
                    # 解析总价、面积、单价
                    total_price_str = house.find('div', class_='totalPrice').span.text
                    total_price = float(total_price_str)
                    unit_price_str = house.find('div', class_='unitPrice').span.text
                    unit_price = float(unit_price_str.replace('万', '').replace('元/平米', ''))
                    house_info = house.find('div', class_='houseInfo').get_text()
                    area_str = house_info.split('|')[1].replace('平米', '')
                    area = float(area_str)
                    # 取行政区（链家详情页小区名字下方有区域）
                    position_info = house.find('div', class_='positionInfo').get_text()
                    districts = position_info.split('-')
                    district = districts[1].strip() if len(districts) > 1 else "未知"

                    data_list.append({
                        'City': '北京', 'District': district, 'Area': area,
                        'Total_Price': total_price, 'Unit_Price': unit_price,
                        'Rent_Price': 0  # 二手房没有租金，后面补
                    })
                except:
                    continue
            time.sleep(0.5)  # 防止访问太快被封锁

        # 北京链家租房真实数据 (抓取前2页)
        for page in range(1, 3):
            url = f'https://bj.lianjia.com/zufang/pg{page}/'
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            house_list = soup.find_all('div', class_='info-panel')
            for house in house_list:
                try:
                    # 解析租房租金和面积
                    rent_price_str = house.find('span', class_='content').text.replace('元/月', '')
                    rent_price = float(rent_price_str)
                    house_info = house.find('div', class_='content').find_all('p')[1].get_text()
                    area_str = house_info.split('|')[2].replace('㎡', '')
                    area = float(area_str)
                    # 租房页面提取城市和行政区
                    region_info = house.find('div', class_='content').find_all('p')[0].get_text()
                    district = region_info.split('-')[1].strip() if '-' in region_info else "未知"

                    data_list.append({
                        'City': '北京', 'District': district, 'Area': area,
                        'Total_Price': 0, 'Unit_Price': 0,  # 租房没有房价
                        'Rent_Price': rent_price
                    })
                except:
                    continue
            time.sleep(0.5)

    except Exception as e:
        st.warning(
            f"⚠️ 由于网络受阻或网站反爬策略限制，实时抓取失败 (报错: {e})。系统已自动切换至【本地仿真数据】以保证界面正常。")

    # 将爬取到的数据转为 DataFrame
    if len(data_list) > 0:
        df = pd.DataFrame(data_list)
        df.dropna(inplace=True)
        # 把二手房和租房合并
    else:
        # ============= 备用回退逻辑：生成真实模拟数据 =============
        np.random.seed(42)
        n_samples = 20000
        cities = ['北京', '上海', '广州', '深圳', '杭州', '成都']
        districts = {'北京': ['朝阳区', '海淀区', '丰台区', '西城区', '通州区'],
                     '上海': ['浦东新区', '静安区', '黄浦区', '闵行区', '松江区'],
                     '广州': ['天河区', '海珠区', '越秀区', '番禺区', '白云区'],
                     '深圳': ['南山区', '福田区', '罗湖区', '宝安区', '龙岗区'],
                     '杭州': ['西湖区', '拱墅区', '滨江区', '余杭区', '上城区'],
                     '成都': ['锦江区', '青羊区', '武侯区', '成华区', '高新区']}
        data = []
        for _ in range(n_samples):
            city = np.random.choice(cities)
            district = np.random.choice(districts[city])
            area = np.random.uniform(40, 180)
            base_price = 40000 if city in ['北京', '上海', '深圳'] else (25000 if city in ['广州', '杭州'] else 15000)
            base_rent = 80 if city in ['北京', '上海', '深圳'] else (60 if city in ['广州', '杭州'] else 40)
            price_factor = np.random.uniform(0.5, 1.8)
            unit_price = base_price * price_factor + np.random.normal(0, 5000)
            total_price = (unit_price * area) / 10000
            rent_price_per_m2 = base_rent * price_factor + np.random.normal(0, 10)
            rent_price = rent_price_per_m2 * area
            data.append(
                {'City': city, 'District': district, 'Area': round(area, 1), 'Total_Price': round(total_price, 2),
                 'Unit_Price': round(unit_price, 2), 'Rent_Price': round(rent_price, 2)})
        df = pd.DataFrame(data)

    # 统一清洗与去重
    df = df[(df['Unit_Price'] > 5000) | (df['Rent_Price'] > 0)]  # 过滤极端异常值
    df.dropna(inplace=True)

    # ===== K-Means 聚类分析 =====
    # 因为只有单类数据可能会存有 0 值，为了防止警告，我们将用于聚类的两列单独处理
    cluster_df = df[(df['Total_Price'] > 0) & (df['Area'] > 0)].copy()
    if not cluster_df.empty:
        X = cluster_df[['Total_Price', 'Area']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_df['Cluster'] = kmeans.fit_predict(X_scaled)
        # 自动命名聚类标签
        cluster_means = cluster_df.groupby('Cluster')[['Total_Price', 'Area']].mean().sort_values(by='Total_Price')
        cluster_order = cluster_means.index.tolist()
        cluster_mapping = {cluster_order[0]: '低总价小户型（过渡型）', cluster_order[1]: '中等总价中户型（刚需型）',
                           cluster_order[2]: '高总价大户型（改善型）'}
        cluster_df['House_Type'] = cluster_df['Cluster'].map(cluster_mapping)
        # 将聚类结果合并回原数据
        df = df.merge(cluster_df[['Area', 'Total_Price', 'House_Type']], on=['Area', 'Total_Price'], how='left')
        df['House_Type'] = df['House_Type'].fillna('未分类房源')
    else:
        df['House_Type'] = '未分类房源'

    return df


# 获取数据
df = load_and_process_data()

# ================= 3. 侧边栏交互筛选 =================
st.sidebar.header("🔍 数据筛选面板")
selected_cities = st.sidebar.multiselect("选择分析城市：", ["全部"] + list(df['City'].unique()), default=["全部"])
price_range = st.sidebar.slider("选择房屋总价预算 (万元)：", float(df['Total_Price'].min()),
                                float(df['Total_Price'].max()), (100.0, 800.0)) if df['Total_Price'].max() > 0 else (
100.0, 800.0)
area_range = st.sidebar.slider("选择房屋面积 (m²)：", float(df['Area'].min()), float(df['Area'].max()), (50.0, 140.0))

# 筛选逻辑
filtered_df = df.copy()
if "全部" not in selected_cities and len(selected_cities) > 0:
    filtered_df = filtered_df[filtered_df['City'].isin(selected_cities)]
filtered_df = filtered_df[(filtered_df['Area'] >= area_range[0]) & (filtered_df['Area'] <= area_range[1])]
# 房价和房租双端处理
if df['Total_Price'].max() > 0:
    filtered_df = filtered_df[
        (filtered_df['Total_Price'] >= price_range[0]) & (filtered_df['Total_Price'] <= price_range[1])]
else:
    filtered_df = filtered_df[
        (filtered_df['Rent_Price'] >= price_range[0] * 10) & (filtered_df['Rent_Price'] <= price_range[1] * 10)]

st.sidebar.info(f"✅ 当前筛选出 **{len(filtered_df)}** 套房源数据")

# ================= 4. KPI 核心指标看板 =================
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric(label="🏠 最高单价", value=f"{filtered_df['Unit_Price'].max():.0f} 元/m²" if filtered_df[
                                                                                                     'Unit_Price'].max() > 0 else "暂无数据")
with col2: st.metric(label="📊 平均单价", value=f"{filtered_df['Unit_Price'].mean():.0f} 元/m²" if filtered_df[
                                                                                                      'Unit_Price'].mean() > 0 else "暂无数据")
with col3: st.metric(label="📐 平均面积", value=f"{filtered_df['Area'].mean():.1f} m²")
with col4: st.metric(label="💰 平均月租金", value=f"{filtered_df['Rent_Price'].mean():.0f} 元" if filtered_df[
                                                                                                     'Rent_Price'].mean() > 0 else "暂无数据")
with col5: st.metric(label="🏘️ 房源总量", value=f"{len(filtered_df)} 套")

st.markdown("---")

# ================= 5. 可视化图表（第一行） =================
if not filtered_df.empty:
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if filtered_df['Unit_Price'].max() > 0:
            fig_box = px.box(filtered_df, x='City', y='Unit_Price', title="📊 各城市房屋单价分布对比（箱线图）",
                             labels={'City': '城市', 'Unit_Price': '单价 (元/m²)'}, color='City')
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("当前数据以租房为主，暂无足够房源单价数据生成箱线图。")

    with row1_col2:
        if filtered_df['Unit_Price'].max() > 0:
            fig_hist = px.histogram(filtered_df, x='Unit_Price', nbins=50, title="🏦 房源单价密集度分布图",
                                    labels={'Unit_Price': '单价 (元/m²)'}, color_discrete_sequence=['#3366cc'])
            fig_hist.update_layout(xaxis_title="单价 (元/m²)", yaxis_title="房源数量")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            fig_hist = px.histogram(filtered_df, x='Rent_Price', nbins=50, title="🏦 租房价格密集度分布图",
                                    labels={'Rent_Price': '月租金 (元)'}, color_discrete_sequence=['#3366cc'])
            fig_hist.update_layout(xaxis_title="月租金 (元)", yaxis_title="房源数量")
            st.plotly_chart(fig_hist, use_container_width=True)

    # ================= 6. 可视化图表（第二行） =================
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if filtered_df['Total_Price'].max() > 0:
            fig_scatter = px.scatter(filtered_df, x='Area', y='Total_Price', color='House_Type',
                                     title="🎯 房屋总价与面积聚类分布 (K-Means)",
                                     labels={'Area': '面积 (m²)', 'Total_Price': '总价 (万元)'},
                                     color_discrete_map={'高总价大户型（改善型）': '#ef553b',
                                                         '中等总价中户型（刚需型）': '#00cc96',
                                                         '低总价小户型（过渡型）': '#ab63fa'})
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            fig_scatter = px.scatter(filtered_df, x='Area', y='Rent_Price', color='House_Type',
                                     title="🎯 租房面积与租金聚类分布 (K-Means)",
                                     labels={'Area': '面积 (m²)', 'Rent_Price': '月租金 (元)'},
                                     color_discrete_map={'未分类房源': '#636efa'})
            st.plotly_chart(fig_scatter, use_container_width=True)

    with row2_col2:
        type_counts = filtered_df['House_Type'].value_counts().reset_index()
        type_counts.columns = ['House_Type', 'Count']
        fig_pie = px.pie(type_counts, values='Count', names='House_Type', title="🍩 房屋市场价值分层占比图",
                         color='House_Type',
                         color_discrete_map={'高总价大户型（改善型）': '#ef553b', '中等总价中户型（刚需型）': '#00cc96',
                                             '低总价小户型（过渡型）': '#ab63fa'})
        fig_pie.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ================= 7. 可视化图表（第三行：租房与详情） =================
    st.subheader("🔥 租房市场热力与详细数据视图")
    row3_col1, row3_col2 = st.columns([2, 1])
    with row3_col1:
        # 热力图：各区租金热力
        fig_heat = px.density_heatmap(filtered_df, x='District', y='Rent_Price', title="各地区租房价格热力图",
                                      labels={'Rent_Price': '月租金 (元)', 'District': '行政区'},
                                      color_continuous_scale='Viridis')
        st.plotly_chart(fig_heat, use_container_width=True)

    with row3_col2:
        st.markdown("#### 📋 数据明细预览")
        with st.expander("点击展开/收起详细数据表"):
            st.dataframe(
                filtered_df[['City', 'District', 'Area', 'Total_Price', 'Unit_Price', 'Rent_Price', 'House_Type']].head(
                    50), use_container_width=True)
else:
    st.warning("当前筛选条件下无匹配房源，请调整侧边栏筛选器。")

st.markdown("---")
st.caption("💡 提示：调整左侧面板的【城市、预算、面积】筛选器，所有图表将立刻自动联动更新。")