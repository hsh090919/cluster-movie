import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 페이지 설정 (브라우저 탭 제목 및 아이콘 설정)
st.set_page_config(
    page_title="영화 유형 나누기 (K-Means)",
    page_icon="🎬",
    layout="wide",
)

# 헤더 영역
st.title("🎬 영화 유형 나누기")
st.caption("KOBIS 영화 데이터를 활용한 K-Means 클러스터링 분석 웹 앱")

# 1. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    total_count = len(df)

    # 필수 데이터 및 조건 필터링 (결측치 제거 & 첫 주 관객 > 0)
    required_cols = [
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi",
        "movieNm",
    ]
    df = df.dropna(subset=required_cols).copy()
    df = df[df["first_week_audi"] > 0].copy()

    # 파생 변수 및 속성 생성
    df["log_scrn"] = np.log10(df["first_scrn"])  # 스크린 수 (상용로그)
    df["log_total_audi"] = np.log10(df["total_audi"])  # 누적 관객 (상용로그)
    df["long_run"] = (df["total_audi"] / df["first_week_audi"]).clip(
        upper=20
    )  # 롱런 지수 (최대 20)

    filtered_count = len(df)
    return df, total_count, filtered_count


df, total_count, filtered_count = load_and_preprocess_data()

# 전체 및 분석 편수 안내
st.info(
    f"📊 전체 **{total_count:,}**편 중 조건(결측치 없음 & 첫 주 관객 0명 초과)을 만족하는 **{filtered_count:,}**편의 영화를 분석합니다."
)
st.markdown("---")

# 2. 분석 속성 선택
st.subheader("1. 군집화 속성 선택")

feature_map = {
    "스크린 수 (로그)": "log_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run",
}

default_features = list(feature_map.keys())
selected_feature_labels = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=default_features,
    default=default_features,
)

if len(selected_feature_labels) < 2:
    st.warning("⚠️ 군집화를 위해 속성을 최소 2개 이상 선택해주세요.")
    st.stop()

selected_features = [feature_map[lbl] for lbl in selected_feature_labels]

# 3. K-Means 군집화 수행
# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

# K-Means (K=3, 난수 고정)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객(total_audi) 평균이 높은 순으로 ㉮, ㉯, ㉰ 군집 이름 부여
cluster_audi_mean = (
    df.groupby("cluster_raw")["total_audi"].mean().sort_values(ascending=False)
)
cluster_mapping = {}
labels = ["㉮", "㉯", "㉰"]

for i, raw_cluster_id in enumerate(cluster_audi_mean.index):
    cluster_mapping[raw_cluster_id] = labels[i]

df["cluster"] = df["cluster_raw"].map(cluster_mapping)

# 4. 시각화 (Plotly)
st.markdown("---")
st.subheader("2. 2차원 산점도")

col2d_1, col2d_2 = st.columns(2)
with col2d_1:
    x_2d_label = st.selectbox(
        "가로축 (X축) 속성:",
        options=selected_feature_labels,
        index=0,
        key="x_2d",
    )
with col2d_2:
    y_2d_label = st.selectbox(
        "세로축 (Y축) 속성:",
        options=selected_feature_labels,
        index=min(1, len(selected_feature_labels) - 1),
        key="y_2d",
    )

fig_2d = px.scatter(
    df,
    x=feature_map[x_2d_label],
    y=feature_map[y_2d_label],
    color="cluster",
    category_orders={"cluster": ["㉮", "㉯", "㉰"]},
    hover_name="movieNm",
    labels={
        feature_map[x_2d_label]: x_2d_label,
        feature_map[y_2d_label]: y_2d_label,
        "cluster": "군집",
    },
    title=f"2D 산점도: {x_2d_label} vs {y_2d_label}",
)
fig_2d.update_layout(height=500)
st.plotly_chart(fig_2d, use_container_width=True)

# 3차원 산점도
st.markdown("---")
st.subheader("3. 3차원 산점도")

if len(selected_feature_labels) < 3:
    st.info("💡 3차원 산점도를 표시하려면 군집화 속성을 3개 이상 선택해주세요.")
else:
    col3d_1, col3d_2, col3d_3 = st.columns(3)
    with col3d_1:
        x_3d_label = st.selectbox(
            "X축 속성:", options=selected_feature_labels, index=0, key="x_3d"
        )
    with col3d_2:
        y_3d_label = st.selectbox(
            "Y축 속성:", options=selected_feature_labels, index=1, key="y_3d"
        )
    with col3d_3:
        z_3d_label = st.selectbox(
            "Z축 속성:", options=selected_feature_labels, index=2, key="z_3d"
        )

    fig_3d = px.scatter_3d(
        df,
        x=feature_map[x_3d_label],
        y=feature_map[y_3d_label],
        z=feature_map[z_3d_label],
        color="cluster",
        category_orders={"cluster": ["㉮", "㉯", "㉰"]},
        hover_name="movieNm",
        labels={
            feature_map[x_3d_label]: x_3d_label,
            feature_map[y_3d_label]: y_3d_label,
            feature_map[z_3d_label]: z_3d_label,
            "cluster": "군집",
        },
        title=f"3D 산점도: {x_3d_label} x {y_3d_label} x {z_3d_label}",
    )
    fig_3d.update_traces(marker=dict(size=3))
    fig_3d.update_layout(height=650)
    st.plotly_chart(fig_3d, use_container_width=True)

# 5. 통계 및 결과 출력
st.markdown("---")
st.subheader("4. 묶음별 통계 및 주요 영화")

# 묶음별 편수 및 네 속성의 원래 단위 평균 표 생성
summary_data = []

for cluster_label in ["㉮", "㉯", "㉰"]:
    c_df = df[df["cluster"] == cluster_label]
    summary_data.append(
        {
            "묶음": cluster_label,
            "편수": f"{len(c_df):,}편",
            "스크린 수 평균": round(c_df["first_scrn"].mean(), 1),
            "누적 관객 수 평균": round(c_df["total_audi"].mean(), 1),
            "10위권 일수 평균": round(c_df["days_in_top10"].mean(), 1),
            "롱런 지수 평균": round(c_df["long_run"].mean(), 2),
        }
    )

summary_df = pd.DataFrame(summary_data)
st.markdown("#### 📌 묶음별 속성 평균 (원래 단위)")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown("#### 🏆 묶음별 누적 관객 Top 5 영화")
cols = st.columns(3)

for idx, cluster_label in enumerate(["㉮", "㉯", "㉰"]):
    with cols[idx]:
        st.markdown(f"##### **묶음 {cluster_label}**")
        top5 = df[df["cluster"] == cluster_label].nlargest(5, "total_audi")[
            ["movieNm", "total_audi"]
        ]

        top5_list = []
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            top5_list.append(
                f"{rank}. **{row['movieNm']}** ({row['total_audi']:,}명)"
            )

        st.markdown("\n".join(top5_list))
