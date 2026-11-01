"""
PipeMemory - Underground Infrastructure Memory
==============================================
A digital history of water pipes, repairs, leakage incidents and maintenance
knowledge for Panchayats, municipalities and local water departments.

Run:   streamlit run app.py
Data:  SQLite by default (zero setup). Set DATABASE_URL to use PostgreSQL/PostGIS.
"""

from __future__ import annotations

import math
import os
from datetime import date, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import Fullscreen, HeatMap, MarkerCluster
from sqlalchemy import (
    Column, Date, Float, ForeignKey, Integer, MetaData, String, Table, Text,
    create_engine, func, insert, select, update,
)
from streamlit_folium import st_folium

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="PipeMemory | Underground Infrastructure Memory",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px;}
    .hero {
        background: linear-gradient(120deg, #0b3c5d 0%, #1d6fa5 55%, #2bb3c0 100%);
        padding: 1.5rem 1.8rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;
        box-shadow: 0 6px 24px rgba(11,60,93,.25);
    }
    .hero h1 {margin: 0; font-size: 1.9rem; font-weight: 750; color: white;}
    .hero p  {margin: .35rem 0 0 0; opacity: .92; font-size: .98rem;}
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.18); border-left: 5px solid #1d6fa5;
        padding: .9rem 1rem; border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {font-size: 1.7rem; font-weight: 700;}
    .pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
           font-weight: 600; color: white;}
    .section-title {font-size: 1.15rem; font-weight: 700; margin: .6rem 0 .3rem 0;}
    section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
    .stTabs [data-baseweb="tab"] {font-weight: 600;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MAP_CENTER = (13.0033, 76.1004)  # Hassan, Karnataka - change to your region
MATERIAL_LIFE = {"PVC": 50, "HDPE": 50, "GI": 30, "Cast Iron": 60, "AC": 40, "DI": 70}
CAUSES = ["Corrosion", "Joint failure", "Third-party damage", "Pressure surge",
          "Root intrusion", "Ground movement", "Poor workmanship", "Unknown"]
SEVERITY = ["Minor", "Moderate", "Major"]
SEVERITY_LOSS = {"Minor": (300, 1500), "Moderate": (1500, 6000), "Major": (6000, 25000)}
REPAIR_TYPES = ["Clamp / patch", "Joint replacement", "Section replacement",
                "Valve repair", "Full segment replacement"]
COMPLAINT_TYPES = ["Leakage", "Low pressure", "No supply", "Dirty water", "Burst pipe"]
COMPLAINT_STATUS = ["Open", "In progress", "Resolved"]
RISK_COLORS = {"Low": "#2e9e5b", "Medium": "#f0a202", "High": "#e8590c", "Critical": "#c92a2a"}
COST_PER_METER = {"Replace segment": 3500, "Inspect & pressure test": 150,
                  "Add to next survey": 40, "Routine monitoring": 0}

# ----------------------------------------------------------------------------
# Database layer (SQLAlchemy - SQLite by default, PostgreSQL/PostGIS ready)
# ----------------------------------------------------------------------------
meta = MetaData()

pipes_t = Table(
    "pipes", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(120), nullable=False),
    Column("ward", String(80), nullable=False),
    Column("material", String(30), nullable=False),
    Column("diameter_mm", Integer, nullable=False),
    Column("install_year", Integer, nullable=False),
    Column("lat1", Float, nullable=False), Column("lon1", Float, nullable=False),
    Column("lat2", Float, nullable=False), Column("lon2", Float, nullable=False),
    Column("length_m", Float, nullable=False),
)
leaks_t = Table(
    "leaks", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("pipe_id", Integer, ForeignKey("pipes.id"), nullable=False),
    Column("date", Date, nullable=False),
    Column("severity", String(20)), Column("cause", String(60)),
    Column("lat", Float), Column("lon", Float),
    Column("water_loss_lpd", Float),
    Column("notes", Text),
)
repairs_t = Table(
    "repairs", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("pipe_id", Integer, ForeignKey("pipes.id"), nullable=False),
    Column("date", Date, nullable=False),
    Column("repair_type", String(60)), Column("cost_inr", Float),
    Column("crew", String(80)), Column("notes", Text),
)
complaints_t = Table(
    "complaints", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("pipe_id", Integer, ForeignKey("pipes.id")),
    Column("date", Date, nullable=False),
    Column("complaint_type", String(40)), Column("status", String(20)),
    Column("lat", Float), Column("lon", Float),
    Column("reported_by", String(80)), Column("description", Text),
)


@st.cache_resource(show_spinner=False)
def get_engine():
    url = os.getenv("DATABASE_URL")
    try:
        url = url or st.secrets.get("DATABASE_URL")
    except Exception:
        pass
    url = url or "sqlite:///pipememory.db"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    engine = create_engine(url, pool_pre_ping=True)
    meta.create_all(engine)
    return engine


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def seed_demo_data(engine) -> None:
    """Populate a realistic demo dataset the first time the app runs."""
    with engine.begin() as c:
        if c.execute(select(func.count()).select_from(pipes_t)).scalar():
            return
    rng = np.random.default_rng(42)
    today = date.today()
    wards = ["Ward 1 - Bus Stand", "Ward 2 - Market Road", "Ward 3 - Temple Street",
             "Ward 4 - Railway Colony", "Ward 5 - Lake View", "Ward 6 - School Layout"]
    street = ["Main", "Cross", "Link", "Feeder", "Branch", "Loop"]
    materials = list(MATERIAL_LIFE)
    mat_w = [0.30, 0.20, 0.22, 0.10, 0.10, 0.08]

    pipes = []
    for i in range(30):
        mat = rng.choice(materials, p=mat_w)
        old = mat in ("GI", "AC", "Cast Iron")
        year = int(rng.integers(1972, 2004) if old else rng.integers(1998, 2024))
        lat1 = MAP_CENTER[0] + rng.uniform(-0.022, 0.022)
        lon1 = MAP_CENTER[1] + rng.uniform(-0.028, 0.028)
        lat2 = lat1 + rng.uniform(-0.004, 0.004)
        lon2 = lon1 + rng.uniform(-0.005, 0.005)
        pipes.append(dict(
            name=f"{street[i % 6]} line {i + 1:02d}", ward=wards[i % len(wards)],
            material=str(mat), diameter_mm=int(rng.choice([63, 90, 110, 160, 200, 250])),
            install_year=year, lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2,
            length_m=round(haversine_m(lat1, lon1, lat2, lon2), 1),
        ))
    leaks, repairs, complaints = [], [], []
    for pid, p in enumerate(pipes, start=1):
        life = MATERIAL_LIFE[p["material"]]
        age = today.year - p["install_year"]
        lam = 0.3 + 2.4 * min(age / life, 1.6)
        for _ in range(min(int(rng.poisson(lam)), 7)):
            d = today - timedelta(days=int(rng.integers(5, 1100)))
            t = rng.uniform(0.05, 0.95)
            sev = str(rng.choice(SEVERITY, p=[0.5, 0.35, 0.15]))
            lo, hi = SEVERITY_LOSS[sev]
            leaks.append(dict(
                pipe_id=pid, date=d, severity=sev,
                cause=str(rng.choice(CAUSES, p=[.28, .2, .12, .1, .06, .08, .08, .08])),
                lat=p["lat1"] + t * (p["lat2"] - p["lat1"]),
                lon=p["lon1"] + t * (p["lon2"] - p["lon1"]),
                water_loss_lpd=float(rng.uniform(lo, hi)), notes="",
            ))
            if rng.random() < 0.9:
                rd = min(d + timedelta(days=int(rng.integers(1, 7))), today)
                rtype = {"Minor": REPAIR_TYPES[0], "Moderate": REPAIR_TYPES[1],
                         "Major": REPAIR_TYPES[2]}[sev]
                repairs.append(dict(
                    pipe_id=pid, date=rd, repair_type=rtype,
                    cost_inr=float(rng.uniform(1500, 9000) * (1 + SEVERITY.index(sev))),
                    crew=str(rng.choice(["Crew A", "Crew B", "Crew C"])), notes="",
                ))
    for _ in range(36):
        pid = int(rng.integers(1, len(pipes) + 1))
        p = pipes[pid - 1]
        t = rng.uniform(0, 1)
        complaints.append(dict(
            pipe_id=pid, date=today - timedelta(days=int(rng.integers(0, 120))),
            complaint_type=str(rng.choice(COMPLAINT_TYPES, p=[.35, .25, .15, .1, .15])),
            status=str(rng.choice(COMPLAINT_STATUS, p=[.3, .2, .5])),
            lat=p["lat1"] + t * (p["lat2"] - p["lat1"]) + rng.normal(0, 0.0003),
            lon=p["lon1"] + t * (p["lon2"] - p["lon1"]) + rng.normal(0, 0.0003),
            reported_by=str(rng.choice(["Resident", "Ward member", "Pump operator", "Field staff"])),
            description="Reported via helpline / ward office.",
        ))
    with engine.begin() as c:
        c.execute(insert(pipes_t), pipes)
        c.execute(insert(leaks_t), leaks)
        c.execute(insert(repairs_t), repairs)
        c.execute(insert(complaints_t), complaints)


def load_table(table: Table) -> pd.DataFrame:
    with get_engine().connect() as c:
        df = pd.read_sql(select(table), c)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def insert_row(table: Table, **values) -> None:
    with get_engine().begin() as c:
        c.execute(insert(table).values(**values))


def update_complaint_status(complaint_id: int, status: str) -> None:
    with get_engine().begin() as c:
        c.execute(update(complaints_t).where(complaints_t.c.id == complaint_id).values(status=status))


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "pipes": load_table(pipes_t), "leaks": load_table(leaks_t),
        "repairs": load_table(repairs_t), "complaints": load_table(complaints_t),
    }


# ----------------------------------------------------------------------------
# Analytics: risk scoring & maintenance prediction
# ----------------------------------------------------------------------------
def risk_category(score: float) -> str:
    return "Critical" if score >= 75 else "High" if score >= 55 else "Medium" if score >= 35 else "Low"


def recommended_action(score: float) -> str:
    if score >= 75:
        return "Replace segment"
    if score >= 55:
        return "Inspect & pressure test"
    if score >= 35:
        return "Add to next survey"
    return "Routine monitoring"


def compute_risk(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Explainable 0-100 risk score + next-failure window for every pipe."""
    pipes, leaks, repairs, complaints = data["pipes"], data["leaks"], data["repairs"], data["complaints"]
    today = pd.Timestamp(date.today())
    rows = []
    for _, p in pipes.iterrows():
        life = MATERIAL_LIFE.get(p.material, 50)
        age = date.today().year - int(p.install_year)
        pl = leaks[leaks.pipe_id == p.id].sort_values("date")
        pr = repairs[repairs.pipe_id == p.id]
        pc = complaints[(complaints.pipe_id == p.id) & (complaints.status != "Resolved")]

        age_pts = min(age / life, 1.5) / 1.5 * 35
        leaks_12m = int((pl.date >= today - pd.Timedelta(days=365)).sum())
        leak_pts = min(leaks_12m * 9 + max(len(pl) - leaks_12m, 0) * 2, 30)
        sev_pts = 0.0
        if len(pl):
            recent = pl[pl.date >= today - pd.Timedelta(days=730)]
            if len(recent):
                sev_pts = min(recent.severity.map({"Minor": 1, "Moderate": 2, "Major": 3}).mean() / 3 * 10, 10)
        last_repair = pr.date.max() if len(pr) else pd.NaT
        since = (today - last_repair).days if pd.notna(last_repair) else (age * 365)
        repair_pts = min(since / 1460, 1) * 15 if len(pl) else min(since / 3650, 1) * 5
        complaint_pts = min(len(pc) * 5, 10)

        score = round(min(age_pts + leak_pts + sev_pts + repair_pts + complaint_pts, 100), 1)

        if len(pl) >= 3:
            gaps = pl.date.diff().dropna().dt.days
            interval = max(int(gaps.mean()), 30)
            nxt = max(pl.date.max() + pd.Timedelta(days=interval), today + pd.Timedelta(days=7))
        else:
            nxt = today + pd.Timedelta(days=int(max(30, (100 - score) * 7)))
        rows.append(dict(
            pipe_id=p.id, pipe=p["name"], ward=p.ward, material=p.material,
            age_yrs=age, length_m=p.length_m, total_leaks=len(pl), leaks_12m=leaks_12m,
            open_complaints=len(pc),
            days_since_repair=int(since) if pd.notna(last_repair) else None,
            risk_score=score, risk_level=risk_category(score),
            action=recommended_action(score), predicted_next_failure=nxt.date(),
            est_cost_inr=round(COST_PER_METER[recommended_action(score)] * p.length_m),
        ))
    out = pd.DataFrame(rows)
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True) if len(out) else out


def nearest_pipe(lat: float, lon: float, pipes: pd.DataFrame) -> int | None:
    if pipes.empty:
        return None
    best, best_d = None, float("inf")
    for _, p in pipes.iterrows():
        for t in (0, 0.25, 0.5, 0.75, 1):
            d = haversine_m(lat, lon, p.lat1 + t * (p.lat2 - p.lat1), p.lon1 + t * (p.lon2 - p.lon1))
            if d < best_d:
                best, best_d = int(p.id), d
    return best


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def pill(level: str) -> str:
    return f'<span class="pill" style="background:{RISK_COLORS[level]}">{level}</span>'


def style_fig(fig, height=330):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), title_font_size=15)
    return fig


def build_map(data, risk, show_pipes=True, show_leaks=True, show_heat=False,
              show_complaints=True, show_repairs=False, pick=None) -> folium.Map:
    pipes, leaks, repairs, complaints = data["pipes"], data["leaks"], data["repairs"], data["complaints"]
    m = folium.Map(location=MAP_CENTER, zoom_start=14, tiles="cartodbpositron", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite").add_to(m)
    Fullscreen().add_to(m)
    rk = risk.set_index("pipe_id") if len(risk) else pd.DataFrame()

    if show_pipes:
        g = folium.FeatureGroup(name="Pipe network (by risk)")
        for _, p in pipes.iterrows():
            if p.id not in rk.index:
                continue
            r = rk.loc[p.id]
            tip = (f"<b>{p['name']}</b><br>{p.ward}<br>{p.material} · {p.diameter_mm} mm · {p.install_year}"
                   f"<br>Risk: <b>{r.risk_score}</b> ({r.risk_level})<br>Leaks: {r.total_leaks}"
                   f"<br>Action: {r.action}")
            folium.PolyLine([(p.lat1, p.lon1), (p.lat2, p.lon2)], color=RISK_COLORS[r.risk_level],
                            weight=5 if r.risk_level in ("High", "Critical") else 4,
                            opacity=.9, tooltip=tip).add_to(g)
        g.add_to(m)
    if show_leaks and len(leaks):
        g = MarkerCluster(name="Leak incidents")
        for _, l in leaks.iterrows():
            col = {"Minor": "blue", "Moderate": "orange", "Major": "red"}[l.severity]
            folium.Marker((l.lat, l.lon), icon=folium.Icon(color=col, icon="tint", prefix="fa"),
                          tooltip=f"{l.severity} leak · {l.date:%d %b %Y} · {l.cause}").add_to(g)
        g.add_to(m)
    if show_heat and len(leaks):
        HeatMap([(l.lat, l.lon, l.water_loss_lpd) for _, l in leaks.iterrows()],
                name="Leak heatmap", radius=22, blur=18).add_to(m)
    if show_complaints and len(complaints):
        g = folium.FeatureGroup(name="Complaints")
        cc = {"Open": "#c92a2a", "In progress": "#f0a202", "Resolved": "#2e9e5b"}
        for _, c in complaints.iterrows():
            folium.CircleMarker((c.lat, c.lon), radius=5, color=cc[c.status], fill=True, fill_opacity=.85,
                                tooltip=f"{c.complaint_type} · {c.status} · {c.date:%d %b %Y}").add_to(g)
        g.add_to(m)
    if show_repairs and len(repairs):
        g = folium.FeatureGroup(name="Repairs")
        pm = pipes.set_index("id")
        for _, r in repairs.iterrows():
            if r.pipe_id in pm.index:
                p = pm.loc[r.pipe_id]
                folium.Marker(((p.lat1 + p.lat2) / 2, (p.lon1 + p.lon2) / 2),
                              icon=folium.Icon(color="green", icon="wrench", prefix="fa"),
                              tooltip=f"{r.repair_type} · {r.date:%d %b %Y} · ₹{r.cost_inr:,.0f}").add_to(g)
        g.add_to(m)
    if pick:
        folium.Marker(pick, icon=folium.Icon(color="purple", icon="crosshairs", prefix="fa"),
                      tooltip="Selected location").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_dashboard(data, risk):
    hero("💧 PipeMemory", "Underground infrastructure memory — every pipe, repair, leak and complaint in one place.")
    pipes, leaks, complaints = data["pipes"], data["leaks"], data["complaints"]
    today = pd.Timestamp(date.today())
    leaks30 = leaks[leaks.date >= today - pd.Timedelta(days=30)]
    loss = leaks[leaks.date >= today - pd.Timedelta(days=90)].water_loss_lpd.sum()

    c = st.columns(6)
    c[0].metric("Pipe segments", len(pipes))
    c[1].metric("Network length", f"{pipes.length_m.sum() / 1000:.1f} km")
    c[2].metric("Leaks (30 days)", len(leaks30))
    c[3].metric("Open complaints", int((complaints.status != "Resolved").sum()))
    c[4].metric("High / critical", int(risk.risk_level.isin(["High", "Critical"]).sum()))
    c[5].metric("Est. loss (90d)", f"{loss / 1000:,.0f} kL")

    a, b = st.columns([3, 2])
    with a:
        monthly = leaks.assign(month=leaks.date.dt.to_period("M").dt.to_timestamp()) \
            .groupby(["month", "severity"]).size().reset_index(name="count")
        fig = px.bar(monthly, x="month", y="count", color="severity", title="Leak incidents per month",
                     color_discrete_map={"Minor": "#4dabf7", "Moderate": "#f0a202", "Major": "#c92a2a"})
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with b:
        dist = risk.risk_level.value_counts().reindex(RISK_COLORS.keys(), fill_value=0).reset_index()
        dist.columns = ["level", "pipes"]
        fig = px.pie(dist, names="level", values="pipes", hole=.55, title="Network risk distribution",
                     color="level", color_discrete_map=RISK_COLORS)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">🚨 Top priority segments</div>', unsafe_allow_html=True)
    top = risk.head(6)[["pipe", "ward", "material", "age_yrs", "leaks_12m", "risk_score", "risk_level", "action"]]
    st.dataframe(top, hide_index=True, use_container_width=True, column_config={
        "pipe": "Pipe", "ward": "Ward", "material": "Material", "age_yrs": "Age (yrs)",
        "leaks_12m": "Leaks (12m)", "action": "Recommended action",
        "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f"),
        "risk_level": "Level"})


def page_map(data, risk):
    hero("🗺️ Network Map", "Colour-coded pipes, leak incidents, complaint locations and repair history on a Leaflet map.")
    with st.expander("Map layers", expanded=True):
        c = st.columns(5)
        sp = c[0].toggle("Pipes by risk", True)
        sl = c[1].toggle("Leak markers", True)
        sh = c[2].toggle("Leak heatmap", False)
        sc = c[3].toggle("Complaints", True)
        sr = c[4].toggle("Repairs", False)
    m = build_map(data, risk, sp, sl, sh, sc, sr, st.session_state.get("pick"))
    out = st_folium(m, height=620, use_container_width=True, returned_objects=["last_clicked"])
    if out and out.get("last_clicked"):
        pick = (out["last_clicked"]["lat"], out["last_clicked"]["lng"])
        st.session_state["pick"] = pick
        st.info(f"📍 Selected {pick[0]:.5f}, {pick[1]:.5f} — it will pre-fill the **Log Data** forms.")
    lg = st.columns(4)
    for col, (lvl, colr) in zip(lg, RISK_COLORS.items()):
        col.markdown(f'<span class="pill" style="background:{colr}">{lvl} risk</span>', unsafe_allow_html=True)


def page_registry(data, risk):
    hero("📋 Pipe Registry & History", "Search any segment and read its full life story.")
    pipes = data["pipes"]
    f = st.columns([2, 1, 1])
    q = f[0].text_input("Search by name", placeholder="e.g. Main line 01")
    ward = f[1].selectbox("Ward", ["All"] + sorted(pipes.ward.unique()))
    mat = f[2].selectbox("Material", ["All"] + sorted(pipes.material.unique()))
    view = pipes.merge(risk[["pipe_id", "risk_score", "risk_level", "total_leaks"]], left_on="id", right_on="pipe_id")
    if q:
        view = view[view.name.str.contains(q, case=False)]
    if ward != "All":
        view = view[view.ward == ward]
    if mat != "All":
        view = view[view.material == mat]
    st.dataframe(view[["id", "name", "ward", "material", "diameter_mm", "install_year", "length_m",
                       "total_leaks", "risk_score", "risk_level"]],
                 hide_index=True, use_container_width=True,
                 column_config={"risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                                "length_m": st.column_config.NumberColumn("Length (m)", format="%.0f"),
                                "diameter_mm": "Dia (mm)", "install_year": st.column_config.NumberColumn("Installed", format="%d")})
    if view.empty:
        st.warning("No pipes match the filters.")
        return

    st.divider()
    names = dict(zip(view.id, view.name))
    pid = st.selectbox("Open pipe history", list(names), format_func=lambda i: names[i])
    p = pipes[pipes.id == pid].iloc[0]
    r = risk[risk.pipe_id == pid].iloc[0]
    c = st.columns(4)
    c[0].metric("Risk score", f"{r.risk_score:.0f}")
    c[1].metric("Age", f"{r.age_yrs} yrs")
    c[2].metric("Total leaks", int(r.total_leaks))
    c[3].metric("Next failure window", str(r.predicted_next_failure))
    st.markdown(f"**{p['name']}** · {p.ward} · {p.material} · {p.diameter_mm} mm · {p.length_m:.0f} m &nbsp; "
                f"{pill(r.risk_level)} &nbsp; → *{r.action}*", unsafe_allow_html=True)

    ev = []
    for _, l in data["leaks"][data["leaks"].pipe_id == pid].iterrows():
        ev.append((l.date, "Leak", f"{l.severity} leak — {l.cause} ({l.water_loss_lpd:,.0f} L/day)"))
    for _, x in data["repairs"][data["repairs"].pipe_id == pid].iterrows():
        ev.append((x.date, "Repair", f"{x.repair_type} · ₹{x.cost_inr:,.0f} · {x.crew}"))
    for _, x in data["complaints"][data["complaints"].pipe_id == pid].iterrows():
        ev.append((x.date, "Complaint", f"{x.complaint_type} ({x.status})"))
    if ev:
        tl = pd.DataFrame(ev, columns=["date", "type", "detail"]).sort_values("date", ascending=False)
        fig = px.scatter(tl, x="date", y="type", color="type", hover_data=["detail"], title="Event timeline",
                         color_discrete_map={"Leak": "#c92a2a", "Repair": "#2e9e5b", "Complaint": "#f0a202"})
        fig.update_traces(marker=dict(size=13))
        st.plotly_chart(style_fig(fig, 260), use_container_width=True)
        st.dataframe(tl, hide_index=True, use_container_width=True)
    else:
        st.success("No recorded events for this pipe yet.")


def page_log(data):
    hero("📝 Log Data", "Capture field knowledge before it is lost — leaks, repairs, complaints and new pipes.")
    pipes = data["pipes"]
    names = dict(zip(pipes.id, pipes["name"] + " — " + pipes.ward))
    pick = st.session_state.get("pick")
    dlat, dlon = (pick if pick else MAP_CENTER)
    if pick:
        st.info(f"Using map-selected location {dlat:.5f}, {dlon:.5f} (change on the Network Map).")
    t1, t2, t3, t4, t5 = st.tabs(["💦 Leak incident", "🔧 Repair", "📣 Complaint", "➕ New pipe", "✅ Close complaint"])

    with t1, st.form("leak", clear_on_submit=True):
        c = st.columns(2)
        pid = c[0].selectbox("Pipe", list(names), format_func=lambda i: names[i], key="lk_p")
        d = c[1].date_input("Date", date.today(), key="lk_d")
        c = st.columns(3)
        sev = c[0].selectbox("Severity", SEVERITY)
        cause = c[1].selectbox("Cause", CAUSES)
        loss = c[2].number_input("Water loss (L/day)", 0.0, 100000.0, 1000.0, 100.0)
        c = st.columns(2)
        lat = c[0].number_input("Latitude", value=float(dlat), format="%.5f", key="lk_la")
        lon = c[1].number_input("Longitude", value=float(dlon), format="%.5f", key="lk_lo")
        notes = st.text_area("Notes", key="lk_n")
        if st.form_submit_button("Save leak", type="primary"):
            insert_row(leaks_t, pipe_id=pid, date=d, severity=sev, cause=cause, lat=lat, lon=lon,
                       water_loss_lpd=loss, notes=notes)
            st.toast("Leak incident saved", icon="💦")
            st.rerun()

    with t2, st.form("repair", clear_on_submit=True):
        c = st.columns(2)
        pid = c[0].selectbox("Pipe", list(names), format_func=lambda i: names[i], key="rp_p")
        d = c[1].date_input("Date", date.today(), key="rp_d")
        c = st.columns(3)
        rt = c[0].selectbox("Repair type", REPAIR_TYPES)
        cost = c[1].number_input("Cost (₹)", 0.0, 10_000_000.0, 5000.0, 500.0)
        crew = c[2].text_input("Crew / contractor")
        notes = st.text_area("Notes", key="rp_n")
        if st.form_submit_button("Save repair", type="primary"):
            insert_row(repairs_t, pipe_id=pid, date=d, repair_type=rt, cost_inr=cost, crew=crew, notes=notes)
            st.toast("Repair saved", icon="🔧")
            st.rerun()

    with t3, st.form("complaint", clear_on_submit=True):
        c = st.columns(3)
        ct = c[0].selectbox("Complaint type", COMPLAINT_TYPES)
        by = c[1].text_input("Reported by")
        d = c[2].date_input("Date", date.today(), key="cp_d")
        c = st.columns(2)
        lat = c[0].number_input("Latitude", value=float(dlat), format="%.5f", key="cp_la")
        lon = c[1].number_input("Longitude", value=float(dlon), format="%.5f", key="cp_lo")
        desc = st.text_area("Description")
        if st.form_submit_button("Register complaint", type="primary"):
            pid = nearest_pipe(lat, lon, pipes)
            insert_row(complaints_t, pipe_id=pid, date=d, complaint_type=ct, status="Open",
                       lat=lat, lon=lon, reported_by=by, description=desc)
            st.toast(f"Complaint linked to: {names.get(pid, 'n/a')}", icon="📣")
            st.rerun()

    with t4, st.form("pipe", clear_on_submit=True):
        c = st.columns(3)
        name = c[0].text_input("Pipe name / ID")
        ward = c[1].text_input("Ward / village")
        mat = c[2].selectbox("Material", list(MATERIAL_LIFE))
        c = st.columns(2)
        dia = c[0].number_input("Diameter (mm)", 20, 1200, 110, 10)
        yr = c[1].number_input("Installed year", 1900, date.today().year, 2005)
        c = st.columns(4)
        la1 = c[0].number_input("Start lat", value=float(dlat), format="%.5f")
        lo1 = c[1].number_input("Start lon", value=float(dlon), format="%.5f")
        la2 = c[2].number_input("End lat", value=float(dlat) + 0.002, format="%.5f")
        lo2 = c[3].number_input("End lon", value=float(dlon) + 0.002, format="%.5f")
        if st.form_submit_button("Add pipe", type="primary"):
            if not name or not ward:
                st.error("Name and ward are required.")
            else:
                insert_row(pipes_t, name=name, ward=ward, material=mat, diameter_mm=int(dia), install_year=int(yr),
                           lat1=la1, lon1=lo1, lat2=la2, lon2=lo2,
                           length_m=round(haversine_m(la1, lo1, la2, lo2), 1))
                st.toast("Pipe added", icon="➕")
                st.rerun()

    with t5:
        openc = data["complaints"][data["complaints"].status != "Resolved"].sort_values("date", ascending=False)
        if openc.empty:
            st.success("No open complaints 🎉")
        else:
            labels = {int(r.id): f"#{r.id} · {r.complaint_type} · {r.date:%d %b} · {r.status}" for _, r in openc.iterrows()}
            cid = st.selectbox("Complaint", list(labels), format_func=lambda i: labels[i])
            ns = st.radio("New status", ["In progress", "Resolved"], horizontal=True)
            if st.button("Update status", type="primary"):
                update_complaint_status(cid, ns)
                st.toast("Status updated", icon="✅")
                st.rerun()


def page_patterns(data, risk):
    hero("📈 Leak Patterns", "Understand where, why and when the network fails.")
    leaks = data["leaks"].merge(data["pipes"][["id", "ward", "material"]], left_on="pipe_id", right_on="id", suffixes=("", "_p"))
    c = st.columns(2)
    with c[0]:
        d = leaks.groupby("cause").size().reset_index(name="leaks").sort_values("leaks")
        st.plotly_chart(style_fig(px.bar(d, x="leaks", y="cause", orientation="h", title="Leaks by cause",
                                         color_discrete_sequence=["#1d6fa5"])), use_container_width=True)
    with c[1]:
        d = leaks.groupby("material").agg(leaks=("id", "count"), loss=("water_loss_lpd", "sum")).reset_index()
        n = data["pipes"].groupby("material").length_m.sum().reset_index()
        d = d.merge(n, on="material")
        d["leaks_per_km"] = (d.leaks / (d.length_m / 1000)).round(1)
        st.plotly_chart(style_fig(px.bar(d, x="material", y="leaks_per_km", title="Leaks per km by material",
                                         color="leaks_per_km", color_continuous_scale="Reds")), use_container_width=True)
    c = st.columns(2)
    with c[0]:
        d = leaks.groupby("ward").water_loss_lpd.sum().reset_index().sort_values("water_loss_lpd")
        st.plotly_chart(style_fig(px.bar(d, x="water_loss_lpd", y="ward", orientation="h",
                                         title="Water loss by ward (L/day, cumulative)",
                                         color_discrete_sequence=["#2bb3c0"])), use_container_width=True)
    with c[1]:
        leaks["month_of_year"] = leaks.date.dt.month_name().str[:3]
        order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        d = leaks.groupby(["month_of_year", "severity"]).size().reset_index(name="n")
        st.plotly_chart(style_fig(px.bar(d, x="month_of_year", y="n", color="severity", title="Seasonality",
                                         category_orders={"month_of_year": order},
                                         color_discrete_map={"Minor": "#4dabf7", "Moderate": "#f0a202", "Major": "#c92a2a"})),
                        use_container_width=True)

    st.markdown('<div class="section-title">🔥 Repeat-failure hotspots (~100 m grid)</div>', unsafe_allow_html=True)
    g = leaks.assign(glat=leaks.lat.round(3), glon=leaks.lon.round(3)) \
        .groupby(["glat", "glon"]).agg(leaks=("id", "count"), loss=("water_loss_lpd", "sum"), ward=("ward", "first")).reset_index()
    hot = g[g.leaks >= 2].sort_values("leaks", ascending=False).head(10)
    if hot.empty:
        st.info("No repeat-failure hotspots yet.")
    else:
        st.dataframe(hot.rename(columns={"glat": "Lat", "glon": "Lon", "leaks": "Leaks", "loss": "Loss (L/day)", "ward": "Ward"}),
                     hide_index=True, use_container_width=True)


def page_maintenance(data, risk):
    hero("🛠️ Maintenance Planner", "Risk-ranked, explainable predictions to plan budgets and crews.")
    c = st.columns(3)
    levels = c[0].multiselect("Risk level", list(RISK_COLORS), default=["High", "Critical"])
    wards = c[1].multiselect("Ward", sorted(risk.ward.unique()))
    horizon = c[2].slider("Failure horizon (days)", 30, 730, 365, 30)
    v = risk[risk.risk_level.isin(levels)]
    if wards:
        v = v[v.ward.isin(wards)]
    cutoff = date.today() + timedelta(days=horizon)
    v = v[v.predicted_next_failure <= cutoff]

    m = st.columns(3)
    m[0].metric("Segments selected", len(v))
    m[1].metric("Length", f"{v.length_m.sum() / 1000:.2f} km")
    m[2].metric("Estimated budget", f"₹{v.est_cost_inr.sum():,.0f}")

    st.dataframe(v[["pipe", "ward", "material", "age_yrs", "total_leaks", "leaks_12m", "open_complaints",
                    "risk_score", "risk_level", "predicted_next_failure", "action", "est_cost_inr"]],
                 hide_index=True, use_container_width=True,
                 column_config={"risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                                "est_cost_inr": st.column_config.NumberColumn("Est. cost (₹)", format="₹%d"),
                                "predicted_next_failure": "Next failure (est.)", "action": "Action"})
    st.download_button("⬇️ Download plan (CSV)", v.to_csv(index=False).encode(), "maintenance_plan.csv", "text/csv")
    with st.expander("How is the risk score calculated?"):
        st.markdown(
            "- **Age vs design life (35 pts)** — material-specific life: " + ", ".join(f"{k} {yrs} yrs" for k, yrs in MATERIAL_LIFE.items()) +
            "\n- **Leak history (30 pts)** — recent leaks weigh more than old ones\n"
            "- **Severity (10 pts)** — average severity over the last 2 years\n"
            "- **Time since last repair (15 pts)**\n- **Open complaints (10 pts)**\n\n"
            "Next-failure estimate uses the mean gap between past leaks (≥3 leaks), otherwise a risk-scaled window. "
            "This is a transparent heuristic — replace it with a trained model once enough data is collected.")


# ----------------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------------
def main():
    engine = get_engine()
    seed_demo_data(engine)
    data = load_all()
    risk = compute_risk(data)

    with st.sidebar:
        st.markdown("## 💧 PipeMemory")
        st.caption("Underground Infrastructure Memory")
        page = st.radio("Navigate", ["🏠 Dashboard", "🗺️ Network Map", "📋 Pipe Registry", "📝 Log Data",
                                     "📈 Leak Patterns", "🛠️ Maintenance Planner"], label_visibility="collapsed")
        st.divider()
        st.caption(f"Database: `{engine.dialect.name}`")
        st.caption(f"Records: {len(data['pipes'])} pipes · {len(data['leaks'])} leaks · "
                   f"{len(data['repairs'])} repairs · {len(data['complaints'])} complaints")
        if st.session_state.get("pick") and st.button("Clear map selection"):
            st.session_state.pop("pick")
            st.rerun()

    if page.endswith("Dashboard"):
        page_dashboard(data, risk)
    elif page.endswith("Network Map"):
        page_map(data, risk)
    elif page.endswith("Pipe Registry"):
        page_registry(data, risk)
    elif page.endswith("Log Data"):
        page_log(data)
    elif page.endswith("Leak Patterns"):
        page_patterns(data, risk)
    else:
        page_maintenance(data, risk)


main()
