"""
SITER-CAE v6.1 DECISION + EVIDENCE LAB
=========================
Laboratorio territorial para CDMX (v5.9: capa organización agregada + red geo + guardrails).

OBJETIVO
--------
Unificar en una sola app:
1) Motor ABM con Mesa.
2) World View tipo NetLogo sobre GIS real cuando exista SHP/GeoJSON.
3) Niveles de evidencia D0–D7:
   - D0 dummy, D1 calibración controlada, D2 sintético puro, D3 sintético coherente,
     D4 real agregado, D5 real+GIS, D6 real+campo, D7 real+GIS+campo+organización.
4) Análisis estadístico territorial.
5) Presupuesto: costos, cobertura, costo/km, costo/unidad, ROI operacional,
   sensibilidad y Monte Carlo.
6) Brigadistas: plan, GPS observado, cobertura por segmento, territorio
   visitado, horas, km, desviaciones y calidad GPS.
7) Desagregación avanzada agregada: IPF, tomografía, Bayes/Kalman y MRF/CRF-like.
8) Question Engine para responder preguntas de cliente a partir de métricas.
9) Reproducibilidad: seed + experiment_id + output_hash.
9) Export JSON/CSV.

NOTA METODOLÓGICA
-----------------
Los estados de opinión y actores son variables de simulación agregadas.
No se construyen perfiles individuales ni PII. La app está diseñada para
análisis territorial, logística, presupuesto, resiliencia y escenarios.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# Mesa
# ---------------------------------------------------------------------
try:
    from mesa import Agent, Model
    MESA_OK = True
    MESA_ERROR = ""
except Exception as exc:
    MESA_OK = False
    MESA_ERROR = repr(exc)

# ---------------------------------------------------------------------
# GIS opcional pero recomendado para SHP/GeoJSON
# ---------------------------------------------------------------------
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union
    HAS_GIS = True
    GIS_ERROR = ""
except Exception as exc:
    HAS_GIS = False
    GIS_ERROR = repr(exc)

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
ALCALDIAS = [
    "ALVARO OBREGON",
    "AZCAPOTZALCO",
    "BENITO JUAREZ",
    "COYOACAN",
    "CUAJIMALPA DE MORELOS",
    "CUAUHTEMOC",
    "GUSTAVO A MADERO",
    "IZTACALCO",
    "IZTAPALAPA",
    "LA MAGDALENA CONTRERAS",
    "MIGUEL HIDALGO",
    "MILPA ALTA",
    "TLAHUAC",
    "TLALPAN",
    "VENUSTIANO CARRANZA",
    "XOCHIMILCO",
]

ALCALDIA_COORDS = {
    "CUAUHTEMOC": (19.4326, -99.1332),
    "BENITO JUAREZ": (19.3984, -99.1576),
    "MIGUEL HIDALGO": (19.4285, -99.2000),
    "COYOACAN": (19.3467, -99.1617),
    "IZTAPALAPA": (19.3550, -99.0620),
    "GUSTAVO A MADERO": (19.4900, -99.1100),
    "ALVARO OBREGON": (19.3580, -99.2270),
    "TLALPAN": (19.2880, -99.1670),
    "XOCHIMILCO": (19.2630, -99.1040),
    "VENUSTIANO CARRANZA": (19.4200, -99.1000),
    "AZCAPOTZALCO": (19.4870, -99.1860),
    "IZTACALCO": (19.3950, -99.0980),
    "CUAJIMALPA DE MORELOS": (19.3570, -99.2900),
    "LA MAGDALENA CONTRERAS": (19.3200, -99.2400),
    "TLAHUAC": (19.2700, -99.0050),
    "MILPA ALTA": (19.1920, -99.0230),
}

STATE_LABEL = {1: "SIMPATIZANTE", -1: "OPOSITOR", 0: "INDECISO"}
STATE_COLORS = {"SIMPATIZANTE": "#2ecc71", "OPOSITOR": "#e74c3c", "INDECISO": "#95a5a6"}
FIELD_COLORS = {
    "CONSOLIDACION": "#2ecc71",
    "DISPUTA_ABIERTA": "#e74c3c",
    "CONTENCION": "#3498db",
}

TRAIT_COLS = [
    "capital_social",
    "acceso_informacion",
    "influencia_liderazgo",
    "arraigo",
    "nivel_movilizacion",
    "desconfianza",
    "exposicion_problema",
]

# Solo estas variables pueden actualizarse con observaciones de campo (nunca GPS).
FIELD_UPDATABLE_VARS = set(TRAIT_COLS + [
    "prioridad_problema",
    "resistencia_institucional",
    "temperatura",
    "opinion_continua",
])

# Capa de organización / militancia AGREGADA (sin PII).
ORG_COLS = [
    "n_militantes_obs",
    "n_universo_est",
    "broker_density",
    "asistencia_evento_rate",
    "aceptacion_mensaje",
    "org_mobilization",
    "org_reliability",
]

# Capa INDIVIDUAL OPERATIVA: equipos y líderes reales autorizados.
# No contiene preferencias electorales ni perfiles de ciudadanos/votantes.
ACTOR_OPERATIONAL_COLS = [
    "capacidad_coordinacion", "conocimiento_territorial", "confiabilidad",
    "capacidad_movilizacion_operativa", "carga_horas", "fatiga",
    "cobertura_operativa", "calidad_reporte",
]
ACTOR_ID_ALIASES = {"actor_id", "lider_id", "leader_id", "id_actor", "id_lider"}
DATA_EVIDENCE_LEVELS = [
    ("D0", "DUMMY · smoke test", "solo prueba de software"),
    ("D1", "SINTÉTICO CALIBRACIÓN", "escenarios controlados conocidos"),
    ("D2", "SINTÉTICO PURO", "stress test generativo"),
    ("D3", "SINTÉTICO COHERENTE", "derivado estadísticamente de real"),
    ("D4", "REAL AGREGADO", "CSV territorial real"),
    ("D5", "REAL + GIS", "CSV real + SHP/GeoJSON/GPKG"),
    ("D6", "REAL + GIS + CAMPO", "además GPS/observaciones"),
    ("D7", "REAL + GIS + CAMPO + ORGANIZACIÓN", "evidencia operacional completa"),
]

NUMERIC_CANDIDATES = TRAIT_COLS + [
    "opinion_continua",
    "temperatura",
    "resistencia_institucional",
    "prioridad_problema",
    "poblacion",
    "longitud_km",
]

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def norm_col(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9áéíóúüñ]+", "_", s)
    return s.strip("_")


def clean_alcaldia(x: Any) -> str:
    s = str(x).strip().upper()
    replacements = {
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "Ü": "U", "Ñ": "N",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s


def sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * r * math.asin(min(1, math.sqrt(a)))


def path_distance_m(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 2:
        return 0.0
    d = 0.0
    x = df.sort_values("timestamp") if "timestamp" in df.columns else df
    for i in range(1, len(x)):
        d += haversine_m(
            x.iloc[i-1]["lat"], x.iloc[i-1]["lon"],
            x.iloc[i]["lat"], x.iloc[i]["lon"]
        )
    return d


def gini(values) -> float:
    x = np.asarray(values, dtype=float)
    x = np.abs(x[np.isfinite(x)])
    if len(x) == 0 or np.allclose(x.sum(), 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def entropy_from_counts(counts) -> float:
    vals = np.asarray(list(counts), dtype=float)
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0.0
    p = vals / vals.sum()
    return float(-np.sum(p * np.log(p)))


def field_state(simpat, indec, stability, polarization):
    if stability >= 0.70 and max(simpat, 1 - simpat - indec) >= 0.50:
        return "CONSOLIDACION"
    if indec >= 0.35 or polarization >= 0.55 or stability < 0.45:
        return "DISPUTA_ABIERTA"
    return "CONTENCION"


def ensure_lat_lon(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    if "alcaldia" not in out:
        out["alcaldia"] = rng.choice(ALCALDIAS, len(out))
    out["alcaldia"] = out["alcaldia"].map(clean_alcaldia)
    lats, lons = [], []
    for a in out["alcaldia"]:
        lat, lon = ALCALDIA_COORDS.get(a, (19.35, -99.15))
        lats.append(lat + rng.normal(0, 0.007))
        lons.append(lon + rng.normal(0, 0.007))
    if "lat" not in out:
        out["lat"] = lats
    else:
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lat"] = out["lat"].fillna(pd.Series(lats, index=out.index))
    if "lon" not in out:
        out["lon"] = lons
    else:
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out["lon"] = out["lon"].fillna(pd.Series(lons, index=out.index))
    return out


def normalize_base(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_col(c) for c in out.columns]
    aliases = {
        "seccion_electoral": "seccion",
        "seccion_id": "seccion",
        "section": "seccion",
        "alcaldía": "alcaldia",
        "delegacion": "alcaldia",
        "territory_id": "territorial_unit_id",
        "id_territorial": "territorial_unit_id",
        "id": "territorial_unit_id",
        "utm_id": "utm",
        "unidad_territorial_media": "utm",
        "manzana_id": "manzana",
        "manzana_geo": "manzana",
    }
    for a, b in aliases.items():
        if a in out.columns and b not in out.columns:
            out[b] = out[a]

    if "territorial_unit_id" not in out:
        if "seccion" in out:
            out["territorial_unit_id"] = "CDMX-SEC-" + out["seccion"].astype(str)
        else:
            out["territorial_unit_id"] = [f"CDMX-SEC-{i+1:05d}" for i in range(len(out))]

    if "seccion" not in out:
        out["seccion"] = out["territorial_unit_id"].astype(str)

    # Jerarquía oficial de trabajo de SITER-CAE: CDMX → Alcaldía → UTM → Sección → Manzana.
    # UTM/manzana se conservan vacías cuando el archivo real no las contiene; nunca se inventan.
    if "utm" not in out:
        out["utm"] = ""
    if "manzana" not in out:
        out["manzana"] = ""

    if "alcaldia" not in out:
        out["alcaldia"] = "NO_ESPECIFICADA"

    out["alcaldia"] = out["alcaldia"].map(clean_alcaldia)

    defaults = {
        "opinion_continua": 0.0,
        "capital_social": 0.5,
        "acceso_informacion": 0.5,
        "influencia_liderazgo": 0.5,
        "arraigo": 0.5,
        "nivel_movilizacion": 0.5,
        "desconfianza": 0.5,
        "exposicion_problema": 0.5,
        "temperatura": 0.5,
        "resistencia_institucional": 0.5,
        "prioridad_problema": 0.5,
        "poblacion": 1000.0,
    }
    for c, v in defaults.items():
        if c not in out:
            out[c] = v

    for c, default_value in defaults.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default_value)

    out["opinion_continua"] = out["opinion_continua"].clip(-1, 1)
    out = ensure_lat_lon(out, seed=seed)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# DataProvider
# ---------------------------------------------------------------------
class DataProvider:
    MODES = {
        "DEMO PÚBLICA CDMX · 16 alcaldías": "demo_public",
        "REAL · CSV + SHP/GeoJSON": "real",
        "DUMMY · prueba mínima": "dummy",
        "SINTÉTICO COHERENTE · derivado de real": "coherent",
        "SINTÉTICO PURO · generativo": "pure",
        "SINTÉTICO CALIBRACIÓN · pruebas controladas": "calib",
    }

    def __init__(self, seed=42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def dummy(self, n=48):
        rng = self.rng
        rows = []
        for i in range(n):
            alc = ALCALDIAS[i % len(ALCALDIAS)]
            lat, lon = ALCALDIA_COORDS[alc]
            op = [-0.7, -0.2, 0.0, 0.2, 0.7][i % 5] + rng.normal(0, 0.04)
            rows.append({
                "territorial_unit_id": f"DUMMY-{i+1:04d}",
                "alcaldia": alc,
                "seccion": str(1000+i),
                "opinion_continua": float(np.clip(op, -1, 1)),
                "capital_social": float(0.35 + 0.45*rng.random()),
                "acceso_informacion": float(0.35 + 0.45*rng.random()),
                "influencia_liderazgo": float(0.25 + 0.65*rng.random()),
                "arraigo": float(0.3 + 0.6*rng.random()),
                "nivel_movilizacion": float(0.25 + 0.7*rng.random()),
                "desconfianza": float(0.15 + 0.55*rng.random()),
                "exposicion_problema": float(0.2 + 0.7*rng.random()),
                "prioridad_problema": float(0.2 + 0.8*rng.random()),
                "resistencia_institucional": float(0.2 + 0.6*rng.random()),
            })
        return normalize_base(pd.DataFrame(rows), self.seed), {"mode": "dummy", "source": "generated"}

    def pure(self, n=300):
        rng = self.rng
        rows = []
        for i in range(n):
            alc = str(rng.choice(ALCALDIAS))
            lat, lon = ALCALDIA_COORDS[alc]
            social = rng.beta(4, 4)
            info = rng.beta(4, 4)
            lead = rng.beta(4, 4)
            arraigo = rng.beta(4, 4)
            mov = rng.beta(4, 4)
            distrust = rng.beta(3, 5)
            exposure = rng.beta(4, 4)
            skill = np.clip(
                .30*social + .25*info + .25*lead + .10*arraigo + .10*mov - .15*distrust,
                0, 1
            )
            op = np.clip(0.7*(2*social-1) + 0.25*(2*lead-1) + rng.normal(0, .12), -1, 1)
            rows.append({
                "territorial_unit_id": f"SYN-{i+1:05d}",
                "alcaldia": alc,
                "seccion": str(2000+i),
                "lat": lat + rng.normal(0, .009),
                "lon": lon + rng.normal(0, .009),
                "opinion_continua": op,
                "capital_social": social,
                "acceso_informacion": info,
                "influencia_liderazgo": lead,
                "arraigo": arraigo,
                "nivel_movilizacion": mov,
                "desconfianza": distrust,
                "exposicion_problema": exposure,
                "prioridad_problema": exposure,
                "resistencia_institucional": rng.beta(2, 5),
                "poblacion": max(100, rng.lognormal(7.0, .45)),
                "saf_skill": skill,
            })
        return normalize_base(pd.DataFrame(rows), self.seed), {"mode": "synthetic_pure", "source": "generative"}

    def calibration(self, n=240, scenario="balance"):
        rng = self.rng
        rows = []
        scenarios = {
            "balance": lambda i: rng.normal(0, .12),
            "polarizacion": lambda i: (0.75 if i % 2 == 0 else -0.75) + rng.normal(0, .05),
            "fragmentacion": lambda i: [-.8, -.3, .0, .35, .8][i % 5] + rng.normal(0, .03),
            "consenso": lambda i: .55 + rng.normal(0, .05),
            "resistencia_alta": lambda i: rng.normal(0, .20),
            "red_hub": lambda i: rng.normal(0, .10),
        }
        fn = scenarios.get(scenario, scenarios["balance"])
        for i in range(n):
            alc = ALCALDIAS[i % len(ALCALDIAS)]
            lat, lon = ALCALDIA_COORDS[alc]
            op = float(np.clip(fn(i), -1, 1))
            rows.append({
                "territorial_unit_id": f"CAL-{scenario[:4].upper()}-{i+1:05d}",
                "alcaldia": alc,
                "seccion": str(3000+i),
                "lat": lat + rng.normal(0, .006),
                "lon": lon + rng.normal(0, .006),
                "opinion_continua": op,
                "capital_social": .55,
                "acceso_informacion": .55,
                "influencia_liderazgo": .55,
                "arraigo": .55,
                "nivel_movilizacion": .55,
                "desconfianza": .35,
                "exposicion_problema": .60,
                "prioridad_problema": .60,
                "resistencia_institucional": .80 if scenario == "resistencia_alta" else .35,
                "poblacion": 1000,
            })
        return normalize_base(pd.DataFrame(rows), self.seed), {
            "mode": "synthetic_calibration",
            "scenario": scenario,
            "source": "controlled_test"
        }

    def coherent_from_real(self, real_df, n=None):
        base = normalize_base(real_df, self.seed)
        n = int(n or len(base))
        rng = self.rng

        cats = {}
        for c in ["alcaldia"]:
            p = base[c].value_counts(normalize=True)
            cats[c] = rng.choice(p.index.to_numpy(), n, p=p.to_numpy())

        numeric = [c for c in TRAIT_COLS + ["opinion_continua", "temperatura",
                                            "resistencia_institucional", "prioridad_problema"]
                   if c in base.columns]
        X = base[numeric].apply(pd.to_numeric, errors="coerce").fillna(base[numeric].median())
        ranks = X.rank(pct=True).to_numpy()
        R = np.corrcoef(ranks.T)
        R = np.nan_to_num(R, nan=0.0)
        R = (R + R.T) / 2
        np.fill_diagonal(R, 1.0)
        vals, vecs = np.linalg.eigh(R)
        vals = np.clip(vals, 1e-6, None)
        L = vecs @ np.diag(np.sqrt(vals))
        Z = rng.normal(size=(n, len(numeric))) @ L.T
        U = 0.5 * (1 + np.vectorize(math.erf)(Z / math.sqrt(2)))
        out = pd.DataFrame({"alcaldia": cats["alcaldia"]})
        for j, c in enumerate(numeric):
            arr = np.sort(X[c].to_numpy(dtype=float))
            q = np.clip(U[:, j], 0, 1)
            idx = np.minimum((q * (len(arr)-1)).astype(int), len(arr)-1)
            out[c] = arr[idx]

        out["territorial_unit_id"] = [f"COH-{i+1:05d}" for i in range(n)]
        out["seccion"] = [f"C{i+1:05d}" for i in range(n)]
        out["poblacion"] = float(base["poblacion"].median()) if "poblacion" in base else 1000
        out = ensure_lat_lon(out, self.seed)
        meta = {
            "mode": "synthetic_coherent",
            "derived_from_real": True,
            "real_n": len(base),
            "synthetic_n": len(out),
            "numeric_basis": numeric,
            "source_hash": sha256_obj({
                "columns": list(base.columns),
                "shape": base.shape,
                "alcaldia_distribution": base["alcaldia"].value_counts().to_dict(),
            })
        }
        return normalize_base(out, self.seed), meta

    def real(self, base_csv, electoral_csv=None, socio_csv=None):
        if base_csv is None:
            raise ValueError("El modo REAL requiere el CSV base territorial.")
        base = pd.read_csv(base_csv)
        base = normalize_base(base, self.seed)

        if electoral_csv is not None:
            elec = pd.read_csv(electoral_csv)
            elec.columns = [norm_col(c) for c in elec.columns]
            if "seccion" in elec:
                elec["seccion"] = elec["seccion"].astype(str)
                if "votos" in elec.columns:
                    group = elec.groupby("seccion")["votos"].sum().rename("votos_total")
                    mx = elec.groupby("seccion")["votos"].max().rename("votos_max")
                    stats = pd.concat([group, mx], axis=1)
                    stats["share_max"] = (stats["votos_max"] / stats["votos_total"]).fillna(0)
                    base["seccion"] = base["seccion"].astype(str)
                    base = base.merge(stats[["share_max"]], left_on="seccion", right_index=True, how="left")
                    base["opinion_continua"] = (2*base["share_max"].fillna(.33)-1).clip(-1,1)

        if socio_csv is not None:
            socio = pd.read_csv(socio_csv)
            socio.columns = [norm_col(c) for c in socio.columns]
            if "seccion" in socio.columns:
                socio["seccion"] = socio["seccion"].astype(str)
                base["seccion"] = base["seccion"].astype(str)
                keep = [c for c in socio.columns if c != "alcaldia"]
                base = base.merge(socio[keep], on="seccion", how="left", suffixes=("", "_socio"))

        base = normalize_base(base, self.seed)
        meta = {
            "mode": "real",
            "source": "csv",
            "n": len(base),
            "columns": list(base.columns),
            "source_hash": sha256_obj({
                "shape": base.shape,
                "columns": list(base.columns),
            }),
        }
        return base, meta


# ---------------------------------------------------------------------
# GIS loader
# ---------------------------------------------------------------------
def read_vector_upload(uploaded) -> Optional["gpd.GeoDataFrame"]:
    if uploaded is None or not HAS_GIS:
        return None
    suffix = Path(uploaded.name).suffix.lower()
    data = uploaded.getvalue()
    tmpdir = Path(tempfile.mkdtemp(prefix="siter_gis_"))
    if suffix == ".zip":
        zpath = tmpdir / uploaded.name
        zpath.write_bytes(data)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmpdir / "unzipped")
        shps = list((tmpdir / "unzipped").rglob("*.shp"))
        if not shps:
            raise ValueError("El ZIP no contiene un .shp.")
        return gpd.read_file(shps[0])
    if suffix in [".geojson", ".json", ".gpkg", ".shp"]:
        p = tmpdir / uploaded.name
        p.write_bytes(data)
        return gpd.read_file(p)
    raise ValueError("Formato GIS no soportado. Usa ZIP de Shapefile, GeoJSON o GPKG.")


def normalize_gdf(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    g = gdf.copy()
    g.columns = [norm_col(c) for c in g.columns]
    if g.crs is None:
        g = g.set_crs(4326, allow_override=True)
    if g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    if "alcaldia" in g:
        g["alcaldia"] = g["alcaldia"].map(clean_alcaldia)
    return g


def make_dummy_cdmx_gis():
    half = {
        "CUAUHTEMOC": (0.025, 0.025),
        "BENITO JUAREZ": (0.022, 0.022),
        "MIGUEL HIDALGO": (0.030, 0.028),
        "COYOACAN": (0.035, 0.032),
        "IZTAPALAPA": (0.045, 0.040),
        "GUSTAVO A MADERO": (0.040, 0.038),
        "ALVARO OBREGON": (0.040, 0.035),
        "TLALPAN": (0.050, 0.045),
        "XOCHIMILCO": (0.035, 0.035),
        "VENUSTIANO CARRANZA": (0.025, 0.022),
        "AZCAPOTZALCO": (0.028, 0.025),
        "IZTACALCO": (0.020, 0.018),
        "CUAJIMALPA DE MORELOS": (0.035, 0.030),
        "LA MAGDALENA CONTRERAS": (0.030, 0.028),
        "TLAHUAC": (0.035, 0.032),
        "MILPA ALTA": (0.045, 0.040),
    }
    features = []
    for i, (name, (lat, lon)) in enumerate(ALCALDIA_COORDS.items(), 1):
        dlat, dlon = half.get(name, (0.03, 0.03))
        ring = [
            [lon - dlon, lat - dlat],
            [lon + dlon, lat - dlat],
            [lon + dlon, lat + dlat],
            [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat],
        ]
        features.append({
            "type": "Feature",
            "properties": {
                "alcaldia": name,
                "seccion": f"DUM-{i:03d}",
                "territorial_unit_id": f"DUMMY-GIS-{i:04d}",
                "note": "poligono_aproximado_prueba_no_oficial",
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    gj = {
        "type": "FeatureCollection",
        "name": "CDMX_alcaldias_dummy_test",
        "features": features,
    }
    if HAS_GIS:
        gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
        gdf.columns = [norm_col(c) for c in gdf.columns]
        if "alcaldia" in gdf.columns:
            gdf["alcaldia"] = gdf["alcaldia"].map(clean_alcaldia)
        return gdf, gj
    return None, gj


def join_base_to_geometry(df, gdf):
    if gdf is None:
        return None
    g = normalize_gdf(gdf)
    for key in ["territorial_unit_id", "seccion"]:
        if key in df.columns and key in g.columns:
            left = df.copy()
            left[key] = left[key].astype(str)
            g[key] = g[key].astype(str)
            merged = g.merge(
                left.drop_duplicates(key),
                on=key,
                how="left",
                suffixes=("_gis", "")
            )
            return merged
    return g


# ---------------------------------------------------------------------
# Behaviors Mesa
# ---------------------------------------------------------------------
class Behavior:
    name = "base"
    def __init__(self, params=None):
        self.params = params or {}
    def step_agent(self, agent):
        raise NotImplementedError


class VoterBehavior(Behavior):
    name = "Voter / difusión local"
    def step_agent(self, agent):
        ns = agent.get_neighbors()
        if not ns:
            return
        other = agent.model.random.choice(ns)
        beta = float(self.params.get("beta", 1.2))
        prob = min(0.90, max(0.0, other.influencia * beta))
        if agent.model.random.random() < prob:
            agent.next_opinion = clamp(agent.opinion + .20*(other.opinion-agent.opinion))


class DeffuantBehavior(Behavior):
    name = "Deffuant-Weisbuch"
    def step_agent(self, agent):
        ns = agent.get_neighbors()
        if not ns:
            agent.next_opinion = agent.opinion
            return
        other = agent.model.random.choice(ns)
        eps = float(self.params.get("epsilon", .40))
        mu = float(self.params.get("mu", .30))
        rep = float(self.params.get("epsilon_repulsion", .80))
        d = abs(agent.opinion-other.opinion)
        if d <= eps:
            agent.next_opinion = clamp(agent.opinion + mu*(other.opinion-agent.opinion))
        elif d >= rep:
            agent.next_opinion = clamp(agent.opinion - .5*mu*(other.opinion-agent.opinion))
        else:
            agent.next_opinion = agent.opinion


class SAFBehavior(Behavior):
    name = "ABM-SAF"
    def step_agent(self, agent):
        ns = agent.get_neighbors()
        if not ns:
            agent.next_opinion = agent.opinion
            return
        mean_n = float(np.mean([x.opinion for x in ns]))
        coupling = float(self.params.get("coupling", .20))
        field_pressure = float(self.params.get("field_pressure", .08))
        org_boost = 1.0 + 0.5 * float(getattr(agent, "broker_density", 0.0))
        org_boost *= 1.0 + 0.3 * float(getattr(agent, "org_mobilization", 0.0))
        coupling_eff = coupling * org_boost
        fatigue_penalty = max(0.0, 1.0 - agent.fatiga)
        agent.next_opinion = clamp(
            agent.opinion
            + coupling_eff * agent.saf_skill * fatigue_penalty * (mean_n - agent.opinion)
            + field_pressure * agent.exposure * (-agent.opinion)
        )


BEHAVIORS = {
    "Voter / difusión local": VoterBehavior,
    "Deffuant-Weisbuch": DeffuantBehavior,
    "ABM-SAF": SAFBehavior,
}


def clamp(x, lo=-1, hi=1):
    return float(max(lo, min(hi, float(x))))


# ---------------------------------------------------------------------
# Mesa model
# ---------------------------------------------------------------------
if MESA_OK:
    class SeccionAgent(Agent):
        def __init__(self, model, row):
            super().__init__(model)
            self.territorial_unit_id = str(row["territorial_unit_id"])
            self.alcaldia = str(row["alcaldia"])
            self.seccion = str(row["seccion"])
            self.lat = float(row["lat"])
            self.lon = float(row["lon"])
            self.population = float(row.get("poblacion", 1000.0) or 1000.0)
            self.temperatura = float(row.get("temperatura", row.get("exposicion_problema", .5)) or .5)
            self.opinion = clamp(row.get("opinion_continua", 0))
            self.next_opinion = self.opinion
            self.capital_social = float(row.get("capital_social", .5))
            self.acceso_informacion = float(row.get("acceso_informacion", .5))
            self.influencia_liderazgo = float(row.get("influencia_liderazgo", .5))
            self.arraigo = float(row.get("arraigo", .5))
            self.nivel_movilizacion = float(row.get("nivel_movilizacion", .5))
            self.desconfianza = float(row.get("desconfianza", .5))
            self.exposure = float(row.get("exposicion_problema", .5))
            self.resistencia_institucional = float(row.get("resistencia_institucional", .5))
            self.prioridad_problema = float(row.get("prioridad_problema", .5))
            self.fatiga = 0.0
            self.influencia = 0.0
            self.es_broker = False
            self.neighbor_ids = []
            self.broker_density = float(row.get("broker_density", 0.0) or 0.0)
            self.org_mobilization = float(row.get("org_mobilization", 0.0) or 0.0)
            self.aceptacion_mensaje = float(row.get("aceptacion_mensaje", 0.0) or 0.0)
            self.org_reliability = float(row.get("org_reliability", 0.0) or 0.0)
            self.n_militantes_obs = float(row.get("n_militantes_obs", 0.0) or 0.0)

        @property
        def spin(self):
            if self.opinion > .25:
                return 1
            if self.opinion < -.25:
                return -1
            return 0

        @property
        def saf_skill(self):
            return clamp(
                .30*self.capital_social
                + .25*self.acceso_informacion
                + .25*self.influencia_liderazgo
                + .10*self.arraigo
                + .10*self.nivel_movilizacion
                - .15*self.desconfianza,
                0, 1
            )

        def get_neighbors(self):
            return [self.model.agent_by_uid[x] for x in self.neighbor_ids if x in self.model.agent_by_uid]

        def step(self):
            self.model.behavior.step_agent(self)

        def advance(self):
            self.opinion = clamp(self.next_opinion)

    class BrokerAgent(SeccionAgent):
        def __init__(self, model, row):
            super().__init__(model, row)
            self.es_broker = True

    class SITERModel(Model):
        def __init__(self, df, behavior_name="ABM-SAF", seed=42,
                     p_intra=.06, p_inter=.015, params=None):
            super().__init__(seed=int(seed))
            self.seed_value = int(seed)
            self.df = df.reset_index(drop=True).copy()
            self.behavior_name = behavior_name
            self.behavior = BEHAVIORS[behavior_name](params or {})
            self.p_intra = float(p_intra)
            self.p_inter = float(p_inter)
            self.params = params or {}
            self.G = nx.Graph()
            self.agent_by_uid = {}
            self._build_agents()
            self._build_network()
            self._collect_history()

        def _build_agents(self):
            for _, row in self.df.iterrows():
                a = SeccionAgent(self, row)
                self.agent_by_uid[str(a.unique_id)] = a
                self.G.add_node(str(a.unique_id), territorial_unit_id=a.territorial_unit_id,
                                alcaldia=a.alcaldia)

        def _build_network(self):
            rng = np.random.default_rng(self.seed_value)
            agents = list(self.agent_by_uid.values())
            use_geo = bool(self.params.get("use_geo_network", False))
            geo_km = float(self.params.get("geo_radius_km", 3.0))
            if use_geo and len(agents) > 1:
                for i, a in enumerate(agents):
                    for b in agents[i+1:]:
                        d_m = haversine_m(a.lat, a.lon, b.lat, b.lon)
                        if d_m <= geo_km * 1000.0:
                            self.G.add_edge(str(a.unique_id), str(b.unique_id))
                        else:
                            p = self.p_intra if a.alcaldia == b.alcaldia else self.p_inter
                            if rng.random() < p * 0.25:
                                self.G.add_edge(str(a.unique_id), str(b.unique_id))
            else:
                for i, a in enumerate(agents):
                    for b in agents[i+1:]:
                        p = self.p_intra if a.alcaldia == b.alcaldia else self.p_inter
                        if rng.random() < p:
                            self.G.add_edge(str(a.unique_id), str(b.unique_id))
            if len(agents) > 1 and self.G.number_of_edges() == 0:
                for i in range(len(agents)-1):
                    self.G.add_edge(str(agents[i].unique_id), str(agents[i+1].unique_id))

            degree = dict(self.G.degree())
            maxd = max(degree.values(), default=1)
            for a in agents:
                a.neighbor_ids = list(self.G.neighbors(str(a.unique_id)))
                a.influencia = degree.get(str(a.unique_id), 0) / maxd if maxd else 0

        def step(self):
            self.agents.do("step")
            self.agents.do("advance")
            for a in self.agents:
                a.fatiga *= .95
            self._collect_history()

        def _collect_history(self):
            self.history.append(self.metrics())

        @property
        def history(self):
            if not hasattr(self, "_history"):
                self._history = []
            return self._history

        def metrics(self):
            n = len(self.agents)
            return {
                "step": int(self.steps),
                "SIMPATIZANTE": self.count_spin(1),
                "OPOSITOR": self.count_spin(-1),
                "INDECISO": self.count_spin(0),
                "Gini": self.compute_gini(),
                "Polarizacion": self.compute_polarization(),
                "MeanOpinion": self.mean_opinion(),
                "edges": self.G.number_of_edges(),
                "density": nx.density(self.G) if n > 1 else 0,
            }

        def count_spin(self, v):
            n = len(self.agents)
            return 0 if n == 0 else sum(a.spin == v for a in self.agents) / n

        def mean_opinion(self):
            return float(np.mean([a.opinion for a in self.agents])) if len(self.agents) else 0

        def compute_polarization(self):
            vals = np.array([a.opinion for a in self.agents], dtype=float)
            return float(np.std(vals)) if len(vals) else 0

        def compute_gini(self):
            return gini([abs(a.opinion) for a in self.agents])

        def agent_dataframe(self):
            return pd.DataFrame([{
                "agent_id": str(a.unique_id),
                "territorial_unit_id": a.territorial_unit_id,
                "alcaldia": a.alcaldia,
                "seccion": a.seccion,
                "lat": a.lat,
                "lon": a.lon,
                "opinion": a.opinion,
                "spin": a.spin,
                "intencion": STATE_LABEL[a.spin],
                "poblacion": getattr(a, "population", getattr(a, "poblacion", 1000.0)),
                "temperatura": getattr(a, "temperatura", getattr(a, "exposure", 0.5)),
                "exposicion_problema": getattr(a, "exposure", 0.5),
                "capital_social": a.capital_social,
                "influencia": a.influencia,
                "saf_skill": a.saf_skill,
                "resistencia_institucional": a.resistencia_institucional,
                "prioridad_problema": a.prioridad_problema,
                "fatiga": a.fatiga,
                "broker": a.es_broker,
                "broker_density": a.broker_density,
                "org_mobilization": a.org_mobilization,
                "aceptacion_mensaje": a.aceptacion_mensaje,
                "org_reliability": a.org_reliability,
                "n_militantes_obs": a.n_militantes_obs,
            } for a in self.agents])

        def insert_broker(self, target_uid=None):
            if target_uid and target_uid in self.agent_by_uid:
                base = self.agent_by_uid[target_uid]
            else:
                base = max(self.agents, key=lambda x: x.influencia)
            row = {
                "territorial_unit_id": f"{base.territorial_unit_id}-BRK",
                "alcaldia": base.alcaldia,
                "seccion": base.seccion,
                "lat": base.lat,
                "lon": base.lon,
                "opinion_continua": base.opinion,
                "capital_social": .90,
                "acceso_informacion": .90,
                "influencia_liderazgo": .90,
                "arraigo": .85,
                "nivel_movilizacion": .85,
                "desconfianza": .10,
                "exposicion_problema": base.exposure,
                "resistencia_institucional": base.resistencia_institucional,
                "prioridad_problema": base.prioridad_problema,
            }
            b = BrokerAgent(self, row)
            self.agent_by_uid[str(b.unique_id)] = b
            self.G.add_node(str(b.unique_id), territorial_unit_id=b.territorial_unit_id,
                            alcaldia=b.alcaldia)
            candidates = sorted(
                [a for a in self.agents if a is not b and a.alcaldia == base.alcaldia],
                key=lambda x: x.influencia, reverse=True
            )[:8]
            for a in candidates:
                self.G.add_edge(str(b.unique_id), str(a.unique_id))
            b.neighbor_ids = [str(a.unique_id) for a in candidates]
            for a in candidates:
                if str(b.unique_id) not in a.neighbor_ids:
                    a.neighbor_ids.append(str(b.unique_id))
            b.influencia = 1.0
            return b

# ---------------------------------------------------------------------
# Capa de organización / militancia AGREGADA (sin PII)
# ---------------------------------------------------------------------
class OrganizationLayer:
    """Fusiona rasgos organizacionales agregados por territorial_unit_id.

    Nunca acepta identificadores personales. Celdas con n_militantes_obs, etc.
    """
    def __init__(self, df=None):
        if df is not None and not df.empty:
            self.df = df.copy()
            self.df.columns = [norm_col(c) for c in self.df.columns]
        else:
            self.df = pd.DataFrame(columns=["territorial_unit_id"] + ORG_COLS)

    def merge_into(self, base_df: pd.DataFrame) -> pd.DataFrame:
        if self.df.empty:
            return base_df.copy()
        if "territorial_unit_id" not in self.df.columns:
            return base_df.copy()
        
        out = base_df.copy()
        out = out.merge(self.df, on="territorial_unit_id", how="left")
        for col in ORG_COLS:
            if col in out.columns:
                out[col] = out[col].fillna(0.0)
        return out

# ---------------------------------------------------------------------
# MAIN STREAMLIT APP UI
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="SITER-CAE v6.1", layout="wide")
    st.title("SITER-CAE v6.1 DECISION + EVIDENCE LAB")
    st.markdown("Laboratorio territorial para CDMX - Sistema Empírico de Inteligencia Territorial")
    
    st.sidebar.header("Configuración de Datos")
    mode_name = st.sidebar.selectbox("Nivel de Evidencia / Modo", list(DataProvider.MODES.keys()))
    
    dp = DataProvider()
    
    df = None
    meta = {}
    
    if "DUMMY" in mode_name:
        df, meta = dp.dummy()
    elif "PURO" in mode_name:
        df, meta = dp.pure()
    elif "CALIBRACIÓN" in mode_name:
        df, meta = dp.calibration()
    else:
        st.info("Para los modos reales, por favor sube tus archivos de datos (CSV).")
        base_file = st.sidebar.file_uploader("CSV Base", type=["csv"])
        if base_file:
            df, meta = dp.real(base_file)
            
    if df is not None:
        st.write("### Vista Previa de Datos Territoriales")
        st.dataframe(df.head())
        
        st.write("### Parámetros de Simulación ABM")
        behavior = st.selectbox("Regla de Comportamiento (Dinámica Social)", list(BEHAVIORS.keys()))
        steps = st.slider("Pasos de Simulación", 1, 50, 10)
        
        if st.button("Ejecutar Simulación"):
            with st.spinner("Inicializando modelo Mesa..."):
                model = SITERModel(df, behavior_name=behavior)
                for i in range(steps):
                    model.step()
                
                history_df = pd.DataFrame(model.history)
                st.success(f"Simulación completada en {steps} pasos.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Evolución de Opinión y Simpatía")
                    fig = px.line(history_df, x="step", y=["SIMPATIZANTE", "OPOSITOR", "INDECISO"])
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.write("### Desigualdad y Polarización (Gini / Std)")
                    fig2 = px.line(history_df, x="step", y=["Polarizacion", "Gini"])
                    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    if MESA_OK:
        main()
    else:
        # Permite visualización de error si no están instaladas las dependencias localmente
        st.error(f"Error cargando dependencias (Mesa): {MESA_ERROR}")
        st.warning("Asegúrate de instalar las dependencias: pip install mesa streamlit pandas geopandas networkx plotly")
