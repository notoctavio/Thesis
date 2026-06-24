from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CACHE_DIR = Path(tempfile.gettempdir()) / "licenta-rul-cache"
RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import artifacts, config, inference  # noqa: E402
from src.prediction import (  # noqa: E402
    available_batteries,
    available_models,
    cycle_summary,
    derived_rul_from_soh_curve,
    filter_predictions,
)
from src.reporting import (  # noqa: E402
    best_model_name,
    drop_naive_models,
    format_metric_row,
    metric_row,
    scenario_label,
    scenario_note,
)


PRIMARY_SCENARIO = "nasa_classic_4"
PRIMARY_BATTERY = "B0007"
PRIMARY_TASK = config.SOH_TASK
PRIMARY_DEMO_THRESHOLD = 0.80
MODEL_FAMILY_LABELS = {
    "baseline": "ML clasic",
    "sequence": "DL secvențial",
    "Classical ML": "ML clasic",
    "Sequence DL": "DL secvențial",
}


st.set_page_config(
    page_title="Li-ion Battery RUL",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.75rem;
            max-width: 1180px;
        }
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
        .thesis-note {
            border: 1px solid rgba(47, 111, 115, 0.42);
            border-left: 4px solid #2f6f73;
            background: rgba(47, 111, 115, 0.08);
            color: inherit;
            padding: 0.85rem 1rem;
            margin: 0.5rem 0 1rem 0;
            border-radius: 6px;
        }
        .muted-text {
            color: inherit;
            opacity: 0.78;
            font-size: 0.94rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(127, 127, 127, 0.08);
            border: 1px solid rgba(127, 127, 127, 0.22);
            border-radius: 8px;
            padding: 0.85rem;
            color: inherit;
        }
        div[data-testid="stMetric"] * {
            color: inherit;
        }
        .st-key-compact_metric_cards div[data-testid="stMetricLabel"] p,
        .st-key-csv_metric_cards div[data-testid="stMetricLabel"] p {
            font-size: 0.84rem;
            line-height: 1.2;
        }
        .st-key-compact_metric_cards div[data-testid="stMetricValue"],
        .st-key-csv_metric_cards div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
            line-height: 1.2;
            white-space: nowrap;
        }
        .st-key-compact_metric_cards div[data-testid="stMetricValue"] > div,
        .st-key-csv_metric_cards div[data-testid="stMetricValue"] > div {
            font-size: inherit;
            line-height: inherit;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_direct_predictions() -> pd.DataFrame:
    baseline = artifacts.load_baseline_predictions().assign(family="Classical ML")
    sequence = artifacts.load_sequence_predictions().assign(family="Sequence DL")
    common_cols = sorted(set(baseline.columns) | set(sequence.columns))
    return pd.concat(
        [baseline.reindex(columns=common_cols), sequence.reindex(columns=common_cols)],
        ignore_index=True,
    )


@st.cache_data(show_spinner=False)
def load_soh_predictions() -> pd.DataFrame:
    return artifacts.load_soh_all_predictions()


@st.cache_data(show_spinner=False)
def load_direct_metrics() -> pd.DataFrame:
    return artifacts.load_model_comparison()


@st.cache_data(show_spinner=False)
def load_soh_metrics() -> pd.DataFrame:
    return artifacts.load_soh_model_comparison()


@st.cache_data(show_spinner=False)
def load_scenarios() -> dict[str, Any]:
    return artifacts.load_scenarios()


def default_index(values: list[str], preferred: str) -> int:
    return values.index(preferred) if preferred in values else 0


def default_cycle(curve: pd.DataFrame) -> int:
    min_cycle = int(curve["cycle_index"].min())
    max_cycle = int(curve["cycle_index"].max())
    return int(round(min_cycle + 0.45 * (max_cycle - min_cycle)))


def metric_cards(
    metrics: dict[str, float | str],
    task: str,
    compact: bool = False,
    key: str = "compact_metric_cards",
) -> None:
    if compact:
        with st.container(key=key):
            _metric_cards(metrics, task)
        return

    _metric_cards(metrics, task)


def _metric_cards(metrics: dict[str, float | str], task: str) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", metrics["RMSE"])
    col2.metric("MAE", metrics["MAE"])
    col3.metric("R²", metrics["R2"])
    if task == config.SOH_TASK:
        st.caption("Pentru SOH, RMSE și MAE sunt erori în unități SOH. 0.01 înseamnă aproximativ 1% sănătate.")
    else:
        st.caption("Pentru RUL direct, RMSE și MAE sunt exprimate în cicluri rămase.")


def render_metric_explainer() -> None:
    st.caption(
        "RMSE: penalizează erorile mari; MAE: eroarea medie absolută; "
        "R²: cât de bine explică modelul variația datelor."
    )


def plot_soh(
    curve: pd.DataFrame,
    threshold: float,
    selected_cycle: int | None = None,
    figsize: tuple[float, float] = (10, 4.8),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(curve["cycle_index"], curve["soh"], label="SOH real", linewidth=2.2, color="#246b75")
    ax.plot(curve["cycle_index"], curve["pred_soh"], label="SOH prezis", linewidth=2.2, color="#c06b2d")
    ax.axhline(threshold, color="#9b2d30", linestyle="--", linewidth=1.6, label=f"Prag EOL {threshold:.0%}")
    if selected_cycle is not None:
        ax.axvline(selected_cycle, color="#56616f", linestyle=":", linewidth=1.5, label="Ciclu selectat")
    ax.set_xlabel("Ciclu de descărcare")
    ax.set_ylabel("SOH")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_inference_soh(curve: pd.DataFrame, threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if "soh" in curve.columns:
        ax.plot(curve["cycle_index"], curve["soh"], label="SOH real", linewidth=2.2, color="#246b75")
    ax.plot(curve["cycle_index"], curve["pred_soh"], label="SOH prezis", linewidth=2.2, color="#c06b2d")
    ax.axhline(threshold, color="#9b2d30", linestyle="--", linewidth=1.6, label=f"Prag EOL {threshold:.0%}")
    ax.set_xlabel("Ciclu de descărcare")
    ax.set_ylabel("SOH")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_direct_rul(curve: pd.DataFrame, selected_cycle: int | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(curve["cycle_index"], curve["rul_cycles"], label="RUL real", linewidth=2.2, color="#246b75")
    ax.plot(curve["cycle_index"], curve["prediction"], label="RUL prezis", linewidth=2.2, color="#c06b2d")
    if selected_cycle is not None:
        ax.axvline(selected_cycle, color="#56616f", linestyle=":", linewidth=1.5, label="Ciclu selectat")
    ax.set_xlabel("Ciclu de descărcare")
    ax.set_ylabel("Cicluri rămase")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def render_summary_table(summary: dict[str, object]) -> None:
    labels = {
        "battery_id": "Baterie",
        "cycle_index": "Ciclu analizat",
        "capacity_ah_clean": "Capacitate curățată (Ah)",
        "target": "Ținta modelului",
        "true_value": "Valoare reală",
        "predicted_value": "Valoare prezisă",
        "absolute_error": "Eroare absolută",
        "true_rul_cycles": "RUL real în date",
        "eol_cycle": "Ciclu EOL estimat",
        "remaining_cycles": "RUL derivat",
        "threshold": "Prag EOL",
        "status": "Status",
    }
    rows = []
    for key, value in summary.items():
        if key not in labels:
            continue
        if value is None:
            display_value = ""
        elif isinstance(value, float):
            display_value = f"{value:.4f}"
        else:
            display_value = str(value)
        rows.append(
            {
                "Câmp": labels[key],
                "Valoare": display_value,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_soh_story(
    selected: dict[str, Any],
    derived: dict[str, Any],
    scenario: str,
    model: str,
) -> None:
    predicted_soh = selected["predicted_value"]
    current_cycle = int(selected["cycle_index"])
    threshold = float(derived["threshold"])
    remaining = derived["remaining_cycles"]
    eol_cycle = derived["eol_cycle"]

    if remaining is None:
        rul_text = "modelul nu identifică în curba disponibilă o trecere clară sub pragul EOL"
    else:
        rul_text = f"RUL estimat este de aproximativ {remaining} cicluri, cu EOL la ciclul {eol_cycle}"

    st.markdown(
        f"""
        <div class="thesis-note">
        <strong>Interpretarea pentru comisie:</strong><br>
        În scenariul <strong>{scenario_label(scenario)}</strong>, pentru bateria
        <strong>{selected["battery_id"]}</strong>, modelul <strong>{model}</strong>
        estimează la ciclul <strong>{current_cycle}</strong> un SOH de
        <strong>{predicted_soh:.4f}</strong>. Pragul de sfârșit de viață este
        <strong>{threshold:.0%}</strong>, deci {rul_text}.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_direct_story(selected: dict[str, Any], scenario: str, model: str) -> None:
    st.markdown(
        f"""
        <div class="thesis-note">
        <strong>Interpretarea benchmark-ului direct RUL:</strong><br>
        În scenariul <strong>{scenario_label(scenario)}</strong>, pentru bateria
        <strong>{selected["battery_id"]}</strong>, modelul <strong>{model}</strong>
        estimează direct numărul de cicluri rămase. La ciclul
        <strong>{selected["cycle_index"]}</strong>, valoarea reală este
        <strong>{selected["true_value"]:.1f}</strong> cicluri, iar predicția este
        <strong>{selected["predicted_value"]:.1f}</strong> cicluri.
        </div>
        """,
        unsafe_allow_html=True,
    )


def ranked_metrics(metrics: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = drop_naive_models(metrics)
    if "split" in rows.columns:
        rows = rows.loc[rows["split"].eq("test")]
    rows = rows.loc[rows["scenario"].eq(scenario)].copy()
    if rows.empty:
        return rows
    rows = rows.sort_values("RMSE").reset_index(drop=True)
    rows.insert(0, "rank", range(1, len(rows) + 1))
    cols = ["rank", "model", "family", "RMSE", "MAE", "R2"]
    available_cols = [col for col in cols if col in rows.columns]
    return rows[available_cols]


def render_metrics_table(metrics: pd.DataFrame, scenario: str, title: str, note: str) -> None:
    st.subheader(title)
    st.caption(note)
    table = ranked_metrics(metrics, scenario)
    if table.empty:
        st.warning("Nu există metrici salvate pentru selecția curentă.")
        return
    display = table.copy()
    if "family" in display.columns:
        display["family"] = display["family"].map(MODEL_FAMILY_LABELS).fillna(display["family"])
    for col in ["RMSE", "MAE", "R2"]:
        if col in display.columns:
            display[col] = display[col].astype(float).round(4)
    display = display.rename(
        columns={
            "rank": "#",
            "model": "Model",
            "family": "Familie",
            "R2": "R²",
        }
    )
    best_row = display.iloc[0]
    st.success(
        f"Model recomandat după RMSE: {best_row['Model']} "
        f"(RMSE={float(best_row['RMSE']):.4f}, R²={float(best_row['R²']):.4f})."
    )
    st.dataframe(display, width="stretch", hide_index=True)
    render_metric_explainer()
    if "R²" in display.columns and (display["R²"].astype(float) < 0).any():
        st.warning(
            "Un R² negativ înseamnă că acel model a generalizat mai slab decât o "
            "predicție de bază pe split-ul selectat. Îl păstrăm în tabel pentru "
            "transparență experimentală, nu ca model recomandat."
        )

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(display["Model"], display["RMSE"], color="#5e7f89")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def scenario_table(scenarios: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, info in scenarios.items():
        split = info.get("split", {})
        rows.append(
            {
                "Scenariu": scenario_label(name),
                "Baterii": len(info.get("battery_ids", [])),
                "Antrenare": ", ".join(split.get("train_batteries", [])),
                "Validare": ", ".join(split.get("validation_batteries", [])),
                "Test": ", ".join(split.get("test_batteries", [])),
            }
        )
    return pd.DataFrame(rows)


def render_methodology(scenarios: dict[str, Any]) -> None:
    st.subheader("Workflow-ul proiectului")
    st.markdown(
        """
        1. Datasetul NASA este citit din varianta curățată CSV.
        2. Se păstrează ciclurile de descărcare și se extrag feature-uri pe ciclu.
        3. Se calculează două ținte: RUL direct și SOH.
        4. Datele sunt împărțite pe baterii, nu pe rânduri aleatorii, pentru a evita scurgerea de informație între antrenare și test.
        5. Se compară modele clasice cu LSTM și CNN-LSTM.
        6. Demo-ul încarcă artefactele salvate, iar tabul CSV folosește modele SOH publicate pentru inferență fără reantrenare.
        """
    )

    st.subheader("Scenarii de evaluare")
    st.dataframe(scenario_table(scenarios), width="stretch", hide_index=True)

    st.subheader("De ce RUL derivat din SOH?")
    st.markdown(
        """
        SOH este indicatorul de sănătate al bateriei și descrie degradarea
        capacității în timp. În literatura de specialitate, finalul vieții utile este definit
        printr-un prag EOL, de obicei raportat la capacitate sau SOH. Din acest
        motiv, abordarea principală a lucrării este: estimăm SOH, alegem pragul
        EOL, apoi calculăm RUL ca număr de cicluri până la atingerea pragului.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **RUL derivat din SOH**

            Modelul prezice SOH, adică starea de sănătate a bateriei. RUL se obține
            căutând ciclul în care curba SOH prezisă ajunge sub pragul EOL.
            Aceasta este formularea principală pentru demo. Modelele secvențiale
            SOH sunt antrenate pe schimbarea față de SOH-ul anterior, iar metricile
            sunt calculate pe SOH-ul reconstruit.
            """
        )
    with col2:
        st.markdown(
            """
            **RUL direct (benchmark)**

            Modelul prezice direct câte cicluri rămân până la finalul curbei
            înregistrate. Este util pentru comparația între modele, dar este mai
            sensibil la lungimea curbelor și la diferențele dintre baterii.
            """
        )

    st.info(
        "Scorurile foarte mari pe `nasa_classic_4` sunt corecte pentru acel subset, "
        "dar trebuie prezentate ca benchmark clasic NASA. Pentru generalizare se "
        "raportează separat `clean_benchmark` și `all_eligible`."
    )


def render_advanced_explorer(
    scenarios: list[str],
    direct_predictions: pd.DataFrame,
    soh_predictions: pd.DataFrame,
    direct_metrics: pd.DataFrame,
    soh_metrics: pd.DataFrame,
) -> None:
    st.subheader("Explorare avansată")
    st.caption(
        "Această zonă este pentru verificări tehnice; demo-ul principal rămâne RUL derivat din SOH."
    )

    task = st.radio(
        "Task",
        [config.SOH_TASK, config.DIRECT_RUL_TASK],
        horizontal=True,
        key="advanced_task",
    )
    scenario = st.selectbox(
        "Scenariu",
        scenarios,
        index=default_index(scenarios, "clean_benchmark"),
        format_func=scenario_label,
        key="advanced_scenario",
    )

    if task == config.DIRECT_RUL_TASK:
        predictions = direct_predictions
        metrics = direct_metrics
        models = available_models(predictions, scenario=scenario)
        recommended = best_model_name(metrics, scenario)
        true_col = "rul_cycles"
        pred_col = "prediction"
    else:
        predictions = soh_predictions
        metrics = soh_metrics
        models = available_models(predictions, scenario=scenario)
        recommended = best_model_name(metrics, scenario)
        true_col = "soh"
        pred_col = "pred_soh"

    if not models:
        st.warning("Nu există modele pentru selecția curentă.")
        return

    model = st.selectbox(
        "Model",
        models,
        index=default_index(models, recommended or models[0]),
        key="advanced_model",
    )
    batteries = available_batteries(predictions, scenario=scenario, model=model)
    battery_id = st.selectbox("Baterie", batteries, key="advanced_battery")
    curve = filter_predictions(predictions, scenario=scenario, model=model, battery_id=battery_id)
    min_cycle = int(curve["cycle_index"].min())
    max_cycle = int(curve["cycle_index"].max())
    selected_cycle = st.slider(
        "Ciclu curent",
        min_cycle,
        max_cycle,
        default_cycle(curve),
        key="advanced_cycle",
    )

    metrics_display = format_metric_row(metric_row(metrics, scenario, model))
    metric_cards(metrics_display, task)

    if task == config.DIRECT_RUL_TASK:
        st.pyplot(plot_direct_rul(curve, selected_cycle), width="stretch")
        selected = cycle_summary(curve, selected_cycle, task="rul")
        render_direct_story(selected, scenario, model)
        render_summary_table(selected)
    else:
        threshold = st.slider(
            "Prag EOL SOH",
            0.60,
            0.90,
            float(config.DEFAULT_EOL_THRESHOLDS.get(scenario, 0.80)),
            0.01,
            key="advanced_threshold",
        )
        st.pyplot(plot_soh(curve, threshold, selected_cycle), width="stretch")
        selected = cycle_summary(curve, selected_cycle, task="soh")
        derived = derived_rul_from_soh_curve(curve, selected["cycle_index"], threshold=threshold)
        render_soh_story(selected, derived, scenario, model)
        render_summary_table({**selected, **derived})

    with st.expander("Date filtrate"):
        display_cols = [
            "battery_id",
            "cycle_index",
            true_col,
            pred_col,
            "capacity_ah_clean",
            "scenario",
            "model",
            "family",
        ]
        available_cols = [col for col in display_cols if col in curve.columns]
        st.dataframe(curve[available_cols], width="stretch", height=320)


def render_csv_inference_tab() -> None:
    st.subheader("Predicție CSV pe date încărcate")
    st.caption(
        "Această zonă rulează un model SOH publicat în repository pe un CSV model-ready. "
        "Nu reantrenează modelele și nu descarcă datasetul Kaggle în timpul execuției."
    )

    manifest = inference.load_model_manifest()
    model_ids = inference.available_model_ids(manifest)
    default_model = manifest.get("default_model_id", model_ids[0])
    model_id = st.selectbox(
        "Model SOH publicat",
        model_ids,
        index=default_index(model_ids, default_model),
        format_func=lambda value: inference.get_model_info(value, manifest)["display_name"],
        key="csv_inference_model",
    )
    model_info = inference.get_model_info(model_id, manifest)
    threshold = st.slider(
        "Prag EOL SOH",
        0.60,
        0.90,
        float(model_info.get("default_threshold", 0.80)),
        0.01,
        key="csv_inference_threshold",
    )

    with st.expander("Formatul CSV așteptat"):
        feature_cols = inference.load_soh_feature_columns()
        st.markdown(
            """
            CSV-ul trebuie să fie la nivel de ciclu și să conțină feature-urile model-ready
            folosite la antrenarea modelelor SOH. Fișierul exemplu inclus în repository este:
            """
        )
        st.code(str(config.EXAMPLE_INFERENCE_CSV_PATH.relative_to(config.ROOT)))
        st.caption(f"Număr de feature-uri SOH necesare: {len(feature_cols)}.")
        st.dataframe(pd.DataFrame({"Coloane necesare": feature_cols}), width="stretch", height=220)

    uploaded = st.file_uploader(
        "Încarcă CSV model-ready",
        type=["csv"],
        key="csv_inference_uploader",
    )

    if uploaded is None:
        st.info("Nu a fost încărcat un fișier. Aplicația folosește exemplul public B0007 inclus în repository.")
        input_frame = inference.load_example_frame()
        source_label = "data/examples/b0007_soh_inference_example.csv"
    else:
        try:
            input_frame = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Fișierul CSV nu a putut fi citit: {exc}")
            return
        source_label = uploaded.name

    try:
        result = inference.predict_soh_from_frame(input_frame, model_id=model_id, threshold=threshold)
    except ValueError as exc:
        st.error(str(exc))
        missing = inference.missing_feature_columns(input_frame, inference.load_soh_feature_columns())
        if missing:
            st.dataframe(pd.DataFrame({"Coloane lipsă": missing}), width="stretch", hide_index=True)
        return

    st.markdown(
        f"""
        <div class="thesis-note">
        <strong>Predicție live:</strong><br>
        Sursa datelor este <strong>{source_label}</strong>, modelul selectat este
        <strong>{result.model_display_name}</strong>, iar pragul EOL este
        <strong>{result.threshold:.0%}</strong>. Aplicația estimează SOH pe fiecare ciclu
        și derivă RUL din prima trecere sub prag.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(4)
    cards[0].metric("Rânduri analizate", f"{len(result.predictions)}")
    cards[1].metric("Baterie", result.battery_id)
    cards[2].metric("Ciclu EOL estimat", "în afara curbei" if result.eol_cycle is None else str(result.eol_cycle))
    cards[3].metric(
        "RUL de la primul ciclu",
        "în afara curbei"
        if result.remaining_cycles_at_first_cycle is None
        else f"{result.remaining_cycles_at_first_cycle} cicluri",
    )

    if "abs_soh_error" in result.predictions.columns:
        error = result.predictions["abs_soh_error"].astype(float)
        y_true = result.predictions["soh"].astype(float)
        y_pred = result.predictions["pred_soh"].astype(float)
        sse = float(((y_true - y_pred) ** 2).sum())
        sst = float(((y_true - y_true.mean()) ** 2).sum())
        r2_value = 1.0 - (sse / sst) if sst > 0 else 0.0
        metric_cards(
            {
                "RMSE": f"{float((error.pow(2).mean()) ** 0.5):.4f}",
                "MAE": f"{float(error.mean()):.4f}",
                "R2": f"{r2_value:.4f}",
            },
            config.SOH_TASK,
            compact=True,
            key="csv_metric_cards",
        )

    st.pyplot(plot_inference_soh(result.predictions, result.threshold), width="stretch")

    display_cols = [
        "battery_id",
        "cycle_index",
        "soh",
        "pred_soh",
        "abs_soh_error",
        "rul_cycles",
        "eol_threshold",
        "model_id",
        "scenario",
    ]
    available_cols = [col for col in display_cols if col in result.predictions.columns]
    st.dataframe(result.predictions[available_cols], width="stretch", height=320)
    st.download_button(
        "Descarcă predicțiile CSV",
        data=result.predictions.to_csv(index=False).encode("utf-8"),
        file_name="soh_predictions.csv",
        mime="text/csv",
    )


def render_demo_controls(
    scenarios: list[str],
    soh_predictions: pd.DataFrame,
    soh_metrics: pd.DataFrame,
) -> dict[str, Any]:
    st.markdown("**Selecție demo**")
    st.caption("Selecția de mai jos controlează graficul, metricile și interpretarea RUL din acest tab.")

    scenario_col, model_col, battery_col = st.columns([1.35, 1.05, 0.8])
    with scenario_col:
        demo_scenario = st.selectbox(
            "Scenariu",
            scenarios,
            index=default_index(scenarios, PRIMARY_SCENARIO),
            format_func=scenario_label,
            key="demo_scenario_main",
        )

    demo_models = available_models(soh_predictions, scenario=demo_scenario)
    demo_recommended_model = best_model_name(soh_metrics, demo_scenario)
    with model_col:
        demo_model = st.selectbox(
            "Model SOH",
            demo_models,
            index=default_index(demo_models, demo_recommended_model or demo_models[0]),
            key=f"demo_model_main_{demo_scenario}",
        )

    demo_batteries = available_batteries(soh_predictions, demo_scenario, demo_model)
    with battery_col:
        demo_battery = st.selectbox(
            "Baterie test",
            demo_batteries,
            index=default_index(demo_batteries, PRIMARY_BATTERY),
            key=f"demo_battery_main_{demo_scenario}_{demo_model}",
        )

    demo_curve = filter_predictions(soh_predictions, demo_scenario, demo_model, demo_battery)
    demo_min_cycle = int(demo_curve["cycle_index"].min())
    demo_max_cycle = int(demo_curve["cycle_index"].max())

    cycle_col, threshold_col = st.columns([1.3, 1])
    with cycle_col:
        demo_cycle = st.slider(
            "Ciclu analizat",
            demo_min_cycle,
            demo_max_cycle,
            default_cycle(demo_curve),
            key=f"demo_cycle_main_{demo_scenario}_{demo_model}_{demo_battery}",
        )
    with threshold_col:
        demo_threshold = st.slider(
            "Prag EOL SOH",
            0.60,
            0.90,
            PRIMARY_DEMO_THRESHOLD
            if demo_scenario == PRIMARY_SCENARIO
            else float(config.DEFAULT_EOL_THRESHOLDS.get(demo_scenario, 0.80)),
            0.01,
            key=f"demo_threshold_main_{demo_scenario}",
        )

    st.caption(scenario_note(demo_scenario))
    if demo_recommended_model:
        st.caption(f"Model recomandat după RMSE pentru scenariul selectat: {demo_recommended_model}")
    if demo_scenario == PRIMARY_SCENARIO:
        st.caption(
            "Notă: NASA clasic este documentat cu prag 70%, dar demo-ul pornește pe 80% "
            "ca prag practic BMS/EV pentru a arăta RUL în curba disponibilă."
        )

    return {
        "scenario": demo_scenario,
        "model": demo_model,
        "battery": demo_battery,
        "curve": demo_curve,
        "cycle": demo_cycle,
        "threshold": demo_threshold,
    }


inject_style()

direct_predictions = load_direct_predictions()
soh_predictions = load_soh_predictions()
direct_metrics = load_direct_metrics()
soh_metrics = load_soh_metrics()
scenarios_info = load_scenarios()
scenarios = list(scenarios_info.keys())

st.title("Predicția duratei de viață utile rămase pentru baterii Li-ion")
st.caption(
    "Dataset: NASA Li-ion Battery Aging. Demo Streamlit pentru explicarea workflow-ului "
    "de data science: modele ML, SOH și RUL."
)
st.info(
    "Aplicația compară predicția directă RUL cu estimarea SOH urmată de derivarea RUL "
    "pe baza pragului EOL. Abordarea principală a lucrării este RUL derivat din SOH."
)

tab_demo, tab_csv, tab_results, tab_methodology, tab_advanced = st.tabs(
    ["Demo ghidat", "Predicție CSV", "Rezultate modele", "Metodologie", "Explorare avansată"]
)

with tab_demo:
    st.subheader("De la degradarea bateriei la RUL")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**1. Date**\n\nCicluri NASA de încărcare/descărcare pentru baterii Li-ion.")
    col2.markdown("**2. Model**\n\nModelul prezice SOH, adică sănătatea bateriei la fiecare ciclu.")
    col3.markdown("**3. RUL**\n\nRUL este numărul de cicluri până când SOH ajunge la pragul EOL.")

    st.markdown(
        """
        <div class="thesis-note">
        <strong>Abordare principală: SOH -> prag EOL -> RUL derivat.</strong><br>
        <strong>Mesajul principal:</strong> nu urmărim doar forma unei curbe,
        ci estimăm când bateria ajunge la sfârșitul duratei utile de viață.
        Pentru demo, SOH este prezis mai întâi, apoi RUL este derivat din pragul EOL.
        </div>
        """,
        unsafe_allow_html=True,
    )

    demo_selection = render_demo_controls(scenarios, soh_predictions, soh_metrics)
    demo_scenario = demo_selection["scenario"]
    demo_model = demo_selection["model"]
    demo_battery = demo_selection["battery"]
    demo_curve = demo_selection["curve"]
    demo_cycle = demo_selection["cycle"]
    demo_threshold = demo_selection["threshold"]

    left, right = st.columns([1.65, 1])
    with left:
        st.pyplot(plot_soh(demo_curve, demo_threshold, demo_cycle), width="stretch")
    with right:
        st.markdown(f"**Scenariu:** {scenario_label(demo_scenario)}")
        st.markdown(f"**Baterie test:** {demo_battery}")
        st.markdown(f"**Model:** {demo_model}")
        st.markdown(f"**Prag EOL:** {demo_threshold:.0%} SOH")
        demo_metric = format_metric_row(metric_row(soh_metrics, demo_scenario, demo_model))
        metric_cards(demo_metric, PRIMARY_TASK, compact=True)

    demo_selected = cycle_summary(demo_curve, demo_cycle, task="soh")
    demo_derived = derived_rul_from_soh_curve(
        demo_curve,
        demo_selected["cycle_index"],
        threshold=demo_threshold,
    )

    cards = st.columns(4)
    cards[0].metric("SOH real", f"{demo_selected['true_value']:.4f}")
    cards[1].metric("SOH prezis", f"{demo_selected['predicted_value']:.4f}")
    cards[2].metric("Eroare SOH", f"{demo_selected['absolute_error']:.4f}")
    if demo_derived["remaining_cycles"] is None:
        cards[3].metric("RUL derivat", "în afara curbei")
    else:
        cards[3].metric("RUL derivat", f"{demo_derived['remaining_cycles']} cicluri")

    render_soh_story(demo_selected, demo_derived, demo_scenario, demo_model)

    with st.expander("Detalii pentru ciclul selectat"):
        render_summary_table({**demo_selected, **demo_derived})

with tab_csv:
    render_csv_inference_tab()

with tab_results:
    result_scenario = st.selectbox(
        "Scenariu pentru comparația modelelor",
        scenarios,
        index=default_index(scenarios, "clean_benchmark"),
        format_func=scenario_label,
        key="results_scenario",
    )
    st.caption(scenario_note(result_scenario))

    render_metrics_table(
        soh_metrics,
        result_scenario,
        "Abordare principală: SOH / RUL derivat",
        "Aceasta este direcția recomandată: prezicem sănătatea bateriei și derivăm RUL din pragul EOL.",
    )
    st.divider()
    render_metrics_table(
        direct_metrics,
        result_scenario,
        "Benchmark comparativ: RUL direct",
        "Această abordare este păstrată pentru comparație. Rezultatele pot fi mai slabe deoarece ținta depinde de lungimea curbei și de eterogenitatea datasetului.",
    )

with tab_methodology:
    render_methodology(scenarios_info)

with tab_advanced:
    render_advanced_explorer(
        scenarios,
        direct_predictions,
        soh_predictions,
        direct_metrics,
        soh_metrics,
    )
