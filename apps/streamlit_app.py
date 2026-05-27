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

from src import artifacts, config  # noqa: E402
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
    "sequence": "DL secvential",
    "Classical ML": "ML clasic",
    "Sequence DL": "DL secvential",
}


st.set_page_config(
    page_title="Li-ion Battery RUL",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.75rem;
            max-width: 1180px;
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


def metric_cards(metrics: dict[str, float | str], task: str) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", metrics["RMSE"])
    col2.metric("MAE", metrics["MAE"])
    col3.metric("R2", metrics["R2"])
    if task == config.SOH_TASK:
        st.caption("Pentru SOH, RMSE si MAE sunt erori in unitati SOH. 0.01 inseamna aproximativ 1% sanatate.")
    else:
        st.caption("Pentru RUL direct, RMSE si MAE sunt exprimate in cicluri ramase.")


def render_metric_explainer() -> None:
    st.caption(
        "RMSE: penalizeaza erorile mari; MAE: eroarea medie absoluta; "
        "R2: cat de bine explica modelul variatia datelor."
    )


def plot_soh(curve: pd.DataFrame, threshold: float, selected_cycle: int | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(curve["cycle_index"], curve["soh"], label="SOH real", linewidth=2.2, color="#246b75")
    ax.plot(curve["cycle_index"], curve["pred_soh"], label="SOH prezis", linewidth=2.2, color="#c06b2d")
    ax.axhline(threshold, color="#9b2d30", linestyle="--", linewidth=1.6, label=f"Prag EOL {threshold:.0%}")
    if selected_cycle is not None:
        ax.axvline(selected_cycle, color="#56616f", linestyle=":", linewidth=1.5, label="Ciclu selectat")
    ax.set_xlabel("Ciclu de descarcare")
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
    ax.set_xlabel("Ciclu de descarcare")
    ax.set_ylabel("Cicluri ramase")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def render_summary_table(summary: dict[str, object]) -> None:
    labels = {
        "battery_id": "Baterie",
        "cycle_index": "Ciclu analizat",
        "capacity_ah_clean": "Capacitate curatata (Ah)",
        "target": "Tinta modelului",
        "true_value": "Valoare reala",
        "predicted_value": "Valoare prezisa",
        "absolute_error": "Eroare absoluta",
        "true_rul_cycles": "RUL real in date",
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
                "camp": labels[key],
                "valoare": display_value,
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
        rul_text = "modelul nu vede in curba disponibila o trecere clara sub pragul EOL"
    else:
        rul_text = f"RUL estimat este de aproximativ {remaining} cicluri, cu EOL la ciclul {eol_cycle}"

    st.markdown(
        f"""
        <div class="thesis-note">
        <strong>Interpretarea pentru comisie:</strong><br>
        In scenariul <strong>{scenario_label(scenario)}</strong>, pentru bateria
        <strong>{selected["battery_id"]}</strong>, modelul <strong>{model}</strong>
        estimeaza la ciclul <strong>{current_cycle}</strong> un SOH de
        <strong>{predicted_soh:.4f}</strong>. Pragul de sfarsit de viata este
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
        In scenariul <strong>{scenario_label(scenario)}</strong>, pentru bateria
        <strong>{selected["battery_id"]}</strong>, modelul <strong>{model}</strong>
        estimeaza direct numarul de cicluri ramase. La ciclul
        <strong>{selected["cycle_index"]}</strong>, valoarea reala este
        <strong>{selected["true_value"]:.1f}</strong> cicluri, iar predictia este
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
        st.warning("Nu exista metrici salvate pentru selectia curenta.")
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
        }
    )
    best_row = display.iloc[0]
    st.success(
        f"Model recomandat dupa RMSE: {best_row['Model']} "
        f"(RMSE={float(best_row['RMSE']):.4f}, R2={float(best_row['R2']):.4f})."
    )
    st.dataframe(display, width="stretch", hide_index=True)
    render_metric_explainer()
    if "R2" in display.columns and (display["R2"].astype(float) < 0).any():
        st.warning(
            "Un R2 negativ inseamna ca acel model a generalizat mai slab decat o "
            "predictie de baza pe split-ul selectat. Il pastram in tabel pentru "
            "transparenta experimentala, nu ca model recomandat."
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
                "scenariu": scenario_label(name),
                "baterii": len(info.get("battery_ids", [])),
                "antrenare": ", ".join(split.get("train_batteries", [])),
                "validare": ", ".join(split.get("validation_batteries", [])),
                "test": ", ".join(split.get("test_batteries", [])),
            }
        )
    return pd.DataFrame(rows)


def render_methodology(scenarios: dict[str, Any]) -> None:
    st.subheader("Workflow-ul proiectului")
    st.markdown(
        """
        1. Datasetul NASA este citit din varianta curatata CSV.
        2. Se pastreaza ciclurile de descarcare si se extrag feature-uri pe ciclu.
        3. Se calculeaza doua tinte: RUL direct si SOH.
        4. Datele sunt impartite pe baterii, nu pe randuri aleatorii, ca sa evitam scurgerea de informatie intre antrenare si test.
        5. Se compara modele clasice cu LSTM si CNN-LSTM.
        6. Demo-ul incarca artefactele salvate si explica predictiile fara reantrenare.
        """
    )

    st.subheader("Scenarii de evaluare")
    st.dataframe(scenario_table(scenarios), width="stretch", hide_index=True)

    st.subheader("De ce SOH-derived RUL?")
    st.markdown(
        """
        SOH este indicatorul de sanatate al bateriei si descrie degradarea
        capacitatii in timp. In literatura, finalul vietii utile este definit
        printr-un prag EOL, de obicei raportat la capacitate sau SOH. Din acest
        motiv, abordarea principala a lucrarii este: estimam SOH, alegem pragul
        EOL, apoi calculam RUL ca numar de cicluri pana la atingerea pragului.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **RUL derivat din SOH**

            Modelul prezice SOH, adica starea de sanatate a bateriei. RUL se obtine
            cautand ciclul in care curba SOH prezisa ajunge sub pragul EOL.
            Aceasta este formularea principala pentru demo. Modelele secventiale
            SOH sunt antrenate pe schimbarea fata de SOH-ul anterior, iar metricile
            sunt calculate pe SOH-ul reconstruit.
            """
        )
    with col2:
        st.markdown(
            """
            **RUL direct (benchmark)**

            Modelul prezice direct cate cicluri raman pana la finalul curbei
            inregistrate. Este util pentru comparatia intre modele, dar este mai
            sensibil la lungimea curbelor si la diferentele dintre baterii.
            """
        )

    st.info(
        "Scorurile foarte mari pe `nasa_classic_4` sunt corecte pentru acel subset, "
        "dar trebuie prezentate ca benchmark clasic NASA. Pentru generalizare se "
        "raporteaza separat `clean_benchmark` si `all_eligible`."
    )


def render_advanced_explorer(
    scenarios: list[str],
    direct_predictions: pd.DataFrame,
    soh_predictions: pd.DataFrame,
    direct_metrics: pd.DataFrame,
    soh_metrics: pd.DataFrame,
) -> None:
    st.subheader("Explorare avansata")
    st.caption(
        "Aceasta zona este pentru verificari tehnice; demo-ul principal ramane SOH-derived RUL."
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
        st.warning("Nu exista modele pentru selectia curenta.")
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


def render_demo_controls(
    scenarios: list[str],
    soh_predictions: pd.DataFrame,
    soh_metrics: pd.DataFrame,
) -> dict[str, Any]:
    st.markdown("**Selectie demo**")
    st.caption("Selectia de mai jos controleaza graficul, metricile si interpretarea RUL din acest tab.")

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
        st.caption(f"Model recomandat dupa RMSE pentru scenariul selectat: {demo_recommended_model}")
    if demo_scenario == PRIMARY_SCENARIO:
        st.caption(
            "Nota: NASA clasic este documentat cu prag 70%, dar demo-ul porneste pe 80% "
            "ca prag practic BMS/EV pentru a arata RUL in curba disponibila."
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

st.title("Predictia duratei de viata utile ramase pentru baterii Li-ion")
st.caption("Demo Streamlit pentru explicarea workflow-ului de data science: date NASA, modele ML, SOH si RUL.")
st.info(
    "Aplicatia compara predictia directa RUL cu estimarea SOH urmata de derivarea RUL "
    "pe baza pragului EOL. Abordarea principala a lucrarii este SOH-derived RUL."
)

st.sidebar.header("Demo recomandat")
st.sidebar.caption(
    "Demo-ul recomandat foloseste NASA clasic si RUL derivat din SOH. Controalele sunt in tabul Demo ghidat."
)

tab_demo, tab_results, tab_methodology, tab_advanced = st.tabs(
    ["Demo ghidat", "Rezultate modele", "Metodologie", "Explorare avansata"]
)

with tab_demo:
    st.subheader("De la degradarea bateriei la RUL")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**1. Date**\n\nCicluri NASA de incarcare/descarcare pentru baterii Li-ion.")
    col2.markdown("**2. Model**\n\nModelul prezice SOH, adica sanatatea bateriei la fiecare ciclu.")
    col3.markdown("**3. RUL**\n\nRUL este numarul de cicluri pana cand SOH ajunge la pragul EOL.")

    st.markdown(
        """
        <div class="thesis-note">
        <strong>Abordare principala: SOH -> prag EOL -> RUL derivat.</strong><br>
        <strong>Mesajul principal:</strong> nu incercam doar sa desenam o curba,
        ci estimam cand bateria ajunge la sfarsitul duratei utile de viata.
        Pentru demo, SOH este prezis mai intai, apoi RUL este derivat din pragul EOL.
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
        metric_cards(demo_metric, PRIMARY_TASK)

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
        cards[3].metric("RUL derivat", "in afara curbei")
    else:
        cards[3].metric("RUL derivat", f"{demo_derived['remaining_cycles']} cicluri")

    render_soh_story(demo_selected, demo_derived, demo_scenario, demo_model)

    with st.expander("Detalii pentru ciclul selectat"):
        render_summary_table({**demo_selected, **demo_derived})

with tab_results:
    result_scenario = st.selectbox(
        "Scenariu pentru comparatia modelelor",
        scenarios,
        index=default_index(scenarios, "clean_benchmark"),
        format_func=scenario_label,
        key="results_scenario",
    )
    st.caption(scenario_note(result_scenario))

    render_metrics_table(
        soh_metrics,
        result_scenario,
        "Abordare principala: SOH / RUL derivat",
        "Aceasta este directia recomandata: prezicem sanatatea bateriei si derivam RUL din pragul EOL.",
    )
    st.divider()
    render_metrics_table(
        direct_metrics,
        result_scenario,
        "Benchmark comparativ: RUL direct",
        "Aceasta abordare este pastrata pentru comparatie. Rezultatele pot fi mai slabe deoarece targetul depinde de lungimea curbei si de eterogenitatea datasetului.",
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
