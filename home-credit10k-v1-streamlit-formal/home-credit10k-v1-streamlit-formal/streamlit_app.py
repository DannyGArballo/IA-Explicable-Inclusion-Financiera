from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_app.inference import (
    InferenceError,
    classify_decision,
    load_resources,
    predict_with_explanation,
    relative_level,
    scenario_values,
)

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "streamlit_app"
ARTIFACT_DIR = APP_DIR / "artifacts"
REPORT_DIR = ROOT / "reports" / "resultados"

SOURCE_LABELS = {
    "credit_bureau_a_1": "Buró de crédito · contratos",
    "credit_bureau_a_2": "Buró de crédito · historial detallado",
    "static_0": "Información consolidada de la solicitud",
    "static_cb_0": "Indicadores históricos consolidados",
    "applprev_1": "Solicitudes anteriores",
    "person_1": "Personas relacionadas",
    "debitcard_1": "Tarjetas de débito",
    "deposit_1": "Depósitos",
}

TRANSFORM_LABELS = {
    "row_count": "Conteo de registros",
    "nunique": "Cantidad de valores distintos",
    "mean": "Promedio",
    "std": "Desviación estándar",
    "range": "Rango (máximo menos mínimo)",
    "min": "Mínimo",
    "max": "Máximo",
    "first": "Primer registro procesado",
    "last": "Último registro procesado",
}

VARIABLE_DETAILS = {
    "credit_bureau_a_2__row_count": (
        "Resume cuántas filas históricas de pagos, contratos y movimientos existen para la solicitud. "
        "No equivale necesariamente al número de créditos, porque un mismo contrato puede generar varios registros."
    ),
    "credit_bureau_a_1__classificationofcontr_400M__nunique": (
        "Cuenta cuántas clasificaciones distintas aparecen entre los contratos del buró. "
        "Ayuda al modelo a distinguir historiales concentrados en un solo tipo de contrato de historiales más diversos."
    ),
    "credit_bureau_a_1__row_count": (
        "Indica la cantidad de registros de contratos asociados con la solicitud en la tabla de nivel contractual del buró."
    ),
    "credit_bureau_a_2__pmts_overdue_1140A__range": (
        "Mide la diferencia entre el mayor y el menor monto vencido observado. "
        "Un rango amplio refleja mayor dispersión entre los episodios de atraso."
    ),
    "static_cb_0__thirdquarter_1082L": (
        "Indicador histórico del tercer trimestre incluido en la tabla consolidada. "
        "Su significado operativo exacto depende del diccionario original de Home Credit; aquí se conserva como variable numérica del modelo."
    ),
    "static_0__mobilephncnt_593L": (
        "Cantidad de personas asociadas con el mismo número de teléfono móvil dentro de la información disponible."
    ),
    "credit_bureau_a_2__pmts_overdue_1140A__mean": (
        "Promedio de los montos vencidos observados en el historial detallado del buró."
    ),
    "static_0__numinstlallpaidearly3d_817L": (
        "Número de cuotas pagadas al menos tres días antes de su fecha de vencimiento."
    ),
    "static_0__pctinstlsallpaidlate1d_3546856L": (
        "Proporción de cuotas pagadas con uno o más días de atraso. "
        "Por ejemplo, 0.25 representa aproximadamente una cuarta parte de las cuotas."
    ),
    "static_cb_0__pmtssum_45A": (
        "Suma histórica de pagos registrada en la tabla consolidada. "
        "Se interpreta como una magnitud acumulada, no como una cuota individual."
    ),
    "static_0__pctinstlsallpaidearl3d_427L": (
        "Proporción de cuotas pagadas al menos tres días antes del vencimiento."
    ),
    "credit_bureau_a_2__pmts_overdue_1140A__std": (
        "Desviación estándar de los montos vencidos. "
        "Valores mayores indican un historial más irregular en la magnitud de los vencimientos."
    ),
}

st.set_page_config(
    page_title="Home Credit · Evaluación de Riesgo 10K",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
      .main-title {font-size: 2.25rem; font-weight: 750; margin-bottom: .15rem;}
      .subtitle {font-size: 1.05rem; opacity: .78; margin-bottom: 1.25rem;}
      .section-note {font-size: .91rem; opacity: .78;}
      .dictionary-card {padding: 1rem 1.1rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.22); margin-bottom: .8rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.20); padding: .75rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def resources():
    return load_resources(ARTIFACT_DIR)


@st.cache_data
def load_spanish_summary() -> dict:
    path = REPORT_DIR / "run_summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def feature_origin(feature: str) -> str:
    family = feature.split("__", 1)[0]
    return SOURCE_LABELS.get(family, family.replace("_", " ").title())


def feature_transformation(feature: str) -> str:
    parts = feature.split("__")
    if len(parts) < 3:
        return "Valor directo"
    return TRANSFORM_LABELS.get(parts[-1], parts[-1].replace("_", " ").title())


def format_reference(value: float, label: str, integer_like: bool) -> str:
    if integer_like:
        return f"{int(round(float(value))):,}"
    if "porcentaje" in label.lower() or "proporción" in label.lower():
        return f"{float(value):.1%}"
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


try:
    model, metadata = resources()
except InferenceError as exc:
    st.error(str(exc))
    st.stop()

spanish_summary = load_spanish_summary()

st.markdown('<div class="main-title">Home Credit · Evaluación de Riesgo 10K</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Herramienta interactiva basada en CatBoost. Ajusta las variables, consulta el índice de riesgo y revisa la explicación de cada resultado.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuración")
    threshold = st.slider(
        "Umbral de clasificación",
        min_value=0.05,
        max_value=0.95,
        value=float(metadata["default_threshold"]),
        step=0.01,
        help=(
            "Los puntajes inferiores al umbral se clasifican como favorables. "
            "El valor inicial se estimó en validación mediante el estadístico de Youden J."
        ),
    )
    st.caption(f"Umbral inicial: {metadata['default_threshold']:.3f}")
    scenario = st.selectbox(
        "Escenario inicial",
        ["Perfil típico", "Ejemplo de menor puntaje", "Ejemplo de mayor puntaje"],
        help="Aplica valores de referencia y después permite modificarlos manualmente.",
    )
    if st.button("Aplicar escenario", width="stretch"):
        scenario_row = scenario_values(metadata, scenario)
        for item in metadata["top_editable_features"]:
            feature = item["feature"]
            st.session_state[f"value::{feature}"] = scenario_row[feature]
            st.session_state[f"missing::{feature}"] = False
        st.rerun()

    st.divider()
    st.subheader("Modelo")
    st.write("CatBoost 10K")
    st.caption(f"Mejor iteración: {metadata['best_iteration']}")
    st.caption("250 variables seleccionadas con entrenamiento")
    st.caption("Partición temporal: periodos antiguos → entrenamiento; recientes → prueba")

sim_tab, model_tab, data_tab, dictionary_tab, guide_tab = st.tabs(
    ["Evaluación", "Resultados del modelo", "Datos", "Diccionario de variables", "Guía de uso"]
)

with sim_tab:
    top_features = metadata["top_editable_features"]
    primary = top_features[:8]
    advanced = top_features[8:]
    base_values = scenario_values(metadata, "Perfil típico")

    st.subheader("Variables principales")
    st.caption(
        "Los demás campos permanecen en valores de referencia calculados con entrenamiento. "
        "Activa “dato no disponible” para enviar un valor faltante al modelo."
    )

    values = base_values.copy()

    def render_feature(item: dict, container) -> None:
        feature = item["feature"]
        spec = metadata["feature_specs"][feature]
        with container:
            with st.container(border=True):
                st.markdown(f"**{spec['label']}**")
                st.caption(spec["description"])
                missing_key = f"missing::{feature}"
                value_key = f"value::{feature}"
                missing = st.toggle("Dato no disponible", key=missing_key)
                if spec["kind"] == "numeric":
                    low = float(spec["p01"])
                    high = float(spec["p99"])
                    if high <= low:
                        high = low + 1.0
                    default = float(spec["default"])
                    default = min(max(default, low), high)
                    if spec.get("integer_like"):
                        low_i, high_i = int(round(low)), int(round(high))
                        if high_i <= low_i:
                            high_i = low_i + 1
                        current = int(round(st.session_state.get(value_key, default)))
                        current = min(max(current, low_i), high_i)
                        chosen = st.slider(
                            "Valor",
                            min_value=low_i,
                            max_value=high_i,
                            value=current,
                            step=max(1, (high_i - low_i) // 100),
                            key=value_key,
                            disabled=missing,
                        )
                    else:
                        span = high - low
                        step = max(span / 200.0, 0.0001)
                        current = float(st.session_state.get(value_key, default))
                        current = min(max(current, low), high)
                        chosen = st.slider(
                            "Valor",
                            min_value=low,
                            max_value=high,
                            value=current,
                            step=step,
                            key=value_key,
                            disabled=missing,
                            format="%.4f" if abs(high) <= 2 else "%.2f",
                        )
                    values[feature] = None if missing else chosen
                else:
                    options = spec.get("options", [spec["default"]])
                    chosen = st.selectbox("Valor", options, key=value_key, disabled=missing)
                    values[feature] = None if missing else chosen
                st.caption(f"Patrón aprendido: {item['direction']}.")
                with st.expander("Detalle técnico"):
                    st.write(f"**Origen:** {feature_origin(feature)}")
                    st.write(f"**Transformación:** {feature_transformation(feature)}")
                    st.code(feature)

    col1, col2 = st.columns(2)
    for idx, item in enumerate(primary):
        render_feature(item, col1 if idx % 2 == 0 else col2)

    if advanced:
        with st.expander("Mostrar cuatro variables adicionales"):
            adv1, adv2 = st.columns(2)
            for idx, item in enumerate(advanced):
                render_feature(item, adv1 if idx % 2 == 0 else adv2)

    score, local_explanation = predict_with_explanation(model, metadata, values)
    level = relative_level(score, metadata["prediction_distribution_test"])
    decision, status = classify_decision(score, threshold)

    st.divider()
    st.subheader("Resultado")
    r1, r2, r3 = st.columns(3)
    r1.metric("Índice de riesgo", f"{score:.3f}")
    r2.metric("Umbral seleccionado", f"{threshold:.3f}")
    r3.metric("Posición relativa", level)
    st.progress(min(max(score, 0.0), 1.0), text=f"Índice relativo: {score:.1%}")

    if status == "success":
        st.success(f"**{decision}** — el índice se encuentra por debajo del umbral.")
    elif status == "warning":
        st.warning(f"**{decision}** — el índice se encuentra cerca del umbral.")
    else:
        st.error(f"**{decision}** — el índice supera el umbral.")

    st.caption(
        "La clasificación se obtiene comparando el índice con el umbral seleccionado; la resolución final puede incorporar política, documentación y revisión."
    )

    st.subheader("Factores que movieron el resultado")
    top_local = local_explanation.head(10).copy()
    chart = top_local[["variable", "contribucion_shap"]].set_index("variable").sort_values("contribucion_shap")
    st.bar_chart(chart, horizontal=True, height=420)
    st.caption(
        "Una contribución SHAP positiva eleva el índice; una contribución negativa lo reduce."
    )
    with st.expander("Ver detalle de contribuciones"):
        table = top_local[["variable", "variable_tecnica", "contribucion_shap"]].copy()
        table.columns = ["Variable", "Nombre técnico", "Contribución SHAP"]
        st.dataframe(table, width="stretch", hide_index=True)

with model_tab:
    st.subheader("Rendimiento del modelo")
    test = metadata["metrics"]["splits"]["test"]
    train = metadata["metrics"]["splits"]["train"]
    validation = metadata["metrics"]["splits"]["validation"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC en prueba", f"{test['roc_auc']:.3f}")
    c2.metric("Gini en prueba", f"{test['gini']:.3f}")
    c3.metric("Gini en validación", f"{validation['gini']:.3f}")
    c4.metric("Brecha entrenamiento–prueba", f"{metadata['metrics']['gaps']['train_minus_test_gini']:.3f}")

    metrics_df = pd.DataFrame(
        [
            {"Partición": "Entrenamiento", "AUC": train["roc_auc"], "Gini": train["gini"]},
            {"Partición": "Validación", "AUC": validation["roc_auc"], "Gini": validation["gini"]},
            {"Partición": "Prueba", "AUC": test["roc_auc"], "Gini": test["gini"]},
        ]
    )
    st.dataframe(metrics_df.style.format({"AUC": "{:.4f}", "Gini": "{:.4f}"}), width="stretch", hide_index=True)
    st.markdown(
        "**Lectura de resultados:** CatBoost conserva capacidad de ordenamiento en prueba. "
        "La diferencia entre entrenamiento y prueba muestra que el rendimiento es menor en periodos no vistos."
    )

    image_col1, image_col2 = st.columns(2)
    global_img = REPORT_DIR / "explicabilidad" / "catboost_global_importance_top30.png"
    shap_img = REPORT_DIR / "explicabilidad" / "catboost_shap_summary.png"
    if global_img.is_file():
        image_col1.image(str(global_img), caption="Importancia global de CatBoost", width="stretch")
    if shap_img.is_file():
        image_col2.image(str(shap_img), caption="Resumen SHAP global", width="stretch")

with data_tab:
    st.subheader("Conjunto de datos")
    dataset = metadata["dataset"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Solicitudes", f"{dataset['rows']:,}")
    d2.metric("Casos positivos", f"{dataset['positives']:,}")
    d3.metric("Tasa positiva", f"{dataset['positive_rate']:.2%}")
    d4.metric("Semanas", dataset["weeks"])

    st.subheader("Partición temporal")
    split_df = pd.DataFrame(metadata["split"]["splits"])
    split_df = split_df.rename(
        columns={
            "split": "Partición", "rows": "Filas", "positives": "Positivos", "negatives": "Negativos",
            "positive_rate": "Tasa positiva", "week_min": "Semana mínima", "week_max": "Semana máxima",
            "weeks": "Semanas", "both_classes": "Ambas clases",
        }
    )
    st.dataframe(split_df, width="stretch", hide_index=True)

    st.subheader("Variables más utilizadas por CatBoost")
    importance_df = pd.DataFrame(metadata["global_importance_top20"])
    importance_df = importance_df[["label", "importance", "feature"]]
    st.bar_chart(importance_df.set_index("label")[["importance"]].head(15), horizontal=True, height=520)
    with st.expander("Ver tabla completa"):
        glossary = importance_df.rename(
            columns={"label": "Variable", "importance": "Importancia", "feature": "Nombre técnico"}
        )
        st.dataframe(glossary, width="stretch", hide_index=True)

with dictionary_tab:
    st.subheader("Diccionario de variables editables")
    st.markdown(
        "Las variables de esta sección fueron seleccionadas por su impacto SHAP global. "
        "Cada ficha explica su procedencia, la transformación aplicada y los valores de referencia observados."
    )

    with st.expander("Cómo leer un nombre técnico"):
        st.code("familia_de_datos__columna_original__transformación")
        st.markdown(
            "- `row_count`: número de registros históricos.\n"
            "- `nunique`: número de valores distintos.\n"
            "- `mean`: promedio.\n"
            "- `std`: desviación estándar o variabilidad.\n"
            "- `range`: diferencia entre máximo y mínimo."
        )

    for item in metadata["top_editable_features"]:
        feature = item["feature"]
        spec = metadata["feature_specs"][feature]
        detail = VARIABLE_DETAILS.get(feature, spec["description"])
        with st.container(border=True):
            st.markdown(f"### {spec['label']}")
            st.write(detail)
            meta1, meta2, meta3 = st.columns(3)
            meta1.markdown(f"**Origen**  \n{feature_origin(feature)}")
            meta2.markdown(f"**Transformación**  \n{feature_transformation(feature)}")
            meta3.markdown(f"**Patrón aprendido**  \n{item['direction'].capitalize()}")

            ref1, ref2, ref3 = st.columns(3)
            ref1.metric("Percentil 25", format_reference(spec["p25"], spec["label"], bool(spec.get("integer_like"))))
            ref2.metric("Mediana", format_reference(spec["p50"], spec["label"], bool(spec.get("integer_like"))))
            ref3.metric("Percentil 75", format_reference(spec["p75"], spec["label"], bool(spec.get("integer_like"))))
            with st.expander("Nombre técnico"):
                st.code(feature)

with guide_tab:
    st.subheader("Cómo usar la herramienta")
    st.markdown(
        """
        1. Selecciona un escenario inicial o comienza con el perfil típico.
        2. Modifica las variables con los controles disponibles.
        3. Revisa el índice de riesgo y la clasificación respecto del umbral.
        4. Ajusta el umbral para comparar criterios de clasificación.
        5. Consulta las contribuciones SHAP para identificar qué variables elevaron o redujeron el índice.

        **Cómo interpretar la pantalla**

        - Un índice menor representa una posición de riesgo relativamente más favorable dentro del modelo.
        - El umbral transforma el índice continuo en una clasificación preliminar.
        - Los valores de referencia corresponden a la distribución observada en entrenamiento.
        - SHAP reparte el resultado entre las variables y muestra la dirección de cada contribución.
        """
    )

    st.subheader("Conceptos principales")
    concepts = pd.DataFrame(
        [
            {"Concepto": "Índice de riesgo", "Explicación": "Puntaje continuo producido por CatBoost para ordenar perfiles de menor a mayor riesgo."},
            {"Concepto": "Umbral", "Explicación": "Punto de corte utilizado para convertir el índice en una clasificación."},
            {"Concepto": "SHAP", "Explicación": "Método que cuantifica cuánto elevó o redujo cada variable el resultado del caso."},
            {"Concepto": "AUC", "Explicación": "Mide la capacidad de ordenar correctamente casos positivos y negativos."},
            {"Concepto": "Gini", "Explicación": "Transformación del AUC mediante Gini = 2 × AUC − 1."},
        ]
    )
    st.dataframe(concepts, width="stretch", hide_index=True)
