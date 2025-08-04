import os
import json
from typing import Dict, List
import streamlit as st
from annotated_text import annotated_text
import spacy

# --- Abhängigkeiten aus dem Originalskript ---
#from langchain_openai import ChatOpenAI
#from langchain.prompts import PromptTemplate
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, AnalysisExplanation
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine, OperatorConfig
from presidio_anonymizer.entities import OperatorResult, RecognizerResult
from presidio_anonymizer.operators import Operator, OperatorType

# --- Operator-Klassen (unverändert) ---
class InstanceCounterAnonymizer(Operator):
    REPLACING_FORMAT = "<{entity_type}_{index}>"
    def operate(self, text: str, params: Dict = None) -> str:
        entity_type: str = params["entity_type"]
        entity_mapping: Dict[str, Dict[str, str]] = params["entity_mapping"]
        entity_mapping_for_type = entity_mapping.get(entity_type, {})
        if text in entity_mapping_for_type:
            return entity_mapping_for_type[text]
        new_index = len(entity_mapping_for_type)
        new_placeholder = self.REPLACING_FORMAT.format(entity_type=entity_type, index=new_index)
        entity_mapping_for_type[text] = new_placeholder
        entity_mapping[entity_type] = entity_mapping_for_type
        return new_placeholder
    def validate(self, params: Dict = None) -> None:
        if "entity_mapping" not in params: raise ValueError("An input Dict called 'entity_mapping' is required.")
        if "entity_type" not in params: raise ValueError("An 'entity_type' param is required.")
    def operator_name(self) -> str: return "entity_counter"
    def operator_type(self) -> OperatorType: return OperatorType.Anonymize

class InstanceCounterDeanonymizer(Operator):
    def operate(self, text: str, params: Dict = None) -> str:
        entity_type: str = params["entity_type"]
        entity_mapping: Dict[str, Dict[str, str]] = params["entity_mapping"]
        if entity_type not in entity_mapping: raise ValueError(f"Entity type {entity_type} not found in entity mapping!")
        mapping_for_type = entity_mapping[entity_type]
        for original_value, placeholder in mapping_for_type.items():
            if placeholder == text: return original_value
        raise ValueError(f"Placeholder {text} not found in entity mapping for {entity_type}")
    def validate(self, params: Dict = None) -> None:
        if "entity_mapping" not in params: raise ValueError("An input Dict called 'entity_mapping' is required.")
        if "entity_type" not in params: raise ValueError("An 'entity_type' param is required.")
    def operator_name(self) -> str: return "entity_counter_deanonymizer"
    def operator_type(self) -> OperatorType: return OperatorType.Deanonymize


# --- Engine-Konfiguration (in Funktionen gekapselt und mit Caching) ---

# Streamlit's Caching, um die Modelle nur einmal zu laden
@st.cache_resource
def setup_gliner_only_analyzer():
    """Initialisiert den Analyzer NUR mit GLiNER."""
    st.info("Initialisiere 'Nur GLiNER'-Analyzer... (erster Start kann dauern)")
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_sm"}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry(supported_languages=["de"])

    STANDARD_PII_GLINER_MAP_DE = {
        "person": "PERSON", "name": "PERSON", "benutzername": "USERNAME",
        "sozialversicherungsnummer": "DE_SOCIAL_SECURITY_NUMBER", "versicherungsnummer": "ID_NUMBER",
        "identifikationsnummer": "ID_NUMBER", "steuer-id": "TAX_ID", "reisepassnummer": "PASSPORT_NUMBER",
        "führerscheinnummer": "DRIVERS_LICENSE", "ort": "LOCATION", "stadt": "LOCATION",
        "land": "LOCATION", "adresse": "ADDRESS", "straße": "ADDRESS", "postleitzahl": "ZIP_CODE",
        "telefonnummer": "PHONE_NUMBER", "handynummer": "PHONE_NUMBER", "e-mail-adresse": "EMAIL_ADDRESS",
        "kreditkartennummer": "CREDIT_CARD_NUMBER", "bankkontonummer": "BANK_ACCOUNT_NUMBER", "iban": "IBAN",
        "ip-adresse": "IP_ADDRESS", "mac-adresse": "MAC_ADDRESS", "webseite": "URL", "firma": "ORGANIZATION",
        "organisation": "ORGANIZATION", "unternehmen": "ORGANIZATION", "datum": "DATE_TIME",
        "uhrzeit": "DATE_TIME", "kundennummer": "ID_NUMBER"
    }

    gliner_recognizer = GLiNERRecognizer(
        model_name="urchade/gliner_multi_pii-v1",
        entity_mapping=STANDARD_PII_GLINER_MAP_DE
    )
    gliner_recognizer.supported_language = "de"
    registry.add_recognizer(gliner_recognizer)
    
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["de"]
    )
    return analyzer

@st.cache_resource
def setup_hybrid_analyzer():
    """Initialisiert den hybriden Analyzer (Presidio-Regeln + GLiNER)."""
    st.info("Initialisiere Hybrid-Analyzer... (erster Start kann dauern)")
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_sm"}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry(supported_languages=["de"])

    # Presidio's starke, musterbasierte Erkenner laden
    registry.load_predefined_recognizers(languages=["de"])

    # GLiNER *nur* für kontextabhängige Entitäten konfigurieren
    gliner_context_mapping = {
        "Person": "PERSON", "Firma": "ORGANIZATION", "Organisation": "ORGANIZATION",
        "Benutzername": "USERNAME", "Datum": "DATE_TIME", "Berufsbezeichnung": "TITLE",
    }
    gliner_recognizer = GLiNERRecognizer(
        model_name="urchade/gliner_multi_pii-v1", 
        entity_mapping=gliner_context_mapping
    )
    gliner_recognizer.supported_language = "de"
    registry.add_recognizer(gliner_recognizer)

    # Standard SpacyRecognizer entfernen, um Konflikte zu vermeiden
    try:
        registry.remove_recognizer("SpacyRecognizer", language="de")
    except ValueError:
        pass # Ignorieren, falls nicht vorhanden

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["de"]
    )
    return analyzer

@st.cache_resource
def setup_presidio_default_analyzer():
    """Initialisiert den Analyzer mit den Presidio-Standard-Erkennern (regelbasiert + Spacy NER)."""
    st.info("Initialisiere 'Nur Presidio (Standard)'-Analyzer...")
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_sm"}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry(supported_languages=["de"])

    # Lädt die Standard-Erkenner von Presidio, inkl. der Muster-Erkenner und des SpacyRecognizers
    registry.load_predefined_recognizers(languages=["de"])

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["de"]
    )
    return analyzer


@st.cache_resource
def get_spacy_model(model_name="de_core_news_sm"):
    """Lädt und gibt ein Spacy-Modell zurück."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        st.info(f"Spacy-Modell '{model_name}' nicht gefunden. Lade es herunter...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp

@st.cache_resource
def setup_anonymizer_deanonymizer():
    """Initialisiert Anonymizer und Deanonymizer."""
    anonymizer = AnonymizerEngine()
    anonymizer.add_anonymizer(InstanceCounterAnonymizer)
    deanonymizer = DeanonymizeEngine()
    deanonymizer.add_deanonymizer(InstanceCounterDeanonymizer)
    return anonymizer, deanonymizer

# --- Hilfsfunktionen für die UI ---

# Farb-Mapping für Entitätstypen für eine schönere Darstellung
ENTITY_COLORS = {
    "PERSON": "#FF6B6B",
    "LOCATION": "#4ECDC4",
    "ORGANIZATION": "#45B7D1",
    "EMAIL_ADDRESS": "#F7D154",
    "PHONE_NUMBER": "#F7B854",
    "CREDIT_CARD_NUMBER": "#FFA07A",
    "IBAN": "#E9967A",
    "DATE_TIME": "#98D8C8",
    "ID_NUMBER": "#CDB4DB",
    "TAX_ID": "#CDB4DB",
    "IP_ADDRESS": "#5E7CE2",
    "DEFAULT": "#F0F2F6" 
}

# Das ist jetzt wieder eine reine Anzeige-Komponente
def render_annotated_text(text: str, edited_results_with_ids: List[Dict]):
    """Baut eine HTML-Darstellung des annotierten Textes (nur zur Anzeige)."""
    
    # CSS für die Darstellung
    st.markdown("""
    <style>
    .annotated-text-container {
        line-height: 2.5;
        border: 1px solid #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .entity-span {
        padding: 0.3em 0.5em;
        margin: 0 0.25em;
        line-height: 1;
        display: inline-block;
        border-radius: 0.35em;
        border: 1px solid;
    }
    .entity-label {
        font-size: 0.7em;
        font-weight: bold;
        line-height: 1;
        border-radius: 0.35em;
        text-transform: uppercase;
        vertical-align: middle;
        margin-left: 0.5em;
    }
    </style>
    """, unsafe_allow_html=True)

    html_parts = []
    last_end = 0
    
    sorted_items = sorted(
        edited_results_with_ids, 
        key=lambda item: item['result'].start
    )

    for item in sorted_items:
        res = item['result']
        if res.start > last_end:
            html_parts.append(text[last_end:res.start])

        base_color_hex = ENTITY_COLORS.get(res.entity_type, ENTITY_COLORS["DEFAULT"])
        base_color_rgb = tuple(int(base_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        alpha = res.score * 0.8 + 0.2
        color_rgba = f"rgba({base_color_rgb[0]}, {base_color_rgb[1]}, {base_color_rgb[2]}, {alpha})"
        border_color_rgba = f"rgba({base_color_rgb[0]}, {base_color_rgb[1]}, {base_color_rgb[2]}, 0.5)"
        
        entity_html = (
            f'<span class="entity-span" style="background-color: {color_rgba}; border-color: {border_color_rgba};">'
            f'{text[res.start:res.end]}'
            f'<span class="entity-label">{res.entity_type} ({res.score:.2f})</span>'
            f'</span>'
        )
        html_parts.append(entity_html)
        last_end = res.end

    if last_end < len(text):
        html_parts.append(text[last_end:])
        
    final_html = "".join(html_parts)
    st.markdown(f'<div class="annotated-text-container">{final_html}</div>', unsafe_allow_html=True)

def remove_entity_by_id(entity_id_to_remove: str):
    """Callback-Funktion, die eine Entität aus der Liste im Session State entfernt."""
    st.session_state.edited_results = [
        item for item in st.session_state.edited_results if item['id'] != entity_id_to_remove
    ]

def process_and_deanonymize_simulation(text: str, edited_results_with_ids: List[Dict]) -> str:
    """
    Führt den Anonymisierungs- und Deanonymisierungs-Workflow aus,
    simuliert aber den LLM-Aufruf.
    """
    anonymizer, deanonymizer = setup_anonymizer_deanonymizer()
    
    analyzer_results = [item['result'] for item in edited_results_with_ids]

    # 1. Anonymisieren (bleibt gleich)
    entity_mapping = {}
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={"DEFAULT": OperatorConfig("entity_counter", {"entity_mapping": entity_mapping})}
    )
    
    st.session_state.anonymized_text = anonymized_result.text
    st.session_state.entity_mapping = entity_mapping
    
    # 2. LLM-Aufruf SIMULIEREN
    st.info("LLM-Aufruf wird simuliert. Es wird keine echte API-Anfrage gesendet.")
    
    simulated_llm_response = (
        f"Sehr geehrte Damen und Herren,\n\n"
        f"hier ist eine vom System bearbeitete Version des Textes:\n\n"
        f"--- Anfang des bearbeiteten Textes ---\n"
        f"{anonymized_result.text}\n"
        f"--- Ende des bearbeiteten Textes ---\n\n"
        f"Mit freundlichen Grüßen,\nIhr PII Redaction Assistant"
    )
    response = simulated_llm_response
    st.session_state.llm_response = response
    
    # 3. Deanonymisieren (bleibt gleich)
    entities_to_deanonymize = []
    for entity_type, mapping in entity_mapping.items():
        for _, placeholder in mapping.items():
            start_index = 0
            while placeholder in response[start_index:]:
                found_index = response.find(placeholder, start_index)
                if found_index == -1: break
                entities_to_deanonymize.append(
                    OperatorResult(start=found_index, end=found_index + len(placeholder), entity_type=entity_type)
                )
                start_index = found_index + 1

    deanonymized_result = deanonymizer.deanonymize(
        text=response,
        entities=entities_to_deanonymize,
        operators={"DEFAULT": OperatorConfig("entity_counter_deanonymizer", {"entity_mapping": entity_mapping})}
    )
    
    return deanonymized_result.text

# --- Streamlit App Hauptlogik ---

st.set_page_config(layout="wide", page_title="PII Redaction Assistant")

# Initialisiere den Session State, um Daten zwischen Interaktionen zu speichern
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.original_text = ""
    # Die Ergebnisse werden jetzt als Liste von Diktionären mit einer stabilen ID gespeichert
    st.session_state.analyzer_results = []
    st.session_state.edited_results = [] 
    st.session_state.processing_done = False
    st.session_state.anonymized_text = ""
    st.session_state.llm_response = ""
    st.session_state.entity_mapping = {}
    st.session_state.final_result = ""

# --- SCHRITT 1: Konfiguration und Texteingabe ---
st.header("Schritt 1: Konfiguration & Texteingabe")

analysis_mode = st.radio(
    "Wählen Sie den Analyse-Modus:",
    ("Hybrid (Empfohlen)", "Nur GLiNER", "Nur Presidio (Standard)"),
    key="analysis_mode",
    horizontal=True,
    help="**Hybrid:** Kombiniert Presidio-Regeln mit GLiNER für Kontext. **Nur GLiNER:** Ausschließlich KI-basiert. **Nur Presidio:** Regelbasiert plus allgemeines spaCy NER."
)

# Lade den passenden Analyzer basierend auf der Auswahl
if analysis_mode == "Nur GLiNER":
    analyzer = setup_gliner_only_analyzer()
elif analysis_mode == "Nur Presidio (Standard)":
    analyzer = setup_presidio_default_analyzer()
else: # Hybrid ist der Standard
    analyzer = setup_hybrid_analyzer()


# Lade Modelle und Engines
# analyzer = setup_analyzer() # Wird jetzt oben basierend auf Modus geladen
nlp = get_spacy_model()
sample_text = "..." # Fügen Sie hier Ihren langen Beispieltext ein, falls gewünscht

# Initialisiere den Session State, um Daten zwischen Interaktionen zu speichern
# -> Dieser Block wird nach oben verschoben

# --- SCHRITT 1: Texteingabe und Analyse ---
st.header("Schritt 1: Text analysieren")
input_text = st.text_area("Geben Sie hier Ihren Text ein:", height=250, key="input_text_area")

# Analyse-Button Logik
if st.button("PII analysieren", type="primary"):
    if input_text:
        with st.spinner("Analysiere Text... Dies kann einen Moment dauern."):
            st.session_state.original_text = input_text
            
            doc = nlp(input_text)
            all_results = []
            score_threshold = 0.3
            
            for sent in doc.sents:
                results_for_sent = analyzer.analyze(
                    text=sent.text,
                    language="de",
                    score_threshold=score_threshold
                )
                for res in results_for_sent:
                    res.start += sent.start_char
                    res.end += sent.start_char
                    all_results.append(res)
            
            # --- NEU: Ergebnisse in Struktur mit stabilen IDs umwandeln ---
            results_with_ids = []
            for i, res in enumerate(all_results):
                # Erstelle eine ID, die auch bei exakten Duplikaten (gleicher Typ, Start, Ende) eindeutig ist
                unique_id = f"{res.entity_type}-{res.start}-{res.end}-{i}"
                results_with_ids.append({'id': unique_id, 'result': res})

            st.session_state.analyzer_results = results_with_ids
            st.session_state.edited_results = list(results_with_ids) # Kopie für die Bearbeitung
            st.session_state.analysis_done = True
            st.session_state.processing_done = False 
            st.rerun() 
    else:
        st.warning("Bitte geben Sie einen Text ein.")

# --- SCHRITT 2: Überprüfung und Bearbeitung ---
if st.session_state.analysis_done:
    st.header("Schritt 2: Analyseergebnisse überprüfen und bearbeiten")
    
    st.subheader("Gefundene PIIs im Kontext")
    st.info("Dies ist eine schreibgeschützte Ansicht. Bearbeiten Sie die Liste unten.")
    render_annotated_text(st.session_state.original_text, st.session_state.edited_results)
    
    st.subheader("PII-Liste bearbeiten")
    st.warning("Klicken Sie auf [x], um eine falsch erkannte Entität zu entfernen. Die Ansicht wird sofort reaktiv aktualisiert.")

    for item in st.session_state.edited_results:
        res = item['result']
        unique_id = item['id']
        text_snippet = st.session_state.original_text[res.start:res.end]
        
        col1, col2, col3 = st.columns([4, 2, 1])
        with col1:
            st.markdown(f"**Text:** `{text_snippet}`")
        with col2:
            st.markdown(f"**Typ:** `{res.entity_type}` (Score: {res.score:.2f})")
        with col3:
            st.button("x", key=f"del_{unique_id}", on_click=remove_entity_by_id, args=(unique_id,), help="Diese Erkennung entfernen")

    # --- SCHRITT 3: Verarbeitung ---
    st.header("Schritt 3: Text bearbeiten (LLM-Aufruf simuliert)")
    if st.button("Anonymisieren und Deanonymisierung simulieren", type="primary"):
        with st.spinner("Text wird anonymisiert und deanonymisiert..."):
            final_text = process_and_deanonymize_simulation(st.session_state.original_text, st.session_state.edited_results)
            st.session_state.final_result = final_text
            st.session_state.processing_done = True
            st.rerun()

# --- Anzeige der finalen Ergebnisse ---
if st.session_state.processing_done:
    st.header("Ergebnisse des Prozesses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Anonymisierter Text (an LLM gesendet)")
        st.text_area("", value=st.session_state.anonymized_text, height=200, disabled=True)
        
        st.subheader("Entitäten-Mapping")
        st.json(st.session_state.entity_mapping)

    with col2:
        st.subheader("Antwort vom LLM (deanonymisiert)")
        st.success(st.session_state.final_result)
        
        st.subheader("Rohe Antwort vom LLM (mit Platzhaltern)")
        st.text_area("", value=st.session_state.llm_response, height=200, disabled=True)
