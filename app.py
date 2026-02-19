# ============================================================================
# ФАЙЛ: app.py
# ПРИЛОЖЕНИЕ: Сборщик датасета судебных актов для Streamlit Cloud
# ВЕРСИЯ: 2.0 (с поддержкой JSONL и инкрементального обновления)
# ============================================================================

import streamlit as st
import pdfplumber
import json
import re
import os
import zipfile
import io
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional

# Импорт утилит (если вынесены в отдельные модули)
# from utils.pdf_extractor import extract_text_from_pdf, clean_text
# from utils.jsonl_handler import load_jsonl, save_jsonl, merge_datasets
# from utils.data_processor import extract_case_info_from_filename

# ============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ============================================================================
st.set_page_config(
    page_title="Сборщик датасета судебных актов",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS СТИЛИ
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .preview-box {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ФУНКЦИИ ОБРАБОТКИ ДАННЫХ
# ============================================================================

def clean_text(text: str) -> str:
    """Очищает текст от нечитаемых символов и артефактов PDF."""
    if not text:
        return ""
    
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    
    # Удаление непечатаемых символов
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines).strip()
    
    # Замена распространённых артефактов
    replacements = {
        'ﬁ': 'фи', 'ﬂ': 'фл', 'ﬀ': 'фф', 'ﬃ': 'ффи', 'ﬄ': 'ффл',
        '–': '-', '—': '-', '«': '"', '»': '"', '„': '"', '‚': "'",
        '′': "'", '″': '"', '…': '...', '•': '-', '©': '(c)',
        '®': '(R)', '™': '(TM)',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def extract_text_from_pdf(pdf_file) -> str:
    """Извлекает текст из PDF-файла с помощью pdfplumber."""
    text_parts = []
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- СТРАНИЦА {page_num} ---\n")
                    text_parts.append(page_text)
                    text_parts.append("\n\n")
        
        full_text = ''.join(text_parts)
        return clean_text(full_text)
    
    except Exception as e:
        st.error(f"Ошибка при извлечении текста: {str(e)}")
        return ""


def extract_case_info_from_filename(filename: str) -> dict:
    """Извлекает номер дела и дату решения из имени файла."""
    name_without_ext = Path(filename).stem
    parts = name_without_ext.split('_')
    
    result = {
        'case_number': None,
        'decision_date': None,
        'raw_filename': filename
    }
    
    if len(parts) >= 2:
        result['case_number'] = parts[0]
        date_str = parts[1]
        
        if len(date_str) == 8 and date_str.isdigit():
            try:
                dt = datetime.strptime(date_str, '%Y%m%d')
                result['decision_date'] = dt.strftime('%Y-%m-%d')
            except ValueError:
                result['decision_date'] = date_str
        else:
            result['decision_date'] = date_str
    
    return result


def create_jsonl_entry(case_number: str, decision_date: str, text: str) -> dict:
    """Создаёт запись формата JSONL для датасета."""
    return {
        "case_number": case_number,
        "decision_date": decision_date,
        "decision_text": text,
        "metadata": {
            "source": "arbitration_court",
            "document_type": "court_decision",
            "language": "ru",
            "created_at": datetime.now().isoformat()
        }
    }


def create_instruction_dataset_entry(case_number: str, decision_date: str, text: str) -> dict:
    """Создаёт запись для инструктивного датасета."""
    return {
        "instruction": f"Проанализируй судебный акт по делу № {case_number} от {decision_date}",
        "input": text[:2000],
        "output": f"Судебное решение по делу {case_number} от {decision_date}. Текст решения: {text[:3000]}..."
    }


# ============================================================================
# РАБОТА С JSONL (ЗАГРУЗКА / СОХРАНЕНИЕ / ОБЪЕДИНЕНИЕ)
# ============================================================================

def load_jsonl_dataset(uploaded_file) -> List[dict]:
    """Загружает датасет из JSONL файла."""
    entries = []
    try:
        content = uploaded_file.read().decode('utf-8')
        for line in content.strip().split('\n'):
            if line.strip():
                entries.append(json.loads(line))
        return entries
    except Exception as e:
        st.error(f"Ошибка загрузки JSONL: {e}")
        return []


def save_jsonl_dataset(entries: List[dict]) -> str:
    """Сохраняет датасет в формате JSONL и возвращает строку."""
    return '\n'.join(json.dumps(entry, ensure_ascii=False) for entry in entries)


def merge_datasets(existing: List[dict], new: List[dict]) -> List[dict]:
    """Объединяет два датасета, избегая дубликатов по номеру дела."""
    existing_cases = {e.get('case_number') for e in existing if e.get('case_number')}
    merged = existing.copy()
    
    for entry in new:
        if entry.get('case_number') not in existing_cases:
            merged.append(entry)
            existing_cases.add(entry.get('case_number'))
    
    return merged


def create_download_package(entries: List[dict], instruction_entries: List[dict] = None) -> bytes:
    """Создаёт ZIP-архив с датасетом для скачивания."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Основной датасет
        jsonl_content = save_jsonl_dataset(entries)
        zip_file.writestr('court_decisions_dataset.jsonl', jsonl_content.encode('utf-8'))
        
        # Инструктивный датасет
        if instruction_entries:
            instr_content = save_jsonl_dataset(instruction_entries)
            zip_file.writestr('instruction_dataset.jsonl', instr_content.encode('utf-8'))
        
        # README
        readme_content = f"""# Датасет судебных актов арбитражных судов

## Описание
Датасет содержит тексты судебных решений арбитражных судов России.

## Структура файлов
- `court_decisions_dataset.jsonl` - Основной датасет
- `instruction_dataset.jsonl` - Инструктивный датасет для Fine-tuning

## Статистика
- Всего записей: {len(entries)}
- Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Назначение
- Обучение LoRA-адаптеров для юридических LLM
- Fine-tuning моделей для анализа судебных решений
"""
        zip_file.writestr('README.md', readme_content.encode('utf-8'))
        
        # Статистика CSV
        if entries:
            df = pd.DataFrame([{
                'case_number': e.get('case_number', ''),
                'decision_date': e.get('decision_date', ''),
                'text_length': len(e.get('decision_text', ''))
            } for e in entries])
            csv_content = df.to_csv(index=False, encoding='utf-8-sig')
            zip_file.writestr('dataset_statistics.csv', csv_content.encode('utf-8'))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================================
# SESSION STATE ИНИЦИАЛИЗАЦИЯ
# ============================================================================
if 'dataset_entries' not in st.session_state:
    st.session_state.dataset_entries = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = None

# ============================================================================
# ЗАГОЛОВОК
# ============================================================================
st.markdown('<h1 class="main-header">⚖️ Сборщик датасета судебных актов</h1>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Подготовка данных для обучения LoRA и инструктивных датасетов | Streamlit Cloud</p>', 
            unsafe_allow_html=True)

# ============================================================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================================================
with st.sidebar:
    st.header("📊 Статистика датасета")
    
    st.metric("Всего записей", len(st.session_state.dataset_entries))
    st.metric("Обработано файлов", len(st.session_state.processed_files))
    
    if st.session_state.dataset_entries:
        dates = [e.get('decision_date', '') for e in st.session_state.dataset_entries if e.get('decision_date')]
        if dates:
            st.metric("Диапазон дат", f"{min(dates)} — {max(dates)}")
        
        total_chars = sum(len(e.get('decision_text', '')) for e in st.session_state.dataset_entries)
        st.metric("Общий объём текста", f"{total_chars:,} символов")
    
    st.divider()
    
    st.subheader("🛠 Управление")
    
    if st.button("🗑 Очистить датасет", use_container_width=True):
        st.session_state.dataset_entries = []
        st.session_state.processed_files = set()
        st.session_state.last_updated = None
        st.rerun()
    
    st.divider()
    
    st.subheader("ℹ️ О приложении")
    st.info("""
    **Версия:** 2.0  
    **Формат:** JSONL  
    **Назначение:** LoRA, Fine-tuning  
    **Деплой:** Streamlit Cloud + GitHub
    """)

# ============================================================================
# ОСНОВНАЯ ОБЛАСТЬ - ЗАГРУЗКА JSONL
# ============================================================================
st.header("📥 Загрузка существующего датасета (JSONL)")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_jsonl = st.file_uploader(
        "Загрузите предыдущий датасет в формате JSONL для продолжения работы",
        type=['jsonl', 'json'],
        help="Это позволит продолжить сбор датасета с того места, где вы остановились"
    )

with col2:
    if uploaded_jsonl:
        if st.button("📂 Загрузить датасет", use_container_width=True):
            entries = load_jsonl_dataset(uploaded_jsonl)
            if entries:
                st.session_state.dataset_entries = merge_datasets(
                    st.session_state.dataset_entries,
                    entries
                )
                st.session_state.last_updated = datetime.now().isoformat()
                st.success(f"✅ Загружено записей: {len(entries)}")
                st.rerun()

if st.session_state.dataset_entries:
    st.success(f"✅ В текущей сессии: {len(st.session_state.dataset_entries)} записей")

# ============================================================================
# ЗАГРУЗКА PDF ФАЙЛОВ
# ============================================================================
st.divider()
st.header("📤 Загрузка новых судебных актов (PDF)")

uploaded_files = st.file_uploader(
    "Загрузите PDF-файлы судебных решений",
    type=['pdf'],
    accept_multiple_files=True,
    help="Можно выбрать несколько файлов одновременно"
)

if uploaded_files:
    st.write(f"Выбрано файлов: **{len(uploaded_files)}**")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    new_entries = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name in st.session_state.processed_files:
            continue
        
        status_text.text(f"Обработка файла {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            file_info = extract_case_info_from_filename(uploaded_file.name)
            
            if not file_info['case_number'] or not file_info['decision_date']:
                st.warning(f"⚠️ Не удалось извлечь данные из имени: {uploaded_file.name}")
                continue
            
            pdf_text = extract_text_from_pdf(uploaded_file)
            
            if not pdf_text or len(pdf_text) < 100:
                st.warning(f"⚠️ Текст слишком короткий: {uploaded_file.name}")
                continue
            
            jsonl_entry = create_jsonl_entry(
                file_info['case_number'],
                file_info['decision_date'],
                pdf_text
            )
            
            new_entries.append(jsonl_entry)
            st.session_state.dataset_entries.append(jsonl_entry)
            st.session_state.processed_files.add(uploaded_file.name)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"❌ Ошибка обработки {uploaded_file.name}: {str(e)}")
            continue
    
    status_text.text("✅ Обработка завершена!")
    progress_bar.empty()
    
    if new_entries:
        st.success(f"✅ Успешно обработано файлов: {len(new_entries)}")
        st.session_state.last_updated = datetime.now().isoformat()

# ============================================================================
# ПРЕДПРОСМОТР ДАТАСЕТА
# ============================================================================
if st.session_state.dataset_entries:
    st.divider()
    st.header("👁 Предпросмотр датасета")
    
    selected_idx = st.selectbox(
        "Выберите запись для просмотра",
        range(len(st.session_state.dataset_entries)),
        format_func=lambda x: f"{st.session_state.dataset_entries[x]['case_number']} от {st.session_state.dataset_entries[x]['decision_date']}"
    )
    
    if selected_idx is not None:
        entry = st.session_state.dataset_entries[selected_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Метаданные")
            st.json({
                "Номер дела": entry['case_number'],
                "Дата решения": entry['decision_date'],
                "Длина текста": len(entry['decision_text']),
                "Создано": entry['metadata']['created_at'][:19]
            })
        
        with col2:
            st.subheader("📄 Текст решения (фрагмент)")
            preview_text = entry['decision_text'][:2000]
            st.text_area(
                "Содержимое",
                value=preview_text,
                height=400,
                label_visibility="collapsed"
            )
    
    # Таблица всех записей
    st.subheader("📊 Все записи в датасете")
    
    df_display = pd.DataFrame([
        {
            "№": idx + 1,
            "Номер дела": e['case_number'],
            "Дата решения": e['decision_date'],
            "Длина текста": len(e['decision_text']),
            "Статус": "✅"
        }
        for idx, e in enumerate(st.session_state.dataset_entries)
    ])
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# ============================================================================
# ЭКСПОРТ ДАТАСЕТА
# ============================================================================
if st.session_state.dataset_entries:
    st.divider()
    st.header("💾 Экспорт датасета")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        jsonl_content = save_jsonl_dataset(st.session_state.dataset_entries)
        
        st.download_button(
            label="📥 Скачать JSONL",
            data=jsonl_content.encode('utf-8'),
            file_name=f"court_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
            mime="text/jsonl",
            use_container_width=True
        )
    
    with col2:
        instruction_entries = [
            create_instruction_dataset_entry(
                e['case_number'], 
                e['decision_date'], 
                e['decision_text']
            )
            for e in st.session_state.dataset_entries
        ]
        
        instr_content = save_jsonl_dataset(instruction_entries)
        
        st.download_button(
            label="📥 Скачать Instruction Dataset",
            data=instr_content.encode('utf-8'),
            file_name=f"instruction_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
            mime="text/jsonl",
            use_container_width=True
        )
    
    with col3:
        zip_data = create_download_package(
            st.session_state.dataset_entries,
            instruction_entries
        )
        
        st.download_button(
            label="📦 Скачать ZIP-архив",
            data=zip_data,
            file_name=f"court_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    # Информация о форматах
    with st.expander("ℹ️ Описание форматов экспорта"):
        st.markdown("""
        ### 📄 JSONL (Основной датасет)
        - Формат: JSON Lines (одна JSON-запись на строку)
        - Поля: `case_number`, `decision_date`, `decision_text`, `metadata`
        - Кодировка: UTF-8
        
        ### 📚 Instruction Dataset
        - Формат: JSON Lines с полями `instruction`, `input`, `output`
        - Назначение: Fine-tuning LLM (LoRA, QLoRA)
        
        ### 📦 ZIP-архив
        - Включает основной датасет, инструктивный датасет, статистику и README
        """)

# ============================================================================
# ПОДВАЛ
# ============================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>⚖️ Сборщик датасета судебных актов | Версия 2.0 | Streamlit Cloud</p>
    <p>Данные предназначены для исследовательских целей</p>
    <p>Последнее обновление: {last_updated}</p>
</div>
""".format(last_updated=st.session_state.last_updated or "Неизвестно"), unsafe_allow_html=True)
