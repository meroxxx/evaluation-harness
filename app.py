import streamlit as st
import subprocess
import pandas as pd
import os


from API_main import calculate_metrics

st.set_page_config(page_title="ML Evaluation", layout="wide")
st.title("ML Model Evaluation")

# Боковая панель
with st.sidebar:
    st.header("Разделение данных")
    train_ratio = st.slider("Train ratio", 0.1, 0.9, 0.6, 0.05)
    
    if st.button("Запустить сплиттер"):
        with st.spinner("Разделение..."):
            r = subprocess.run(f"python data_splitter.py --train_ratio {train_ratio}", 
                              shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                st.success("Готово!")
            else:
                st.error(f"Ошибка: {r.stderr}")

# Основная область
tab1, tab2 = st.tabs(["Оценка модели", "Визуализация"])

with tab1:
    file = st.file_uploader("CSV с предсказаниями", type="csv")
    
    if file and st.button("Рассчитать метрики"):
        df = pd.read_csv(file)
        
       
        preds = []
        for _, row in df.iterrows():
            preds.append({
                "id": str(row.get("id", 0)),
                "predicted_score": float(row.get("score", 0.5)),
                "predicted_class": int(row.get("pred_class", 1 if row.get("score", 0.5) >= 0.5 else 0)),
                "true_label": int(row["true_label"])
            })
        
        try:
            
            metrics = calculate_metrics(preds)
            st.success("Метрики получены!")
            
            
            os.makedirs("outputs", exist_ok=True)
            pd.DataFrame([metrics]).to_csv("outputs/metrics_comparison.csv", index=False)
            
          
            cols = st.columns(4)
            for i, (k, v) in enumerate(metrics.items()):
                if isinstance(v, (int, float)) and i < 4:
                    cols[i].metric(k.upper(), f"{v:.4f}")
            
            st.json(metrics)
        except Exception as e:
            st.error(f"Ошибка: {e}")

with tab2:
    if st.button("Сгенерировать графики"):
        if os.path.exists("visualize_metrics.py"):
            with st.spinner("Генерация..."):
                subprocess.run("python visualize_metrics.py", shell=True)
                
                if os.path.exists("outputs/plots/dashboard.png"):
                    st.image("outputs/plots/dashboard.png")
                else:
                    st.info("Графики не найдены")
        else:
            st.error("Файл визуализатора не найден")
