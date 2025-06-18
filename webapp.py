import streamlit as st
import shap
import joblib
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm
from pathlib import Path

def main():
    best_model = joblib.load('best_model_lgbm.pkl')

    class Subject:
        def __init__(self, AGE, BMI, RACE, C4_0):
            self.RACE = RACE
            self.BMI = BMI
            self.AGE = AGE
            self.C4_0 = C4_0


        def make_predict(self):
            subject_data = {
                "RACE": [self.RACE],
                "BMI": [self.BMI],
                "AGE": [self.AGE],
                "C4_0": [self.C4_0]
            }

            # Create a DataFrame
            df_subject = pd.DataFrame(subject_data)

            # Make the prediction
            prediction = best_model.predict_proba(df_subject)[:, 1]
            adjusted_prediction = np.round(prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>The model predicts the prevalence of preserved ratio impaired spirometry is {adjusted_prediction} %</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            explainer = shap.Explainer(best_model)
            shap_values = explainer (df_subject)
            # 力图
            #shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
            shap.force_plot(                          # 兼容性调用
                shap_values.base_values[0],           # 取第 1 条样本的基线
                shap_values.values[0],                # 取对应 SHAP 向量
                df_subject.iloc[0, :],
                matplotlib=True
            )
            # 瀑布图
            # ex = shap.Explanation(shap_values[1][0, :], explainer.expected_value[1], df_subject.iloc[0, :])
            # shap.waterfall_plot(ex)
            st.pyplot(plt.gcf())

    st.set_page_config(page_title='the prevalence of preserved ratio impaired spirometry')
    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>Predicting the prevalence of preserved ratio impaired spirometry</h1>
                    <p class='intro'></p>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:14px; line-height:1.5; color:#6c757d;'>
        <b>Acknowledgments</b><br>
        Developed by: Deng <i>et&nbsp;al.</i><br>
        Contact: dengcy0758@163.com<br><br>

        Terms of Use
        This tool is for research use only.
        It is not intended for clinical diagnosis or treatment decision-making.
        The model was trained on individuals aged 20–79 years.

        Data Privacy
        No input data is stored on the server.
        All inputs are processed only temporarily and not used for any other purposes.
    </div>
    """, unsafe_allow_html=True)

    RACE = st.selectbox(
    "Race (1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White, 4=Non-Hispanic Black, 5=Other)",
    ("", 1, 2, 3, 4, 5),          # 空字符串作为占位
    index=0,
    key="race"
    )

    BMI = st.text_input("BMI (kg/m²)",      placeholder="Enter BMI",  key="bmi")
    AGE = st.text_input("Age (years)",       placeholder="Enter age", key="age")
    C4_0 = st.text_input("C4:0 FA intake (g/day)", placeholder="Enter value", key="c4_0")

    # ----------- 提交按钮 -----------
    if st.button("Submit"):
        if RACE and BMI and AGE and C4_0:
            user = Subject(float(AGE), float(BMI), int(RACE), float(C4_0))
            user.make_predict()
        else:
            st.warning("Please complete all fields before submitting.")

# ----------- 重置按钮 -----------
    if st.button("Reset all inputs"):
        for k in ("race", "bmi", "age", "c4_0"):
            st.session_state[k] = ""      # 清空各字段
        st.experimental_rerun()
main()
