import tkinter as tk
from tkinter import ttk
import numpy as np
import xgboost
import tensorflow as tf
import joblib


class App:
    def __init__(self, master):
        self.master = master
        master.title("Расчет характеристик материала")
        master.geometry("400x200")

        self.create_buttons()

    def create_buttons(self):
        btn1 = ttk.Button(self.master,
                          text="Модуль упругости при растяжении",
                          command=lambda: self.open_input_window("модуль упругости"))
        btn1.pack(pady=10)

        btn2 = ttk.Button(self.master,
                          text="Прочность при растяжении",
                          command=lambda: self.open_input_window("прочность"))
        btn2.pack(pady=10)

        btn3 = ttk.Button(self.master,
                          text="Матрица-наполнитель",
                          command=lambda: self.open_input_window("матрица"))
        btn3.pack(pady=10)

    def open_input_window(self, calculation_type):
        input_window = tk.Toplevel(self.master)
        input_window.title(f"Ввод параметров - {calculation_type}")
        input_window.geometry("480x500")

        parameters = [
            'Плотность, кг/м3',
            'модуль упругости, ГПа',
            'Количество отвердителя, м.%',
            'Содержание эпоксидных групп, %_2',
            'Температура вспышки, С_2',
            'Поверхностная плотность, г/м2',
            'Потребление смолы, г/м2',
            'Угол нашивки, град',
            'Шаг нашивки',
            'Плотность нашивки'
        ]

        self.entries = {}
        for param in parameters:
            frame = ttk.Frame(input_window)
            frame.pack(fill='x', padx=5, pady=5)

            label = ttk.Label(frame, text=param, width=35)
            label.pack(side='left')

            entry = ttk.Entry(frame)
            entry.pack(side='left', expand=True, fill='x')
            self.entries[param] = entry

        calculate_btn = ttk.Button(input_window, text="Рассчитать", command=lambda: self.calculate(calculation_type))
        calculate_btn.pack(pady=10)

    def calculate(self, calculation_type):
        values = [float(entry.get()) for entry in self.entries.values()]
        X = np.array(values).reshape(1, -1)

        if calculation_type == "модуль упругости":
            X = xgboost.DMatrix(X)
            xgb_modulus = joblib.load('models/xgb_modulus.joblib')
            y_pred = xgb_modulus.predict(X)[0]
            result = f"Модуль упругости при растяжении: {round(float(y_pred), 3)} ГПа"
        elif calculation_type == "прочность":
            X = xgboost.DMatrix(X)
            xgb_strength = joblib.load('models/xgb_strength.joblib')
            y_pred = xgb_strength.predict(X)[0]
            result = f"Прочность при растяжении: {round(float(y_pred), 3)} МПа"
        else:
            model = tf.keras.models.load_model('models/model.h5')
            y_pred = model.predict(X)[0]
            result = f"Соотношение матрица-наполнитель: {round(float(y_pred[0]), 3)}"

        result_window = tk.Toplevel(self.master)
        result_window.title("Результат")
        result_window.geometry("300x100")

        result_label = ttk.Label(result_window, text=result)
        result_label.pack(pady=20)


root = tk.Tk()
app = App(root)
root.mainloop()
