import tkinter as tk
from tkinter import messagebox, Toplevel, Text, END
import os
import json
import random
from collections import Counter
from PIL import Image, ImageTk
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Try optional XGBoost, fall back gracefully if unavailable
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore


# --- PATHS AND CONFIGURATION ---
# Allow overriding via env var; otherwise use a local data directory next to this script
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "mega_data")
MODEL_DIR = os.environ.get("MEGA_MODEL_DIR", DEFAULT_DATA_DIR)

svm_model_path = os.path.join(MODEL_DIR, "svm_model.mega.pkl")
gb_model_path = os.path.join(MODEL_DIR, "gb_model.mega.pkl")
rf_model_path = os.path.join(MODEL_DIR, "rf_model.mega.pkl")
xgboost_model_path = os.path.join(MODEL_DIR, "xgboost_model.mega.pkl")
data_file = os.path.join(MODEL_DIR, "game_data.mega.json")
img_path = os.path.join(MODEL_DIR, "mega.png")

# Create directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# --- SEGMENT MAPPING AND PIE ZONES ---
segment_mapping = {
    0: "00M40", 1: "01M2", 2: "02M1", 3: "03M8", 4: "04M1", 5: "05M5", 6: "06M1", 7: "07M15",
    8: "08M2", 9: "09M1", 10: "10M10", 11: "11M1", 12: "12M5", 13: "13M1", 14: "14M20",
    15: "15M2", 16: "16M1", 17: "17M8", 18: "18M2", 19: "19M1", 20: "20M2", 21: "21M10",
    22: "22M1", 23: "23M2", 24: "24M5", 25: "25M1", 26: "26M2", 27: "27M30", 28: "28M1",
    29: "29M8", 30: "30M1", 31: "31M5", 32: "32M1", 33: "33M2", 34: "34M15", 35: "35M1",
    36: "36M5", 37: "37M2", 38: "38M10", 39: "39M1", 40: "40M2", 41: "41M20", 42: "42M1",
    43: "43M8", 44: "44M2", 45: "45M1", 46: "46M5", 47: "47M1", 48: "48M10", 49: "49M2",
    50: "50M1", 51: "51M5", 52: "52M2", 53: "53M1"
}

pie_zones = {
    "P1": ["01M2", "02M1", "03M8", "04M1", "05M5", "06M1", "07M15", "08M2", "09M1", "10M10", "11M1", "12M5", "13M1"],
    "P2": ["15M2", "16M1", "17M8", "18M2", "19M1", "20M2", "21M10", "22M1", "23M2", "24M5", "25M1", "26M2"],
    "P3": ["28M1", "29M8", "30M1", "31M5", "32M1", "33M2", "34M15", "35M1", "36M5", "37M2", "38M10", "39M1", "40M2"],
    "P4": ["42M1", "43M8", "44M2", "45M1", "46M5", "47M1", "48M10", "49M2", "50M1", "51M5", "52M2", "53M1"]
}

SPLIT_SEGMENTS = ["00M40", "14M20", "27M30", "41M20"]


# Helper functions
def get_pie_for_label(label: str) -> str:
    for pie, labels in pie_zones.items():
        if label in labels:
            return pie
    return "SPLIT"


def get_wheel_number(segment_number):
    if segment_number is None:
        return ""
    if isinstance(segment_number, int) and segment_number in [1, 2, 5, 8, 10, 15, 20, 30, 40]:
        return str(segment_number)
    if isinstance(segment_number, str) and 'M' in segment_number:
        return segment_number.split('M')[0]
    if isinstance(segment_number, int):
        label = segment_mapping.get(segment_number, "")
        if label and 'M' in label:
            return label.split('M')[0]
    return str(segment_number)


class MegaPredictorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mega Predictor - Cross-Platform")
        self.root.geometry("1600x900")
        self.root.configure(bg='#FFFFFF')

        # Initialize game data
        self.game_data = self.load_game_data()

        # Initialize state variables
        self.show_me_results_list = ["[SVM] [GB] [RF] [XG] [LAST RESULT] [WIN OR LOSS]"]
        self.pie_history_list = []
        self.history = []
        self.last_model_predictions = {}
        self.last_tarzan_prediction = ""
        self.total_entries = 0
        self.total_wins = 0
        self.last_result = None
        self.show_me_window = None

        # Dynamically set available model names based on optional deps
        self.available_model_names = ['SVM', 'GB', 'RF'] + (['XG'] if XGBClassifier is not None else [])

        # Initialize models
        self.models = {name: None for name in self.available_model_names}
        self.model_paths = {
            'SVM': svm_model_path,
            'GB': gb_model_path,
            'RF': rf_model_path,
            'XG': xgboost_model_path,
        }
        self.initialize_models()

        # Create UI
        self._create_widgets()
        self.load_image()

    def load_image(self) -> None:
        try:
            if os.path.exists(img_path):
                img = Image.open(img_path)
            else:
                # Create a placeholder if not present
                img = Image.new('RGB', (500, 500), color='white')
            img = img.resize((500, 500), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)
        except Exception as e:  # pragma: no cover
            print(f"Error loading image: {e}")
            blank_img = Image.new('RGB', (500, 500), color='white')
            self.photo = ImageTk.PhotoImage(blank_img)
            self.image_label.config(image=self.photo)

    def load_game_data(self):
        if not os.path.exists(data_file):
            return []
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def save_game_data(self) -> None:
        with open(data_file, 'w') as f:
            json.dump(self.game_data, f)

    def initialize_models(self) -> None:
        for name in self.available_model_names:
            model = None
            path = self.model_paths[name]
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                except Exception:
                    model = self.create_new_model(name)
            else:
                model = self.create_new_model(name)
            self.models[name] = model

    def create_new_model(self, name: str):
        if name == 'SVM':
            model = SVC(probability=True, random_state=42)
        elif name == 'GB':
            model = GradientBoostingClassifier(random_state=42)
        elif name == 'RF':
            model = RandomForestClassifier(random_state=42)
        elif name == 'XG' and XGBClassifier is not None:
            model = XGBClassifier(
                eval_metric='mlogloss',
                objective='multi:softprob',
                num_class=54,
                random_state=42,
            )
        else:  # Should not happen
            return None

        # Train with initial data if available
        if len(self.game_data) > 5:
            X, y = self.prepare_training_data()
            try:
                model.fit(X, y)
                joblib.dump(model, self.model_paths[name])
            except Exception as e:  # pragma: no cover
                print(f"Initial training failed for {name}: {e}")
        return model

    def prepare_training_data(self):
        X, y = [], []
        for i in range(len(self.game_data) - 5):
            X.append(self.game_data[i:i + 5])
            y.append(self.game_data[i + 5])

        if len(X) < 100:
            X = [[(i + j) % 54 for j in range(5)] for i in range(100)]
            y = [(i + 5) % 54 for i in range(100)]
        return np.array(X), np.array(y)

    def predict_next_numbers(self, model_name: str):
        if len(self.game_data) < 5:
            available_segments = list(segment_mapping.values())
            random.shuffle(available_segments)
            return available_segments[:2]
        try:
            last_5 = self.game_data[-5:]
            X = np.array([last_5])
            probs = self.models[model_name].predict_proba(X)[0]
            top_indices = np.argsort(probs)[-2:][::-1]
            predictions = [segment_mapping[i] for i in top_indices]
            return predictions
        except Exception as e:  # pragma: no cover
            print(f"Prediction error with {model_name}: {e}")
            available_segments = list(segment_mapping.values())
            random.shuffle(available_segments)
            return available_segments[:2]

    def _create_widgets(self) -> None:
        # Main container
        main_container = tk.Frame(self.root, bg='#FFFFFF')
        main_container.pack(fill='both', expand=True)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = tk.Frame(main_container, bg='#000000')
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        tk.Label(
            header_frame,
            text="MEGAWHEEL Predictor",
            font=("Courier New", 24, "bold"),
            bg='#000000', fg='#00FFFF'
        ).pack(pady=10)

        # Left Column
        left_column = tk.Frame(main_container, bg='#FFFFFF')
        left_column.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        self.image_label = tk.Label(left_column, bg='#FFFFFF')
        self.image_label.pack(pady=20)

        # Right Column
        right_column = tk.Frame(main_container, bg='#FFFFFF')
        right_column.grid(row=1, column=1, sticky='nsew', padx=10, pady=10)
        right_column.grid_rowconfigure(0, weight=1)
        right_column.grid_rowconfigure(1, weight=1)
        right_column.grid_rowconfigure(2, weight=1)
        right_column.grid_columnconfigure(0, weight=1)

        # Monitors
        top_container = tk.Frame(right_column, bg='#FFFFFF')
        top_container.grid(row=0, column=0, sticky='ew', pady=10)

        shana_monitor_frame = tk.Frame(top_container, bg='#FFFFFF')
        shana_monitor_frame.pack(side='left', fill='both', expand=True, padx=5)
        tk.Label(
            shana_monitor_frame,
            text="Shana Pie Monitor",
            font=("Courier New", 16, "bold"),
            bg='#FFFFFF', fg='#000000'
        ).pack()
        self.shana_monitor_text = Text(
            shana_monitor_frame, height=10, width=40, bg="#FFFFFF", fg="#000000",
            wrap=tk.WORD, state='disabled', font=("Courier New", 12, 'bold')
        )
        self.shana_monitor_text.pack(fill='both', expand=True)

        pie_history_frame = tk.Frame(top_container, bg='#000000')
        pie_history_frame.pack(side='left', fill='both', expand=True, padx=5)
        tk.Label(
            pie_history_frame,
            text="Pie History",
            font=("Courier New", 16, 'bold'),
            bg='#000000', fg='#00FFFF'
        ).pack()
        self.pie_history_text = Text(
            pie_history_frame, height=10, width=30, bg="#000000", fg="#00FFFF",
            wrap=tk.WORD, state='disabled', font=("Courier New", 12)
        )
        self.pie_history_text.pack(fill='both', expand=True)

        # Predictions
        prediction_frame = tk.Frame(right_column, bg='#FFFFFF')
        prediction_frame.grid(row=1, column=0, sticky='ew', pady=20)

        # Last Result Label
        self.last_result_label = tk.Label(
            prediction_frame, text="LAST RESULT: --",
            font=("Courier New", 16, 'bold'), bg='#FFFFFF', fg='#0000FF'
        )
        self.last_result_label.pack(anchor='w', pady=(0, 10))

        # Tarzan Prediction
        self.tarzan_label = tk.Label(
            prediction_frame, text="TARZAN Predicts: --",
            font=("Courier New", 16, 'bold'), bg='#FFFFFF', fg='#FF0000'
        )
        self.tarzan_label.pack(anchor='w', pady=(0, 10))

        # Model Predictions
        self.model_prediction_labels = {}
        for name in self.available_model_names:
            frame = tk.Frame(prediction_frame, bg='#FFFFFF')
            frame.pack(anchor='w', fill='x', pady=2)
            label = tk.Label(
                frame, text=f"{name}: ", font=("Courier New", 14, 'bold'),
                bg='#FFFFFF', fg='#000000'
            )
            label.pack(side='left')

            text_widget = Text(
                frame, height=1, width=30, font=("Courier New", 14),
                bg='#FFFFFF', fg='#000000', bd=0, highlightthickness=0, wrap='none'
            )
            text_widget.pack(side='left', fill='x', expand=True)
            text_widget.tag_configure('bold', font=("Courier New", 14, 'bold'))
            text_widget.tag_configure('normal', font=("Courier New", 14))
            text_widget.config(state='disabled')

            self.model_prediction_labels[name] = {
                'label': label,
                'text': text_widget,
                'frame': frame
            }

        # Controls
        control_frame = tk.Frame(right_column, bg='#FFFFFF')
        control_frame.grid(row=2, column=0, sticky='ew', pady=10)
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)

        self.entry_box = tk.Entry(
            control_frame, width=4, font=("Courier New", 16, 'bold'),
            bg='#333333', fg='#FFFFFF', justify='center'
        )
        self.entry_box.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.entry_box.bind('<Return>', self.handle_enter)

        tk.Button(
            control_frame, text='ENTER', font=("Courier New", 14, 'bold'), width=10, height=2,
            bg='#FFFF00', fg='#000000', command=self.handle_enter
        ).grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        tk.Button(
            control_frame, text="SHOW ME", font=("Courier New", 14, 'bold'), width=10, height=2,
            bg='#00FFFF', fg='#000000', command=self.show_me_results
        ).grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        tk.Button(
            control_frame, text='RETRAIN', font=("Courier New", 14, 'bold'), width=10, height=2,
            bg='#000000', fg='#FF4500', command=self.retrain_models
        ).grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        tk.Button(
            control_frame, text='RESET', font=("Courier New", 14, 'bold'), width=10, height=2,
            bg='#FFFF00', fg='#000000', command=self.reset_session
        ).grid(row=1, column=2, padx=5, pady=5, sticky='ew')

        # Footer
        footer_frame = tk.Frame(self.root, bg='#000000')
        footer_frame.pack(side='bottom', fill='x')
        tk.Label(
            footer_frame, text="Mega Wheel Predictor Systems",
            font=("Courier New", 10), fg='#888888', bg='#000000'
        ).pack(pady=5)

    def handle_enter(self, event=None) -> None:
        result_str = self.entry_box.get().strip()
        if result_str.isdigit() and 0 <= int(result_str) <= 53:
            result_int = int(result_str)
            self.game_data.append(result_int)
            self.save_game_data()
            self.total_entries += 1

            self.last_result = result_int
            wheel_result = get_wheel_number(result_int)
            self.last_result_label.config(text=f"LAST RESULT: {wheel_result}")

            self.entry_box.delete(0, tk.END)

            # Make predictions and update displays
            self.predict_all_models()
            self.check_win_loss(result_int)

            # Compute consensus predicted pie and update pie history
            actual_segment = segment_mapping.get(result_int, "")
            actual_pie = get_pie_for_label(actual_segment)
            consensus_pie = self.compute_consensus_predicted_pie()
            if consensus_pie is None:
                consensus_pie = "--"
            self.pie_history_list.insert(0, f"{consensus_pie} - {actual_pie}")
            if len(self.pie_history_list) > 50:
                self.pie_history_list = self.pie_history_list[:50]

            self.update_shana_pie_monitor_by_zone()
            self.update_pie_history_display()

            # Retrain models with new data periodically
            if len(self.game_data) % 10 == 0:
                self.retrain_models()
        else:
            messagebox.showwarning("Entry Error", "Please enter a valid number between 0 and 53.")

    def compute_consensus_predicted_pie(self):
        zone_counts = Counter()
        for name in self.available_model_names:
            preds = self.last_model_predictions.get(name, []) or []
            for label in preds:
                pie = get_pie_for_label(label)
                if pie in pie_zones:
                    zone_counts[pie] += 1
        if not zone_counts:
            return None
        return zone_counts.most_common(1)[0][0]

    def predict_all_models(self) -> None:
        self.last_model_predictions.clear()
        for name in self.available_model_names:
            predictions = self.predict_next_numbers(name)
            self.last_model_predictions[name] = predictions

        # SHANA Logic - predict based on most common recent pie zone
        if len(self.game_data) >= 10:
            last_10_results = self.game_data[-10:]
            pie_zones_seen = [get_pie_for_label(segment_mapping.get(num, "")) for num in last_10_results]
            if pie_zones_seen:
                most_common_pie = max(set(pie_zones_seen), key=pie_zones_seen.count)
                pie_segments = pie_zones.get(most_common_pie, [])
                if pie_segments:
                    random.shuffle(pie_segments)
                    self.last_model_predictions['SHANA'] = pie_segments[:2]

        # TARZAN Logic - random pie zone prediction
        self.last_tarzan_prediction = random.choice(list(pie_zones.keys()))
        self.tarzan_label.config(text=f"TARZAN Predicts: {self.last_tarzan_prediction}")

        self.update_prediction_displays()
        self.update_shana_pie_monitor_by_zone()

    def check_win_loss(self, last_result_int: int) -> None:
        if not self.last_model_predictions:
            return
        actual_segment = segment_mapping.get(last_result_int, "")
        actual_pie = get_pie_for_label(actual_segment)

        win_status = {}
        for name in self.available_model_names:
            predictions = self.last_model_predictions.get(name, [])
            win_status[name] = False
            for pred in predictions:
                pred_pie = get_pie_for_label(pred)
                if pred_pie == actual_pie:
                    win_status[name] = True
                    break

        display_predictions = []
        for name in self.available_model_names:
            preds = self.last_model_predictions.get(name, ['--', '--'])
            formatted = [f"{pred.split('M')[0]}-{pred.split('M')[1]}" for pred in preds if pred and pred != '--']
            display_predictions.append(','.join(formatted) if formatted else '--')

        wheel_result = get_wheel_number(last_result_int)
        is_win = any(win_status.values())
        status = "WIN" if is_win else "LOSS"
        if is_win:
            self.total_wins += 1

        result_line = ' '.join([f'[{pred}]' for pred in display_predictions]) + f' [{wheel_result}] [{status}]'
        self.show_me_results_list.append(result_line)

        win_rate = (self.total_wins / self.total_entries * 100) if self.total_entries > 0 else 0
        self.root.title(f"Mega Predictor - Wins: {self.total_wins}/{self.total_entries} ({win_rate:.1f}%)")
        self.root.after(5000, self.clear_last_result)

    def update_shana_pie_monitor_by_zone(self) -> None:
        self.shana_monitor_text.config(state='normal')
        self.shana_monitor_text.delete(1.0, END)

        all_predicted_labels = set()
        for name in self.available_model_names:
            if name in self.last_model_predictions:
                all_predicted_labels.update(self.last_model_predictions[name])

        self.shana_monitor_text.tag_configure("green", foreground="#00FF00",
                                              font=("Courier New", 10, 'bold'))
        self.shana_monitor_text.tag_configure("red", foreground="#FF0000",
                                              font=("Courier New", 10, 'bold'))
        self.shana_monitor_text.tag_configure("zone_header", foreground="#000000",
                                              font=("Courier New", 12, 'bold'))

        for zone in sorted(pie_zones.keys()):
            self.shana_monitor_text.insert(END, f"{zone}: ", "zone_header")
            zone_labels = pie_zones[zone]
            multipliers_and_labels = sorted([(int(label.split('M')[1]), label) for label in zone_labels])
            for i, (multiplier, label) in enumerate(multipliers_and_labels):
                tag = "green" if label in all_predicted_labels else "red"
                separator = "." if i < len(multipliers_and_labels) - 1 else ""
                self.shana_monitor_text.insert(END, f"{multiplier}{separator}", tag)
            self.shana_monitor_text.insert(END, "\n")

        self.shana_monitor_text.config(state='disabled')

    def update_pie_history_display(self) -> None:
        self.pie_history_text.config(state='normal')
        self.pie_history_text.delete(1.0, 'end')
        self.pie_history_text.config(bg='#000000', fg='#00FFFF')

        self.pie_history_text.tag_configure('p1', foreground='#00FFFF', font=('Courier New', 12, 'bold'))
        self.pie_history_text.tag_configure('p2', foreground='#00FF00', font=('Courier New', 12, 'bold'))
        self.pie_history_text.tag_configure('p3', foreground='#FF00FF', font=('Courier New', 12, 'bold'))
        self.pie_history_text.tag_configure('p4', foreground='#FFFF00', font=('Courier New', 12, 'bold'))
        self.pie_history_text.tag_configure('normal', foreground='#00FFFF', font=('Courier New', 12, 'bold'))

        headers = "Pred  Result\n" + "-" * 15 + "\n"
        self.pie_history_text.insert('end', headers, 'normal')

        for entry in self.pie_history_list[:10]:
            try:
                pred_pie, actual_pie = entry.split(' - ')
                if pred_pie == 'P1':
                    self.pie_history_text.insert('end', pred_pie, 'p1')
                elif pred_pie == 'P2':
                    self.pie_history_text.insert('end', pred_pie, 'p2')
                elif pred_pie == 'P3':
                    self.pie_history_text.insert('end', pred_pie, 'p3')
                elif pred_pie == 'P4':
                    self.pie_history_text.insert('end', pred_pie, 'p4')
                else:
                    self.pie_history_text.insert('end', pred_pie, 'normal')
                self.pie_history_text.insert('end', ' - ', 'normal')
                if actual_pie == 'P1':
                    self.pie_history_text.insert('end', actual_pie, 'p1')
                elif actual_pie == 'P2':
                    self.pie_history_text.insert('end', actual_pie, 'p2')
                elif actual_pie == 'P3':
                    self.pie_history_text.insert('end', actual_pie, 'p3')
                elif actual_pie == 'P4':
                    self.pie_history_text.insert('end', actual_pie, 'p4')
                else:
                    self.pie_history_text.insert('end', actual_pie, 'normal')
                self.pie_history_text.insert('end', '\n')
            except (ValueError, AttributeError):
                continue

        self.pie_history_text.config(state='disabled')

    def update_prediction_displays(self) -> None:
        for name in self.available_model_names:
            if name in self.last_model_predictions:
                predictions = self.last_model_predictions[name]
                if not isinstance(predictions, list):
                    predictions = [predictions]
                text_widget = self.model_prediction_labels[name]['text']
                text_widget.config(state='normal')
                text_widget.delete(1.0, 'end')
                for i, pred in enumerate(predictions):
                    if pred and 'M' in pred:
                        num, mult = pred.split('M')
                        text_widget.insert('end', num, 'normal')
                        text_widget.insert('end', f"-{mult}", 'bold')
                        if i < len(predictions) - 1:
                            text_widget.insert('end', ', ', 'normal')
                text_widget.config(state='disabled')

    def show_me_results(self) -> None:
        if not hasattr(self, 'show_me_window') or not self.show_me_window or not self.show_me_window.winfo_exists():
            self.show_me_window = Toplevel(self.root)
            self.show_me_window.title("SHOW ME Results")
            self.show_me_window.geometry("800x600")
            self.show_me_window.configure(bg='#FFFFFF')

            frame = tk.Frame(self.show_me_window, bg='#FFFFFF')
            frame.pack(expand=True, fill='both', padx=10, pady=10)

            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side='right', fill='y')

            self.show_me_text = Text(
                frame,
                wrap='word',
                font=('Courier New', 12),
                bg='#FFFFFF',
                fg='#000000',
                yscrollcommand=scrollbar.set,
                padx=10,
                pady=10
            )
            self.show_me_text.pack(expand=True, fill='both')
            scrollbar.config(command=self.show_me_text.yview)

            header_models = '   '.join(self.available_model_names)
            self.show_me_text.insert('end', f"{header_models}    LAST    WIN/LOSS\n")
            self.show_me_text.insert('end', "-" * 40 + "\n")

            for item in self.show_me_results_list[1:]:
                self.show_me_text.insert('end', item + '\n')

            btn_frame = tk.Frame(self.show_me_window, bg='#FFFFFF')
            btn_frame.pack(fill='x', padx=5, pady=5)

            clear_btn = tk.Button(
                btn_frame,
                text="CLEAR",
                font=("Courier New", 12, 'bold'),
                bg='#FF0000',
                fg='#FFFFFF',
                command=self.clear_show_me_results
            )
            clear_btn.pack(side='right', padx=5)

            close_btn = tk.Button(
                btn_frame,
                text="CLOSE",
                font=("Courier New", 12, 'bold'),
                bg='#333333',
                fg='#FFFFFF',
                command=self.show_me_window.destroy
            )
            close_btn.pack(side='right', padx=5)

            self.show_me_window.resizable(True, True)
            self.show_me_window.minsize(600, 400)
            self.show_me_window.protocol("WM_DELETE_WINDOW", self.show_me_window.destroy)
        else:
            self.show_me_text.config(state='normal')
            self.show_me_text.delete(1.0, 'end')
            header_models = '   '.join(self.available_model_names)
            self.show_me_text.insert('end', f"{header_models}    LAST    WIN/LOSS\n")
            self.show_me_text.insert('end', "-" * 40 + "\n")
            for item in self.show_me_results_list[1:]:
                self.show_me_text.insert('end', item + '\n')
            self.show_me_text.see('end')
            self.show_me_text.config(state='disabled')

        self.show_me_window.lift()
        self.show_me_window.focus_force()

    def clear_last_result(self) -> None:
        self.last_result_label.config(text="LAST RESULT: --")

    def clear_show_me_results(self) -> None:
        header_models = '   '.join(self.available_model_names)
        self.show_me_results_list = [f"[{header_models}] [LAST RESULT] [WIN OR LOSS]"]
        if hasattr(self, 'show_me_text') and self.show_me_text.winfo_exists():
            self.show_me_text.config(state='normal')
            self.show_me_text.delete(1.0, 'end')
            self.show_me_text.insert('end', '\n'.join(self.show_me_results_list))
            self.show_me_text.config(state='disabled')

    def reset_session(self) -> None:
        self.game_data = []
        self.save_game_data()
        header_models = '   '.join(self.available_model_names)
        self.show_me_results_list = [f"[{header_models}] [LAST RESULT] [WIN OR LOSS]"]
        self.pie_history_list = []
        self.history = []
        self.last_model_predictions = {}
        self.last_tarzan_prediction = ""
        self.total_entries = 0
        self.total_wins = 0
        self.last_result = None
        self.entry_box.delete(0, tk.END)
        self.tarzan_label.config(text="TARZAN Predicts: --")
        self.last_result_label.config(text="LAST RESULT: --")
        for name in self.model_prediction_labels:
            self.model_prediction_labels[name]['text'].config(state='normal')
            self.model_prediction_labels[name]['text'].delete(1.0, 'end')
            self.model_prediction_labels[name]['text'].config(state='disabled')
        self.update_shana_pie_monitor_by_zone()
        self.update_pie_history_display()
        self.root.title("Mega Predictor - Cross-Platform")

    def retrain_models(self) -> None:
        try:
            X, y = self.prepare_training_data()
            for name in self.available_model_names:
                model = self.models[name]
                try:
                    model.fit(X, y)
                    joblib.dump(model, self.model_paths[name])
                    print(f"Retrained {name} model successfully.")
                except Exception as e:
                    print(f"Error retraining {name}: {str(e)}")
            messagebox.showinfo("Retrain Complete", "All models have been retrained with current data.")
        except Exception as e:
            messagebox.showerror("Retrain Error", f"Failed to retrain models: {str(e)}")


if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    root = tk.Tk()
    app = MegaPredictorApp(root)
    root.mainloop()