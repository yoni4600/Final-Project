import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from config import Config
import main_code


class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        if text.strip():
            self.text_widget.after(0, self._append_text, text)

    def flush(self):
        pass

    def _append_text(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Final-Project")

        # Store file paths for edges and labels
        self.edges_file_path = tk.StringVar()
        self.labels_file_path = tk.StringVar()

        # Create frames for better organization
        self._create_file_selection_frame()
        self._create_parameters_frame()
        self._create_run_frame()
        self._create_log_frame()

        self._redirect_console_output()

    def _create_file_selection_frame(self):
        file_frame = tk.LabelFrame(self, text="Select Dataset Files", padx=10, pady=10)
        file_frame.pack(fill="x", padx=5, pady=5)

        # Edgelist file
        tk.Label(file_frame, text="Edgelist File:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        tk.Entry(file_frame, textvariable=self.edges_file_path, width=50).grid(row=0, column=1, padx=5, pady=2)
        tk.Button(file_frame, text="Browse", command=self.browse_edges_file).grid(row=0, column=2, padx=5, pady=2)

        # Labels file
        tk.Label(file_frame, text="Labels File:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        tk.Entry(file_frame, textvariable=self.labels_file_path, width=50).grid(row=1, column=1, padx=5, pady=2)
        tk.Button(file_frame, text="Browse", command=self.browse_labels_file).grid(row=1, column=2, padx=5, pady=2)

    def _create_parameters_frame(self):
        params_frame = tk.LabelFrame(self, text="Hyperparameters", padx=10, pady=10)
        params_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 1) DIMENSION
        self.dimension_var = tk.IntVar(value=Config.DIMENSION)
        tk.Label(params_frame, text="Embedding Vector Dimension:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.dimension_var).grid(row=0, column=1, padx=5, pady=2)

        # 2) TRESHOLD1
        self.threshold1_var = tk.DoubleVar(value=Config.TRESHOLD1)
        tk.Label(params_frame, text="TRESHOLD1 (0.0-1.0): ").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.threshold1_var).grid(row=1, column=1, padx=5, pady=2)

        # 3) TRESHOLD2
        self.threshold2_var = tk.IntVar(value=Config.TRESHOLD2)
        tk.Label(params_frame, text="TRESHOLD2 (%):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.threshold2_var).grid(row=2, column=1, padx=5, pady=2)

        # 4) PERCENTAGE
        self.percentage_var = tk.IntVar(value=Config.PERCENTAGE)
        tk.Label(params_frame, text="Edges Removal (%):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.percentage_var).grid(row=3, column=1, padx=5, pady=2)

        # 5) K
        self.k_var = tk.IntVar(value=Config.K)
        tk.Label(params_frame, text="K - Number Of Iterations:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.k_var).grid(row=4, column=1, padx=5, pady=2)

        # 6) Alpha
        self.alpha_var = tk.DoubleVar(value=Config.ALPHA)
        tk.Label(params_frame, text="Alpha (0.0-1.0):").grid(row=5, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.alpha_var).grid(row=5, column=1, padx=5, pady=2)

        # 7) Pyramid Scales
        self.pyramidScales_var = tk.IntVar(value=Config.PYRAMID_SCALES)
        tk.Label(params_frame, text="Pyramid Scales:").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.pyramidScales_var).grid(row=6, column=1, padx=5, pady=2)

    def _create_run_frame(self):
        run_frame = tk.Frame(self)
        run_frame.pack(fill="x", padx=5, pady=5)

        tk.Button(run_frame, text="Run Evaluation", command=self.run_evaluation).pack(side="left", padx=5, pady=5)
        tk.Button(run_frame, text="Save Config", command=self.save_config).pack(side="left", padx=5, pady=5)
        tk.Button(run_frame, text="Load Config", command=self.load_config).pack(side="left", padx=5, pady=5)

    def _create_log_frame(self):
        log_frame = tk.LabelFrame(self, text="Console Output", padx=5, pady=5)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.log_text = ScrolledText(log_frame, wrap="word", width=80, height=10)
        self.log_text.pack(fill="both", expand=True)

    def _redirect_console_output(self):
        # Create redirectors for stdout and stderr
        stdout_redirector = StdoutRedirector(self.log_text)
        stderr_redirector = StdoutRedirector(self.log_text)

        sys.stdout = stdout_redirector
        sys.stderr = stderr_redirector

    def browse_edges_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Edgelist TXT File",
            filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
        )
        if file_path:
            self.edges_file_path.set(file_path)

    def browse_labels_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Labels TXT File",
            filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
        )
        if file_path:
            self.labels_file_path.set(file_path)

    def run_evaluation(self):
        """
        1. Update Config with the values from the UI
        2. Call the main process from main_code.py
        """
        # Update config with UI
        Config.DIMENSION = self.dimension_var.get()
        Config.TRESHOLD1 = self.threshold1_var.get()
        Config.TRESHOLD2 = self.threshold2_var.get()
        Config.PERCENTAGE = self.percentage_var.get()
        Config.K = self.k_var.get()
        Config.ALPHA = self.alpha_var.get()
        Config.PYRAMID_SCALES = self.pyramidScales_var.get()

        if not self.edges_file_path.get() or not self.labels_file_path.get():
            messagebox.showwarning("Input Error", "Please select both Edgelist and Labels files.")
            return

        # Example: we override the dataset path generation
        # If your main_code relies on these global variables, make sure you apply them there.
        # For a more robust solution, pass these as arguments to main_code.main(...)
        main_code.EDGES_TXT = self.edges_file_path.get()
        main_code.LABELS_TXT = self.labels_file_path.get()


        try:
            main_code.main()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def save_config(self):
        """Save current Config settings to a JSON file using Config.save_to_json."""
        file_path = filedialog.asksaveasfilename(
            title="Save Config to JSON",
            defaultextension=".json",
            filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
        )
        if file_path:
            # Update config with UI values
            Config.DIMENSION = self.dimension_var.get()
            Config.TRESHOLD1 = self.threshold1_var.get()
            Config.TRESHOLD2 = self.threshold2_var.get()
            Config.PERCENTAGE = self.percentage_var.get()
            Config.K = self.k_var.get()
            Config.ALPHA = self.alpha_var.get()
            Config.PYRAMID_SCALES = self.pyramidScales_var.get()

            try:
                Config.save_to_json(file_path)
                messagebox.showinfo("Success", f"Config saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {e}")

    def load_config(self):
        """Load Config settings from a JSON file using Config.load_from_json."""
        file_path = filedialog.askopenfilename(
            title="Load Config from JSON",
            filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
        )
        if file_path:
            try:
                Config.load_from_json(file_path)
                self.dimension_var.set(Config.DIMENSION)
                self.threshold1_var.set(Config.TRESHOLD1)
                self.threshold2_var.set(Config.TRESHOLD2)
                self.percentage_var.set(Config.PERCENTAGE)
                self.k_var.set(Config.K)
                self.alpha_var.set(Config.ALPHA)
                self.pyramidScales_var.set(Config.PYRAMID_SCALES)

                messagebox.showinfo("Success", f"Config loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
