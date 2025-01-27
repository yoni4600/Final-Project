import io
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from config import Config
import main_code



class BufferedStdoutRedirector:
    def __init__(self, text_widget, flush_interval=10000):
        """
        :param text_widget: The ScrolledText or Text widget for output.
        :param flush_interval: Interval in milliseconds for flushing the buffer (e.g. 10000 = 10s).
        """
        self.text_widget = text_widget
        self.buffer = io.StringIO()
        self.flush_interval = flush_interval

        # Schedule the first flush
        self.text_widget.after(self.flush_interval, self.periodic_flush)

    def write(self, text):
        # Accumulate text in the buffer.
        self.buffer.write(text)

    def flush(self):
        # We do nothing in flush(), because we flush in periodic_flush().
        pass

    def periodic_flush(self):
        """
        This method is called every 'flush_interval' ms.
        It appends whatever is in the buffer to the ScrolledText widget
        and then clears the buffer.
        """
        output = self.buffer.getvalue()
        if output:
            self.text_widget.insert(tk.END, output)
            self.text_widget.see(tk.END)

            self.buffer.truncate(0)
            self.buffer.seek(0)

        self.text_widget.after(self.flush_interval, self.periodic_flush)


class ToolTip:
    """A class to create a tooltip for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window:
            return
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{self.widget.winfo_rootx() + 20}+{self.widget.winfo_rooty() + 20}")
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background="white",
            relief="solid",
            borderwidth=1,
            wraplength=300,
            anchor="w",
            justify="left"
        )
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Final-Project")
        self.geometry("1000x700")

        # Store file paths for edges and labels
        self.edges_file_path = tk.StringVar()
        self.labels_file_path = tk.StringVar()

        # Create frames for better organization
        self._create_file_selection_frame()
        self._create_parameters_frame()
        self._create_run_frame()
        self._create_log_frame()

        self._redirect_console_output()
        self._create_help_label()
        
        self.stop_event = threading.Event()
        self.worker_thread = None

        # Handle the close event
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_help_label(self):
        help_label = tk.Label(self, text="?", font=("Arial", 16, "bold"), fg="blue", cursor="hand2")
        help_label.pack(side="top", anchor="ne", padx=10, pady=5)
        ToolTip(help_label, text="Instructions:\n\n"
                                 "1. Select the Edgelist and Labels files.\n"
                                 "2. Set hyperparameters as needed.\n"
                                 "3. Use 'Run Evaluation' to start the process.\n"
                                 "4. Use 'Save Config' to save settings, and 'Load Config' to restore them.\n"
                                 "5. View progress and output in the Console Output section.")

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
        thresh1_help = tk.Label(params_frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        thresh1_help.grid(row=1, column=2, padx=5, pady=2)
        ToolTip(thresh1_help, text="Classify “strong” or “weak” relation of an edge for each iteration in the Research Plan "
                                   "based on the cosine similarity score.")

        # 3) TRESHOLD2
        self.threshold2_var = tk.IntVar(value=Config.TRESHOLD2)
        tk.Label(params_frame, text="TRESHOLD2 (%):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.threshold2_var).grid(row=2, column=1, padx=5, pady=2)
        thresh2_help = tk.Label(params_frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        thresh2_help.grid(row=2, column=2, padx=5, pady=2)
        ToolTip(thresh2_help, text="Classify “strong” or “weak” relation of an edge after all the K iterations "
                                   "based on the classification scores of each iteration.")

        # 4) PERCENTAGE
        self.percentage_var = tk.IntVar(value=Config.PERCENTAGE)
        tk.Label(params_frame, text="Edges Removal (%):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.percentage_var).grid(row=3, column=1, padx=5, pady=2)
        percent_help = tk.Label(params_frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        percent_help.grid(row=3, column=2, padx=5, pady=2)
        ToolTip(percent_help, text="The percentage of edges to remove during preprocessing.")

        # 5) K
        self.k_var = tk.IntVar(value=Config.K)
        tk.Label(params_frame, text="K - Number Of Iterations:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.k_var).grid(row=4, column=1, padx=5, pady=2)
        k_help = tk.Label(params_frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        k_help.grid(row=4, column=2, padx=5, pady=2)
        ToolTip(k_help, text="The number of iterations to run for the algorithm.")

        # 6) Alpha
        self.alpha_var = tk.DoubleVar(value=Config.ALPHA)
        tk.Label(params_frame, text="Alpha (0.0-1.0):").grid(row=5, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.alpha_var).grid(row=5, column=1, padx=5, pady=2)
        alpha_help = tk.Label(params_frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        alpha_help.grid(row=5, column=2, padx=5, pady=2)
        ToolTip(alpha_help, text="The alpha value controls the weights and the transition probability when walking.")

        # 7) Pyramid Scales
        self.pyramidScales_var = tk.IntVar(value=Config.PYRAMID_SCALES)
        tk.Label(params_frame, text="Pyramid Scales:").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.pyramidScales_var).grid(row=6, column=1, padx=5, pady=2)
        pyramid_help = tk.Label(params_frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        pyramid_help.grid(row=6, column=2, padx=5, pady=2)
        ToolTip(pyramid_help, text="Defines the number of scales in the graph pyramid.")

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
        # Create redirectors for stdout and stderr with a 10-second flush interval
        stdout_redirector = BufferedStdoutRedirector(self.log_text, flush_interval=10000)
        stderr_redirector = BufferedStdoutRedirector(self.log_text, flush_interval=10000)

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
        2. Set the EDGES_TXT and LABELS_TXT in main_code
        3. Store our custom tqdm writer in Config, so other modules can use it
        4. Start the worker thread (which calls main_code.main())
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

        # Set global vars in main_code
        main_code.EDGES_TXT = self.edges_file_path.get()
        main_code.LABELS_TXT = self.labels_file_path.get()

        # Create our custom tqdm writer for a single-line console progress bar
        tqdm_writer = TqdmTextWriter(self.log_text)
        # Store it in the Config as a global reference
        Config.TQDM_WRITER = tqdm_writer

        self.stop_event.clear()
        # Start the worker thread (no need to pass tqdm_writer as an arg)
        worker_thread = threading.Thread(
            target=self._threaded_main_code,
            daemon=True
        )
        worker_thread.start()

    def _threaded_main_code(self):
        """
        Runs main_code.main() in a worker thread, so the UI doesn't freeze.
        If there's an exception, show it via messagebox in the main thread.
        """
        try:
            while not self.stop_event.is_set():
                main_code.main()  # Your main_code logic
                break  # Exit after one run (remove this line if main_code.main() should repeat)
        except Exception as e:
            self.log_text.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {e}"))
        finally:
            self.stop_event.set()  # Ensure the event is set when thread exits

    def on_close(self):
            """Handle the window close event to stop the worker thread."""
            if self.worker_thread and self.worker_thread.is_alive():
                # Set the stop event to signal the worker thread
                self.stop_event.set()
                self.worker_thread.join()  # Wait for the thread to finish

            # Restore stdout and stderr (optional)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            # Destroy the UI
            self.destroy()
            
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


class TqdmTextWriter(io.StringIO):
    """
    A file-like object that allows 'tqdm' to write a single-line progress bar
    in the Tkinter ScrolledText widget (without spamming multiple lines).
    """
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, text):
        # Because this is called in a worker thread, schedule the insertion
        # on the main thread via 'after(0, ...)'.
        self.text_widget.after(0, lambda: self._write_in_main_thread(text))

    def flush(self):
        pass

    def _write_in_main_thread(self, text):
        # If 'tqdm' sends carriage returns, overwrite the last line
        if '\r' in text:
            parts = text.split('\r')
            # The last part is the fresh portion
            last_part = parts[-1]
            self._delete_last_line()
            self.text_widget.insert(tk.END, last_part)
        else:
            # Normal text (may have \n)
            self.text_widget.insert(tk.END, text)

        # Always scroll to the end
        self.text_widget.see(tk.END)

    def _delete_last_line(self):
        last_line_index = self.text_widget.index('end-1c')
        line_number = last_line_index.split('.')[0]
        self.text_widget.delete(f"{line_number}.0", tk.END)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
