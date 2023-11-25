import librosa
import librosa.display
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, Text
import numpy as np

class BPMAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("BPM Analyzer")

        # Initialize variables
        self.y = None
        self.sr = None
        self.segment_starts = None
        self.estimated_bpm = None

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Frame to group widgets
        frame = tk.Frame(self.root, bg='#323232')
        frame.pack(fill=tk.BOTH, expand=True)

        # Labels
        self.file_label = tk.Label(frame, text="No file selected", bg='#323232', fg='#bd3254', bd=4)
        self.file_label.grid(row=0, column=0, columnspan=2)

        # Buttons
        self.select_button = tk.Button(frame, text="Select Audio File", command=self.select_file, bg='#bd3254', fg='white', bd=4, relief=tk.RAISED)
        self.select_button.grid(row=1, column=0, pady=10)

        self.analyze_button = tk.Button(frame, text="Analyze BPM", command=self.analyze_bpm, bg='#bd3254', fg='white', bd=4, relief=tk.RAISED)
        self.analyze_button.grid(row=1, column=1, pady=10)

        self.log_view_button = tk.Button(frame, text="View BPM Log", command=self.view_bpm_log, bg='#bd3254', fg='white', bd=4, relief=tk.RAISED)
        self.log_view_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.on_resize_handler = lambda event, log_text=None: self.on_resize(event, log_text)   


    def select_file(self):
        # Ask the user to select an audio file
        audio_file = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio files", "*.wav;*.mp3")])

        # Load the selected audio file
        self.y, self.sr = librosa.load(audio_file)

        # Update file label
        self.file_label.config(text=f"Selected File: {audio_file}")

    def analyze_bpm(self):
        if self.y is not None and len(self.y) >= 2048:
        # Estimate BPM for each segment
            self.segment_starts, self.estimated_bpm = self.estimate_tempo_for_segments()

        # Plot the linear-frequency power spectrogram

            plt.figure(figsize=(12, 6), facecolor='#323232')

        # Linear-frequency graph
            plt.subplot(2, 1, 1)
            D_linear = librosa.amplitude_to_db(np.abs(librosa.stft(self.y, n_fft=2048)), ref=np.max)
            librosa.display.specshow(D_linear, y_axis='linear', x_axis='time', sr=self.sr)
            plt.title('Linear-Frequency Power Spectrogram', color='#bd3254')

        # Logarithmic-frequency graph
            plt.subplot(2, 1, 2)
            D_log = librosa.amplitude_to_db(np.abs(librosa.stft(self.y, n_fft=2048)), ref=np.max)
            librosa.display.specshow(D_log, y_axis='log', x_axis='time', sr=self.sr)
            plt.title('Logarithmic-Frequency Power Spectrogram', color='#bd3254')

            plt.tight_layout(pad=2)  # Increase the pad value for better spacing
            plt.show()



        else:
            print("Audio signal is too short for analysis.")


    def estimate_tempo_for_segments(self, segment_duration=10):
        # Calculate the total duration of the audio in seconds
        total_duration = librosa.get_duration(y=self.y, sr=self.sr)

        # Check if the signal is long enough for analysis
        if total_duration < segment_duration:
            raise ValueError("Audio signal is too short for analysis.")

        # Calculate the number of segments
        num_segments = int(np.ceil(total_duration / segment_duration))

        # Initialize arrays to store segment start times and estimated BPMs
        segment_starts = np.arange(0, total_duration, segment_duration)
        estimated_bpm = np.zeros(num_segments)

        for i in range(num_segments):
            # Extract the segment from the audio
            start_time = int(segment_starts[i])
            end_time = int(min(start_time + segment_duration, total_duration))
            segment, _ = librosa.effects.trim(self.y[start_time * self.sr:end_time * self.sr])

            # Estimate BPM for the segment
            tempo, _ = librosa.beat.beat_track(y=segment, sr=self.sr)
            estimated_bpm[i] = tempo

        return segment_starts, estimated_bpm

    def view_bpm_log(self):
        # Check if BPM analysis has been performed
        if self.segment_starts is None or self.estimated_bpm is None:
            print("Please analyze BPM before viewing the log.")
            return

        # Create a new window for BPM log view
        log_window = tk.Toplevel(self.root)
        log_window.title("BPM Log View")

        log_window.bind("<Configure>", self.on_resize_handler)

        # Create a text widget to display the BPM log
        log_text = Text(log_window, bg="#323232", fg="#bd3254", highlightbackground="#323232")  # Set background, text, and highlight background color
        log_text.pack(fill=tk.BOTH, expand=True)

        # Insert BPM log into the text widget
        for i, (start_time, bpm) in enumerate(zip(self.segment_starts, self.estimated_bpm)):
            log_text.insert(tk.END, f"Segment {i + 1}: Start Time = {start_time:.2f}s, Estimated BPM = {bpm:.2f}\n")

    def waveform(self):
        # Create a new window for waveform view
        waveform_window = tk.Toplevel(self.root)
        waveform_window.title("Waveform View")

        waveform_window.bind("<Configure>", self.on_resize_handler)

        # Create a text widget to display the waveform
        waveform_text = Text(waveform_window, bg="#323232", fg="#bd3254", highlightbackground="#323232")
        plt.figure(figsize=(12, 6), facecolor='#323232')
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(self.y, sr=self.sr)
        waveform_text.pack(fill=tk.BOTH, expand=True)

    def on_resize(self, event, log_text):
        # Resize the text widget to fit the window
        log_text.configure(bg="#323232", fg="#bd3254", highlightbackground="#323232", width=event.width, height=event.height)


# Create the main Tkinter window
root = tk.Tk()
root.resizable(width=True, height=True)
# Create an instance of BPMAnalyzer
bpm_analyzer = BPMAnalyzer(root)
# Start the Tkinter event loop
root.mainloop()
