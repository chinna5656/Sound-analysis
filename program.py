import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import platform
import warnings
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ตั้งค่าฟอนต์ภาษาไทย
import matplotlib
matplotlib.use('TkAgg')
system_os = platform.system()
if system_os == 'Windows':
    font_name = 'Tahoma'
elif system_os == 'Darwin':
    font_name = 'Ayuthaya'
else:
    font_name = 'Waree'
matplotlib.rcParams['font.family'] = font_name
matplotlib.rcParams['axes.unicode_minus'] = False

class SpectrumAnalyzerPro:
    def __init__(self, root):
        self.root = root
        self.root.title("Ponglang Spectrum Analyst (Numerical & Visual)")
        
        # ปรับขนาดหน้าจอ
        w = min(1300, self.root.winfo_screenwidth() - 50)
        h = min(900, self.root.winfo_screenheight() - 80)
        self.root.geometry(f"{w}x{h}")
        self.root.configure(bg="#f1f8e9")

        self.file1 = ""
        self.file2 = ""
        self.data1 = None
        self.data2 = None

        self.setup_ui()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#33691e", height=60)
        header.pack(fill="x")
        tk.Label(header, text="🎛️ Advanced Spectrum & Harmonic Analysis", 
                 font=("Tahoma", 16, "bold"), bg="#33691e", fg="white").pack(pady=15)

        # Main Layout (Split Left/Right)
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#f1f8e9", sashwidth=5)
        paned.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Left Panel: Controls & Numerical Report ---
        left_frame = tk.Frame(paned, bg="white", relief="solid", bd=1)
        paned.add(left_frame, minsize=400, width=450)

        self.create_controls(left_frame)
        self.create_report_area(left_frame)

        # --- Right Panel: Graphs ---
        right_frame = tk.Frame(paned, bg="white", relief="solid", bd=1)
        paned.add(right_frame, stretch="always")
        
        self.create_graph_area(right_frame)

    def create_controls(self, parent):
        frame = tk.LabelFrame(parent, text="Control Panel", font=("Tahoma", 10, "bold"), bg="white", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=10)

        tk.Button(frame, text="📂 1. ต้นฉบับ (Ref)", command=lambda: self.select_file(1), bg="#558b2f", fg="white").grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.lbl1 = tk.Label(frame, text="-", bg="white", fg="gray", anchor="w")
        self.lbl1.grid(row=0, column=1, sticky="w")

        tk.Button(frame, text="📂 2. ทดสอบ (Test)", command=lambda: self.select_file(2), bg="#ff8f00", fg="white").grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        self.lbl2 = tk.Label(frame, text="-", bg="white", fg="gray", anchor="w")
        self.lbl2.grid(row=1, column=1, sticky="w")

        tk.Button(frame, text="🚀 วิเคราะห์และสรุปผล", command=self.process, 
                  bg="#2e7d32", fg="white", font=("Tahoma", 12, "bold"), height=2).grid(row=2, column=0, columnspan=2, sticky="ew", pady=15)
        frame.columnconfigure(1, weight=1)

    def create_report_area(self, parent):
        tk.Label(parent, text="📝 ผลการวิเคราะห์ (Analysis Report)", font=("Tahoma", 11, "bold"), bg="white", anchor="w").pack(fill="x", padx=15, pady=(5, 5))
        self.txt_report = tk.Text(parent, font=("Consolas", 10), bg="#f9fbe7", relief="solid", bd=1)
        self.txt_report.pack(fill="both", expand=True, padx=15, pady=10)

    def create_graph_area(self, parent):
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def select_file(self, num):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if path:
            name = path.split("/")[-1]
            if num == 1:
                self.file1 = path
                self.lbl1.config(text=name, fg="#33691e")
                self.data1 = None
            else:
                self.file2 = path
                self.lbl2.config(text=name, fg="#ef6c00")
                self.data2 = None

    def analyze_spectrum(self, path):
        try:
            y, sr = librosa.load(path, sr=22050, mono=True)
            y, _ = librosa.effects.trim(y, top_db=20)

            # 1. Fundamental Frequency (f0)
            f0 = librosa.yin(y, fmin=60, fmax=2000)
            f0[f0==2000] = np.nan
            times = librosa.times_like(f0, sr=sr)
            avg_f0 = np.nanmean(f0)

            # 2. Spectral Bandwidth (ความกว้างของย่านเสียง)
            bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            avg_bw = np.mean(bw)

            # 3. Harmonic Spectrum (FFT)
            n_fft = 2048
            S = np.abs(librosa.stft(y, n_fft=n_fft))
            spectrum_mean = np.mean(S, axis=1) # Average Magnitude per Frequency
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # 4. Spectral Roll-off (บอกความใส/ความถี่สูง)
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

            return {
                "f0": f0, "times": times, "avg_f0": avg_f0,
                "bw": bw, "avg_bw": avg_bw,
                "spectrum": spectrum_mean, "freqs": freqs,
                "rolloff": rolloff,
                "sr": sr
            }
        except Exception as e:
            return None

    def process(self):
        if not self.file1 or not self.file2:
            messagebox.showwarning("เตือน", "กรุณาเลือกไฟล์ให้ครบ")
            return

        self.root.config(cursor="wait")
        self.root.update()

        if not self.data1: self.data1 = self.analyze_spectrum(self.file1)
        if not self.data2: self.data2 = self.analyze_spectrum(self.file2)

        if self.data1 and self.data2:
            self.generate_report(self.data1, self.data2)
            self.plot_graphs(self.data1, self.data2)
        
        self.root.config(cursor="")

    def generate_report(self, d1, d2):
        # 1. คำนวณความต่าง f0 (Cents)
        if d1['avg_f0'] > 0 and d2['avg_f0'] > 0:
            cents = 1200 * np.log2(d2['avg_f0'] / d1['avg_f0'])
        else:
            cents = 0

        # 2. คำนวณความเหมือนของ Spectrum (Correlation)
        # ตัดความถี่ช่วง 0-5000Hz มาเทียบกัน (ช่วงสำคัญของดนตรี)
        limit_idx = np.where(d1['freqs'] > 5000)[0][0]
        corr, _ = pearsonr(d1['spectrum'][:limit_idx], d2['spectrum'][:limit_idx])
        spec_similarity = max(0, corr * 100)

        # 3. วิเคราะห์ Bandwidth & Rolloff
        diff_bw = d2['avg_bw'] - d1['avg_bw']
        diff_roll = d2['rolloff'] - d1['rolloff']

        # สร้างข้อความวิเคราะห์อัตโนมัติ
        analysis_pitch = "✅ เสียงตรงคีย์" if abs(cents) < 15 else ("⚠️ เสียงสูงไป (Sharp)" if cents > 0 else "⚠️ เสียงต่ำไป (Flat)")
        
        analysis_timbre = ""
        if diff_bw > 200: analysis_timbre += "- เสียงหนากว่าต้นฉบับ (More Fullness)\n"
        elif diff_bw < -200: analysis_timbre += "- เสียงบางกว่าต้นฉบับ (Thinner)\n"
        else: analysis_timbre += "- ความหนาใกล้เคียงกัน\n"

        if diff_roll > 300: analysis_timbre += "- เสียงใส/แหลมกว่า (Brighter)"
        elif diff_roll < -300: analysis_timbre += "- เสียงทึบ/อู้อี้กว่า (Duller)"
        else: analysis_timbre += "- ความใสใกล้เคียงกัน"

        report = f"""
{'='*40}
        NUMERICAL REPORT & ANALYSIS
{'='*40}

1. FUNDAMENTAL FREQUENCY (f0)
   • Ref:  {d1['avg_f0']:.2f} Hz
   • Test: {d2['avg_f0']:.2f} Hz
   👉 Deviation: {cents:+.2f} Cents
   📝 สรุป: {analysis_pitch}

2. HARMONIC SIMILARITY (ความเหมือนเนื้อเสียง)
   👉 Score: {spec_similarity:.2f} %
   (คำนวณจากรูปทรงสเปกตรัมช่วง 0-5kHz)

3. SPECTRAL BANDWIDTH (ความกว้างย่านเสียง)
   • Ref:  {d1['avg_bw']:.0f} Hz
   • Test: {d2['avg_bw']:.0f} Hz
   👉 Diff: {diff_bw:+.0f} Hz

4. TIMBRE ANALYSIS (วิเคราะห์เนื้อเสียง)
{analysis_timbre}

{'='*40}
"""
        self.txt_report.config(state="normal")
        self.txt_report.delete("1.0", tk.END)
        self.txt_report.insert("1.0", report)
        self.txt_report.config(state="disabled")

    def plot_graphs(self, d1, d2):
        self.fig.clear()
        gs = self.fig.add_gridspec(3, 1, hspace=0.45)

        # Graph 1: f0 Comparison
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(d1['times'], d1['f0'], label='Ref', color='#33691e', lw=2)
        ax1.plot(d2['times'], d2['f0'], label='Test', color='#ff8f00', lw=2, ls='--')
        ax1.set_title("1. Fundamental Frequency ($f_0$)", fontsize=10, fontweight='bold')
        ax1.set_ylabel("Hz")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Graph 2: Harmonic Spectrum (Log Scale for Magnitude)
        ax2 = self.fig.add_subplot(gs[1])
        # Plot Magnitude in dB vs Frequency
        idx_lim = np.where(d1['freqs'] > 4000)[0][0]
        
        spec1_db = librosa.amplitude_to_db(d1['spectrum'], ref=np.max)
        spec2_db = librosa.amplitude_to_db(d2['spectrum'], ref=np.max)
        
        ax2.plot(d1['freqs'][:idx_lim], spec1_db[:idx_lim], label='Ref', color='#33691e', alpha=0.8)
        ax2.plot(d2['freqs'][:idx_lim], spec2_db[:idx_lim], label='Test', color='#ff8f00', alpha=0.8)
        ax2.set_title("2. Harmonic Spectrum (ความถี่และฮาร์มอนิก)", fontsize=10, fontweight='bold')
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Graph 3: Bandwidth Envelope
        ax3 = self.fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(d1['times'], d1['bw'], label='Ref', color='#33691e')
        ax3.plot(d2['times'], d2['bw'], label='Test', color='#ff8f00')
        ax3.fill_between(d1['times'], d1['bw'], color='#33691e', alpha=0.1)
        ax3.set_title("3. Spectral Bandwidth (ความหนาของเสียงตามเวลา)", fontsize=10, fontweight='bold')
        ax3.set_ylabel("Hz")
        ax3.set_xlabel("Time (s)")
        ax3.grid(alpha=0.3)

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrumAnalyzerPro(root)
    root.mainloop()
