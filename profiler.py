import os
import time
import threading
import psutil
import pandas as pd

class ResourceProfiler:
    def __init__(self, interval_sec=0.05, profile_name="profile"):
        """
        A background thread-based resource profiler that logs current process resources.
        
        Args:
            interval_sec (float): Polling interval in seconds (default: 50ms).
            profile_name (str): Label for the raw CSV log file.
        """
        self.interval_sec = interval_sec
        self.profile_name = profile_name
        self.process = psutil.Process(os.getpid())
        
        # Thread state
        self._stop_event = threading.Event()
        self._thread = None
        
        # Data store
        self.timestamps = []
        self.cpu_percentages = []
        self.rss_memory_mb = []
        self.vms_memory_mb = []
        
        # Summary stats
        self.summary = {}

    def _monitor(self):
        """Worker thread loop to sample CPU and RAM."""
        # Prime the CPU calculation (first call always returns 0.0)
        try:
            self.process.cpu_percent(interval=None)
        except Exception:
            pass
            
        start_time = time.time()
        
        while not self._stop_event.is_set():
            t_curr = time.time() - start_time
            try:
                # Capture current process CPU percent (shares CPU core; can exceed 100% on multi-core)
                cpu = self.process.cpu_percent(interval=None)
                
                # Capture memory info in MB
                mem_info = self.process.memory_info()
                rss = mem_info.rss / (1024 * 1024)
                vms = mem_info.vms / (1024 * 1024)
                
                self.timestamps.append(t_curr)
                self.cpu_percentages.append(cpu)
                self.rss_memory_mb.append(rss)
                self.vms_memory_mb.append(vms)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                # Silently catch exceptions to prevent crash, continue sampling
                pass
                
            # Sleep in small increments to respond quickly to stop events
            sleep_chunks = int(self.interval_sec / 0.01)
            for _ in range(sleep_chunks):
                if self._stop_event.is_set():
                    break
                time.sleep(0.01)

    def start(self):
        """Start the background resource profiling."""
        self.timestamps.clear()
        self.cpu_percentages.clear()
        self.rss_memory_mb.clear()
        self.vms_memory_mb.clear()
        self.summary.clear()
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self, save_dir=None):
        """
        Stop the profiling run and compute summary statistics.
        
        Args:
            save_dir (str): Optional directory path to save the raw timeseries CSV.
            
        Returns:
            dict: Summary metrics containing start memory, peak memory, memory growth, and average CPU pressure.
        """
        if self._thread is None:
            return {}
            
        self._stop_event.set()
        self._thread.join()
        
        # Guard if no samples were taken
        if not self.timestamps:
            return {}
            
        # Calculate summary statistics
        start_mem = self.rss_memory_mb[0]
        peak_mem = max(self.rss_memory_mb)
        mem_growth = peak_mem - start_mem
        
        # Filter out first CPU element which can sometimes spike or be uncalibrated
        valid_cpus = self.cpu_percentages[1:] if len(self.cpu_percentages) > 1 else self.cpu_percentages
        avg_cpu = sum(valid_cpus) / len(valid_cpus) if valid_cpus else 0.0
        peak_cpu = max(valid_cpus) if valid_cpus else 0.0
        
        self.summary = {
            "duration_sec": self.timestamps[-1],
            "start_memory_mb": round(start_mem, 2),
            "peak_memory_mb": round(peak_mem, 2),
            "memory_growth_mb": round(mem_growth, 2),
            "avg_cpu_percent": round(avg_cpu, 2),
            "peak_cpu_percent": round(peak_cpu, 2),
            "samples_collected": len(self.timestamps)
        }
        
        # Export raw timeseries to CSV if requested
        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                csv_filename = os.path.join(save_dir, f"{self.profile_name}_raw.csv")
                df = pd.DataFrame({
                    "relative_time_sec": self.timestamps,
                    "cpu_percent": self.cpu_percentages,
                    "rss_mem_mb": self.rss_memory_mb,
                    "vms_mem_mb": self.vms_memory_mb
                })
                df.to_csv(csv_filename, index=False)
            except Exception as e:
                print(f"Warning: Failed to save raw profile {self.profile_name} to CSV: {e}")
                
        return self.summary
