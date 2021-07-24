import timesynth as ts
import pandas as pd
import datetime


class Settings:

    def __init__(self):
        self.date_range = None
        self.base = 0.0
        self.trend = 0.0
        self.total_impact = 1.0
        self.signals = []
        self.noises = []

    # Setters for Core Values --------------------------
    def set_base(self, base):
        self.base = base

    def set_trend(self, trend):
        self.trend = trend

    def set_total_impact(self, total_impact):
        self.total_impact = total_impact

    def set_date_range(self, date_range):
        self.date_range = date_range

    # Methods for adding Noise --------------------------
    def add_noise_gaussian_noise(self, impact: float, std=0.1):
        noise = ts.noise.GaussianNoise(std=std)
        code = f"ts.noise.GaussianNoise(std={std})"
        self.noises.append([noise, impact, code])

    def add_noise_red_noise(self, impact: float, std=1.5, tau=0.8):
        noise = ts.noise.RedNoise(std=std, tau=tau)
        code = f"ts.noise.RedNoise(std={std}, tau={tau})"
        self.noises.append([noise, impact, code])

    # Methods for adding Signals --------------------------
    def add_signal_sinusoidal(self, impact: float, frequency=0.25):
        signal = ts.signals.Sinusoidal(frequency=frequency)
        code = f"ts.signals.Sinusoidal(frequency={frequency})"
        self.signals.append([signal, impact, code])

    def add_signal_car(self, impact: float, ar_param=0.9, sigma=2):
        signal = ts.signals.CAR(ar_param=ar_param, sigma=sigma)
        code = f"ts.signals.CAR(ar_param={ar_param}, sigma={sigma})"
        self.signals.append([signal, impact, code])

    def add_signal_auto_regressive(self, impact: float, ar_param=[1.5, -0.75]):
        signal = ts.signals.AutoRegressive(ar_param=ar_param)
        code = f"ts.signals.AutoRegressive(ar_param={ar_param})"
        self.signals.append([signal, impact, code])

    def add_signal_pseudo_periodic(self, impact: float, frequency=2, freqSD=0.01, ampSD=0.5):
        signal = ts.signals.PseudoPeriodic(frequency=frequency, freqSD=freqSD, ampSD=ampSD)
        code = f"ts.signals.PseudoPeriodic(frequency={frequency}, freqSD={freqSD}, ampSD={ampSD})"
        self.signals.append([signal, impact, code])

    def add_signal_mackey_glass(self, impact: float):
        signal = ts.signals.MackeyGlass()
        code = f"ts.signals.MackeyGlass())"
        self.signals.append([signal, impact, code])

    def add_signal_narma(self, impact: float, order=10):
        signal = ts.signals.NARMA(order=order)
        code = f"ts.signals.NARMA(order={order})"
        self.signals.append([signal, impact, code])

    def add_signal_gaussian_process(self, impact: float, kernel='Matern', nu=3./2):
        # TODO: Add easy use for different kernels
        signal = ts.signals.GaussianProcess(kernel=kernel, nu=nu)
        code = f"ts.signals.GaussianProcess(kernel='{kernel}', nu={nu})"
        self.signals.append([signal, impact, code])


def generate(settings: Settings):
    # Define daterange
    date_range = settings.date_range
    if date_range is None:
        date_range = util_make_daterange(datetime.date.today(), 100)
        settings.set_date_range(date_range)

    # Define utils
    number_datapoints = len(date_range)
    zeros = pd.DataFrame(data=range(0, number_datapoints), index=date_range) * 0
    total_impact = settings.total_impact

    # Define Base
    base = zeros + settings.base

    # Define Trend
    trend = pd.DataFrame(data=range(0, number_datapoints), index=date_range)
    trend = trend * settings.trend

    # Set up Main Samplers (and util)
    flat_signal = ts.signals.Sinusoidal(frequency=0.0)
    irregular_time_samples = ts.TimeSampler(stop_time=20).sample_irregular_time(num_points=number_datapoints*2, keep_percentage=50)

    # Define Signals
    signals = []
    for signal in settings.signals:
        _, data, _ = ts.TimeSeries(signal[0], noise_generator=None).sample(irregular_time_samples)
        temp = pd.DataFrame(data=data, index=date_range) * signal[1]
        signals.append(temp)

    # Define Noises
    noises = []
    for noise in settings.noises:
        _, _, data = ts.TimeSeries(flat_signal, noise_generator=noise[0]).sample(irregular_time_samples)
        temp = pd.DataFrame(data=data, index=date_range) * noise[1]
        noises.append(temp)

    # Build Composite List to return
    composite_list = [base, trend]
    for signal in signals:
        composite_list.append(signal)
    for noise in noises:
        composite_list.append(noise)

    # Define combined timeseries to return
    out = base + trend
    for signal in signals:
        out += signal
    for noise in noises:
        out += noise
    out = out * total_impact

    return [out, composite_list]


# Util methods --------------------------
def util_make_daterange(start_date: datetime, number_of_datapoints: int, interval_type='D'):
    out = pd.date_range(start=start_date, periods=number_of_datapoints, freq=interval_type)
    return out


# Returns Python code representing the signal generation of the given Settings
def util_stringify_settings(settings: Settings):
    # TODO: Look into random seeds used by some of the signals (and noise?) & Fix Date Range Freq to be input String
    out = f"# Python Code to generate a defined Timeseries\n"
    out += f"import timesynth as ts\n"
    out += f"import pandas as pd\n"

    # Date Range
    date_range = settings.date_range
    out += f"\n"
    out += f"# Define Date Range\n"
    out += f"date_range = pd.date_range(start='{date_range[0].date()}', periods={len(date_range)}, freq={date_range.freq})\n"

    # Base & Trend
    out += f"\n"
    out += f"# Define Base & Trend\n"
    out += f"base = (pd.DataFrame(data=range(0, len(date_range)), index=date_range)*0) + {settings.base}\n"
    out += f"trend = pd.DataFrame(data=range(0, len(date_range)), index=date_range) * {settings.trend}\n"

    # Result Frame
    out += f"\n"
    out += f"# Initialize the result DataFrame as a combination of Base & Trend\n"
    out += f"result_frame = base + trend\n"

    # Time Sampler & Noise util
    if len(settings.noises) > 0 or len(settings.signals) > 0:
        out += f"\n"
        out += f"# Define a Timesampler\n"
        out += f"time_sampler = ts.TimeSampler(stop_time=20).sample_irregular_time(num_points=len(date_range)*2, keep_percentage=50)\n"

    # Noise
    if len(settings.noises) > 0:
        out += f"\n"
        out += f"# Define Noise and add it to the result\n"
        out += f"flat_signal = ts.signals.Sinusoidal(frequency=0.0)\n"
        for noise in settings.noises:
            out += f"\n"
            out += f"noise = {noise[2]}\n"
            out += f"_, _, noise_data = ts.TimeSeries(flat_signal, noise_generator=noise).sample(time_sampler)\n"
            out += f"noise_frame = pd.DataFrame(data=noise_data, index=date_range)\n"
            out += f"result_frame += noise_frame * {noise[1]}\n"

    # Signal
    if len(settings.signals) > 0:
        out += f"\n"
        out += f"# Define Signals and add them to the result\n"
        for index, signal in enumerate(settings.signals):
            if index>0:
                out += f"\n"
            out += f"signal = {signal[2]}\n"
            out += f"_, signal_data, _ = ts.TimeSeries(signal, noise_generator=None).sample(time_sampler)\n"
            out += f"signal_frame = pd.DataFrame(data=signal_data, index=date_range)\n"
            out += f"result_frame += signal_frame * {signal[1]}\n"

    # Return complete String
    return out
