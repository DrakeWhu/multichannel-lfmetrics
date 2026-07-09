from base_command import Command
from insert_result import InsertResult
from singletons.config import Config
from singletons.logger import get_logger
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import numpy as np
import sys
import os
import h5py
import json

# --- al inicio del archivo, tras imports ---
ALL_METRICS = {
    # Wakefields (longitudinal & transverse) en unidades “naturales” de tus arrays
    "longitudinal_amplitude": {"unit": "", "desc": "Peak amplitude of Ez (fit)"},
    "longitudinal_wavelength": {"unit": "cells", "desc": "Weighted avg wavelength of Ez (fit)"},
    "transverse_amplitude": {"unit": "", "desc": "Peak amplitude of Ey - c Bz (fit)"},
    "transverse_wavelength": {"unit": "cells", "desc": "Weighted avg wavelength of Ey - c Bz (fit)"},

    # Haz (energía y emittancias)
    "mean_energy": {"unit": "MeV", "desc": "Mean energy of beam"},
    "std_energy": {"unit": "MeV", "desc": "Energy spread (std)"},
    "mean_energy_hot_electrons": {"unit": "MeV", "desc": "Mean energy for E>threshold"},

    "emittance_x": {"unit": "um·rad", "desc": "Normalized emittance x (μm·rad)"},
    "emittance_y": {"unit": "um·rad", "desc": "Normalized emittance y (μm·rad)"},
    "emittance_xy": {"unit": "um·rad", "desc": "Geometric mean of normalized emittances"},

    "alpha_x": {"unit": "", "desc": "Twiss alpha (x)"},
    "beta_x": {"unit": "", "desc": "Twiss beta (x)"},
    "gamma_x": {"unit": "", "desc": "Twiss gamma (x)"},
}

def _should(metric_name, selected):
    """Devuelve True si hay que extraer/insertar metric_name según 'selected'."""
    if not selected:  # None, [], etc. => todas
        return True
    return metric_name in selected


class ExtractLFMetrics(Command):
    """
    Este comando extrae métricas de campos y partículas para el último frame de una simulación.
    
    Extrae y analiza diferentes propiedades físicas de la simulación, incluyendo:
    - Amplitud y longitud de onda de la onda longitudinal
    - Amplitud y longitud de onda de la onda transversal
    - Parámetros de Twiss del haz
    - Espectro de energía del haz
    
    Las métricas extraídas se almacenan en la base de datos para su posterior análisis.
    """

    def __init__(self, sim_id, metrics_to_extract=None):
        """
        Inicializa el extractor de métricas.
        
        Args:
            sim_id (int): Identificador único de la simulación.
            metrics_to_extract (list, optional): Lista de métricas específicas a extraer.
                Si es None, se extraen las métricas predeterminadas.
        """
        self.c = 299792458.0
        self.sim_id = sim_id
        self.directory_path = os.getcwd()
        self.logger = get_logger()
        self.config = Config().get_instance()
        self.db_path = self.config.get_config('Directories', 'db_path')
        self.simulation_type = self.detect_simulation_type()
        
        # Usar solo las métricas especificadas
        self.metrics_to_extract = metrics_to_extract or []
        
        print(f'Tipo de simulación: {self.simulation_type}')
        if self.simulation_type == 'h5':
            self.load_data()
        else:
            raise NotImplementedError('Tipo de archivo no soportado.')

    def detect_simulation_type(self):
        """
        Detecta el tipo de simulación basado en los archivos disponibles.
        
        Returns:
            str: Tipo de simulación detectado ('h5' actualmente soportado).
            
        Raises:
            FileNotFoundError: Si no se encuentra el directorio 3D o archivos h5.
        """
        if os.path.exists(f'{self.directory_path}/3D'):
            data_dir = f'{self.directory_path}/3D'
        else:
            raise FileNotFoundError('No se encontró el directorio 3D.')
        extensions = [file.split('.')[-1] for file in os.listdir(data_dir)]
        print(f'Extensiones en {data_dir}: {extensions}')
        if 'h5' in extensions:
            return 'h5'
        else:
            raise FileNotFoundError('No se encontraron archivos h5 en el directorio.')

    def load_data(self):
        """
        Carga los datos del último frame de la simulación desde archivos h5.
        
        Extrae campos electromagnéticos, densidad de carga y datos de partículas.
        
        Raises:
            FileNotFoundError: Si no se encuentran archivos h5 en el directorio.
        """
        data_dir = f'{self.directory_path}/3D'
        h5_files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.h5')]
        if not h5_files:
            self.logger.error('No se encontraron archivos h5 en el directorio.')
            raise FileNotFoundError('No se encontraron archivos h5 en el directorio.')

        last_frame_file = h5_files[-1]
        with h5py.File(last_frame_file, 'r') as f:
            data_grp = f['data']
            it_keys = sorted([k for k in data_grp.keys()], key=lambda s: int(s))
            it = it_keys[-1]
            base_path = data_grp[it]

            # Campos (si alguno falta, seguimos con lo que haya)
            fields = base_path.get('fields', None)
            if fields is None:
                raise FileNotFoundError("No 'fields' group in last frame.")

            E_group = fields.get('E', None)
            B_group = fields.get('B', None)
            rho_dataset = fields.get('rho', None)

            if E_group is None or B_group is None:
                raise FileNotFoundError("Missing fields/E or fields/B in last frame.")

            self.ex_data = np.array(E_group['x'])
            self.ey_data = np.array(E_group['y'])
            self.ez_data = np.array(E_group['z'])
            self.bx_data = np.array(B_group['x'])
            self.by_data = np.array(B_group['y'])
            self.bz_data = np.array(B_group['z'])
            self.rho_data = np.array(rho_dataset) if rho_dataset is not None else None

            # Partículas (opcional)
            parts_root = base_path.get('particles', None)
            self.particles_available = []
            self.species = None
            if parts_root is not None:
                self.particles_available = [k for k in parts_root.keys()]
                self.species = self._choose_species(base_path)

                if self.species is not None:
                    pg = parts_root[self.species]
                    # openPMD estándar: position/{x,y,z}, momentum/{x,y,z}
                    self.x_data  = np.array(pg['position']['x'])
                    self.y_data  = np.array(pg['position']['y'])
                    self.z_data  = np.array(pg['position']['z'])
                    self.px_data = np.array(pg['momentum']['x'])
                    self.py_data = np.array(pg['momentum']['y'])
                    self.pz_data = np.array(pg['momentum']['z'])
                else:
                    self.x_data = self.y_data = self.z_data = None
                    self.px_data = self.py_data = self.pz_data = None
            else:
                self.x_data = self.y_data = self.z_data = None
                self.px_data = self.py_data = self.pz_data = None

        self.logger.info(f"Cargando datos desde {last_frame_file}, iter={it}")
        self.logger.info(f"Species disponibles: {self.particles_available} ; elegida: {self.species}")


        # Calcular el wakefield transversal
        self.transverse_wakefield_data = self.ey_data - self.c * self.bz_data
        self.longitudinal_wakefield_data = self.ez_data

    def _list_particle_species(self, base_path):
        parts = base_path.get('particles', None)
        if parts is None:
            return []
        return [k for k in parts.keys()]

    def _choose_species(self, base_path):
        # 1) override por entorno
        sp_env = os.environ.get("SPECIES", "").strip()
        if sp_env and ('particles' in base_path) and (sp_env in base_path['particles'].keys()):
            return sp_env

        # 2) heurística común
        candidates = ["beam", "electrons", "ionized_electrons"]
        available = self._list_particle_species(base_path)
        # prioriza beam/electrons conocidos
        for c in candidates:
            if c in available:
                return c
        # luego cualquier que contenga "electron"
        for s in available:
            if "electron" in s.lower():
                return s
        # por último, primero disponible
        return available[0] if available else None


    def _centerline_longitudinal(self):
        nx, ny, nz = self.ez_data.shape
        cy = ny // 2
        # perfil en x a z central
        return np.arange(nx), self.ez_data[:, cy, nz // 2]

    def _centerline_transverse(self):
        # usa self.transverse_wakefield_data ya construida como Ey - c Bz
        nx, ny, nz = self.transverse_wakefield_data.shape
        cy = ny // 2
        return np.arange(nx), self.transverse_wakefield_data[:, cy, nz // 2]


    def calculate_twiss_parameters(self):
        """
        Twiss y emitancia normalizada (x-plano) usando x' ≈ p_x/p_z (paraxial).
        Devuelve µm·rad para la emitancia normalizada.
        """
        # Guard: partículas disponibles
        if any(v is None for v in (self.x_data, self.px_data, self.py_data, self.pz_data)):
            self.logger.warning("No particle arrays available; skipping Twiss/emitance (x).")
            return {'emittance_x': None, 'beta_x': None, 'gamma_x': None, 'alpha_x': None}

        x = np.asarray(self.x_data, dtype=float)
        px = np.asarray(self.px_data, dtype=float)
        py = np.asarray(self.py_data, dtype=float)
        pz = np.asarray(self.pz_data, dtype=float)

        # Evita divisiones por ~0 en pz para definir x' = px/pz
        m_e = 9.10938356e-31
        c   = 299792458.0
        eps_p = 1e-30 * m_e * c
        pz_safe = np.where(np.abs(pz) < eps_p, np.sign(pz) * eps_p + (pz == 0) * eps_p, pz)
        xprime = px / pz_safe  # adimensional (rad)

        # Centrados
        x_c  = x - np.mean(x)
        xp_c = xprime - np.mean(xprime)

        # Momentos de 2º orden
        sig_x2  = np.mean(x_c**2)
        sig_xp2 = np.mean(xp_c**2)
        sig_xxp = np.mean(x_c * xp_c)

        # Emitancia geométrica (>=0 numéricamente)
        eps_geom2 = sig_x2 * sig_xp2 - sig_xxp**2
        if eps_geom2 <= 0 or not np.isfinite(eps_geom2):
            self.logger.warning(f"Geometric emittance² non-positive: {eps_geom2}. Clamping.")
            eps_geom = 0.0
        else:
            eps_geom = float(np.sqrt(eps_geom2))

        # <gamma> a partir del momento total p = sqrt(px^2+py^2+pz^2)
        p_tot = np.sqrt(px*px + py*py + pz*pz)
        gamma = np.sqrt(1.0 + (p_tot / (m_e * c))**2)
        gamma_bar = float(np.mean(gamma))

        # Emitancia normalizada: eps_n = <gamma> * eps_geom  (paraxial, beta≈1)
        eps_norm = gamma_bar * eps_geom  # [rad·m]
        eps_norm_um = eps_norm * 1e6     # -> µm·rad

        # Twiss a partir de la geométrica
        if eps_geom > 0.0:
            beta_x  = sig_x2  / eps_geom
            gamma_x = sig_xp2 / eps_geom
            alpha_x = - sig_xxp / eps_geom
        else:
            beta_x = gamma_x = alpha_x = None

        return {
            'emittance_x': eps_norm_um,  # µm·rad
            'beta_x': float(beta_x) if beta_x is not None else None,
            'gamma_x': float(gamma_x) if gamma_x is not None else None,
            'alpha_x': float(alpha_x) if alpha_x is not None else None,
        }
        
    def calculate_energy_spectrum(self):
        """
        Calcula el espectro de energía del haz de partículas.
        
        Ajusta la distribución de energía a una función gaussiana para obtener
        la energía media y la dispersión de energía.
        
        Returns:
            dict: Diccionario con las propiedades del espectro de energía:
                - mean_energy: Energía media del haz en MeV
                - std_energy: Desviación estándar de la energía en MeV
        """
        m_e = 9.10938356e-31
        c = 299792458.0
        J_to_MeV = 1/(1.60217662e-13)
        p_array = np.sqrt(self.px_data ** 2 + self.py_data ** 2 + self.pz_data ** 2)
        gamma = np.sqrt(1 + (p_array / (m_e * c)) ** 2)
        E_J = gamma * m_e * c ** 2
        E_MeV = E_J * J_to_MeV
        # Estadísticos directos (el fit queda opcional)
        return float(np.mean(E_MeV)), float(np.std(E_MeV))
    
    def gaussian(self, x, A, mu, sigma):
        """
        Función gaussiana para ajuste de distribuciones.
        
        Args:
            x (array): Valores de entrada.
            A (float): Amplitud de la gaussiana.
            mu (float): Media de la gaussiana.
            sigma (float): Desviación estándar de la gaussiana.
            
        Returns:
            array: Valores de la función gaussiana evaluada en x.
        """
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    
    def sine_multi_mode(self, x, *params):
        """
        Función de múltiples modos sinusoidales para ajuste de ondas.
        
        Args:
            x (array): Valores de entrada.
            *params: Parámetros de la función:
                - params[0]: Término constante
                - params[3*i+1], params[3*i+2], params[3*i+3]: Amplitud, número de onda y fase
                  para el i-ésimo modo.
                  
        Returns:
            array: Valores de la función evaluada en x.
        """
        result = params[0]  # Término constante
        num_modes = (len(params) - 1) // 3  # Cada modo tiene 3 parámetros: A, k, phi
        for i in range(num_modes):
            A = params[3 * i + 1]
            k = params[3 * i + 2]
            phi = params[3 * i + 3]
            result += A * np.sin(k * x + phi)
        return result

    @staticmethod
    def _savgol_safe(y, max_win=51, poly=3):
        n = len(y)
        win = min(max_win, n if n % 2 == 1 else n-1)
        if win < (poly + 2):  # ventana mínima para el polinomio
            return y  # sin suavizado
        return savgol_filter(y, win, poly)

    
    def fit_sine_multi_mode(self, x, y, max_modes=5):
        """
        Ajuste multi-modo robusto. Devuelve:
        popt, error, avg_wavelength, amplitude_global, max_value
        con caídas seguras si la señal es plana o el ajuste falla.
        """
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        n = y.size

        # 0) Señal demasiado corta
        if n < 7:
            amp = 0.5 * (np.max(y) - np.min(y)) if n else 0.0
            return None, None, 0.0, float(abs(amp)), np.max(np.abs(y)) if n else 1.0

        # 1) Suavizado seguro
        y_filtered = self._savgol_safe(y, max_win=51, poly=3)

        # 2) Normalización robusta
        max_value = float(np.max(np.abs(y_filtered)))
        if not np.isfinite(max_value) or max_value <= 1e-20:
            # señal ~ plana
            return None, None, 0.0, 0.0, 1.0
        y_normalized = y_filtered / max_value

        # 3) FFT positiva (descarta DC)
        Y = np.fft.rfft(y_normalized)
        freqs = np.fft.rfftfreq(n)  # ciclos por muestra
        if freqs.size <= 1:
            amp = 0.5 * (np.max(y) - np.min(y))
            return None, None, 0.0, float(abs(amp)), max_value

        power = np.abs(Y)
        # descarta f=0 (DC)
        freqs = freqs[1:]
        power = power[1:]
        if power.size == 0 or np.all(power <= 1e-12):
            # Muy poca energía: fallback directo
            amp = 0.5 * (np.max(y_filtered) - np.min(y_filtered))
            return None, None, 0.0, float(abs(amp)), max_value

        # 4) Selección de modos dominantes
        k_modes = min(max_modes, power.size)
        idx = np.argpartition(power, -k_modes)[-k_modes:]
        # ordénalos por frecuencia ascendente (opcional)
        idx = np.sort(idx)
        dom_freqs = freqs[idx]
        dom_amps  = power[idx]
        sum_dom = float(np.sum(dom_amps))
        if not np.isfinite(sum_dom) or sum_dom <= 0:
            amp = 0.5 * (np.max(y_filtered) - np.min(y_filtered))
            return None, None, 0.0, float(abs(amp)), max_value

        # 5) Inicialización segura: [C0, A1, k1, phi1, A2, k2, phi2, ...]
        init = [0.0]
        for a, f in zip(dom_amps, dom_freqs):
            A0 = 0.5 * (a / sum_dom)  # reparte 0.5 entre modos
            k0 = 2.0 * np.pi * f      # rad/muestra
            phi0 = 0.0
            init.extend([A0, k0, phi0])

        # 6) Bounds físicos: C0, (A,k,phi)*m
        #    - C0, A: [-1e6, 1e6] (finito para 'trf')
        #    - k: [1e-6, pi] (Nyquist)
        #    - phi: [-pi, pi]
        lb = [-1e6]
        ub = [ 1e6]
        for _ in range(k_modes):
            lb.extend([-1e6, 1e-6, -np.pi])
            ub.extend([ 1e6,  np.pi,  np.pi])

        # 7) Ajuste
        try:
            popt, pcov = curve_fit(
                self.sine_multi_mode, x, y_normalized,
                p0=np.array(init, dtype=float),
                bounds=(np.array(lb, dtype=float), np.array(ub, dtype=float)),
                maxfev=80000,
                ftol=1e-8,
                method='trf'
            )

            fit_norm = self.sine_multi_mode(x, *popt)
            fit_den  = fit_norm * max_value
            # Error cuadrático
            err = float(np.sum((y_filtered - fit_den) ** 2))

            # Longitud de onda promedio ponderada por amplitud
            wavelengths = []
            weights = []
            # popt = [C0, A1,k1,phi1, A2,k2,phi2, ...]
            for i in range((len(popt) - 1) // 3):
                A = abs(popt[3*i + 1])
                k = abs(popt[3*i + 2])
                if np.isfinite(A) and np.isfinite(k) and k > 1e-9:
                    wavelengths.append(2*np.pi / k)
                    weights.append(A)
            if wavelengths and np.sum(weights) > 0:
                avg_wl = float(np.sum(np.array(wavelengths) * np.array(weights)) / np.sum(weights))
            else:
                # Fallback: pico FFT dominante
                j = int(np.argmax(power))
                f_dom = float(freqs[j]) if j < len(freqs) else 0.0
                avg_wl = float(1.0 / f_dom) if f_dom > 1e-9 else 0.0

            # Amplitud global (ajuste des-normalizado)
            amplitude_global = float(np.max(np.abs(fit_den)))

            return popt, err, avg_wl, amplitude_global, max_value

        except Exception:
            # 8) Fallback totalmente determinista (sin ajuste)
            #    - amplitud por pico-a-pico
            amp = 0.5 * (np.max(y_filtered) - np.min(y_filtered))
            #    - longitud de onda por pico del espectro
            j = int(np.argmax(power))
            f_dom = float(freqs[j]) if j < len(freqs) else 0.0
            wl = float(1.0 / f_dom) if f_dom > 1e-9 else 0.0
            return None, None, wl, float(abs(amp)), max_value

    def calculate_twiss_parameters_y(self):
        """
        Calcula los parámetros de Twiss y la emitancia normalizada en el plano y.
        """
        y_mean = np.mean(self.y_data)
        py_mean = np.mean(self.py_data)
        y_rms = np.sqrt(np.mean((self.y_data - y_mean) ** 2))
        py_rms = np.sqrt(np.mean((self.py_data - py_mean) ** 2))
        ypy_cov = np.mean((self.y_data - y_mean) * (self.py_data - py_mean))
        emittance_y_squared = y_rms ** 2 * py_rms ** 2 - ypy_cov ** 2
        if emittance_y_squared <= 0:
            self.logger.warning(f'Emittance y squared is too small: {emittance_y_squared}')
            emittance_y = 1e-25
        else:
            emittance_y = np.sqrt(emittance_y_squared)
        m_e = 9.10938356e-31
        c = 299792458.0
        mec = m_e * c
        gamma_beta = np.mean(np.sqrt(1 + (self.pz_data / mec) ** 2))
        emittance_y_normalized = emittance_y * gamma_beta
        emittance_y_normalized *= 1e6
        return float(emittance_y_normalized)

    def calculate_emittance_xy(self):
        """
        Calcula la media geométrica de las emitancias normalizadas en x e y.
        """
        emittance_x = self.calculate_twiss_parameters()['emittance_x']
        emittance_y = self.calculate_twiss_parameters_y()
        emittance_xy = np.sqrt(emittance_x * emittance_y)
        return float(emittance_xy)

    def calculate_mean_energy_hot_electrons(self, threshold_MeV=20.0):
        """
        Calcula la energía media de los electrones con energía mayor a threshold_MeV.
        """
        m_e = 9.10938356e-31
        c = 299792458.0
        J_to_MeV = 1/(1.60217662e-13)
        p_array = np.sqrt(self.px_data ** 2 + self.py_data ** 2 + self.pz_data ** 2)
        gamma = np.sqrt(1 + (p_array / (m_e * c)) ** 2)
        E_J = gamma * m_e * c ** 2
        E_MeV = E_J * J_to_MeV
        hot_electrons = E_MeV[E_MeV > threshold_MeV]
        if hot_electrons.size == 0:
            self.logger.warning('No se encontraron hot electrons por encima del umbral.')
            return 0.0
        return float(np.mean(hot_electrons))

    def execute(self):
        try:
            self.logger.info(f'Extrayendo métricas para la simulación {self.sim_id}')
            if self.ex_data.size == 0 or self.ey_data.size == 0 or self.bz_data.size == 0:
                self.logger.error('Uno o más de los arrays recuperados están vacíos.')
                return False

            metrics_out = {}

            # --- Longitudinal (Ez) ---
            if _should("longitudinal_amplitude", self.metrics_to_extract) or \
            _should("longitudinal_wavelength", self.metrics_to_extract):
                x, y = self._centerline_longitudinal()
                poptL, errL, wlL, ampL, _ = self.fit_sine_multi_mode(x, y)
                if poptL is None:
                    ampL, wlL = 0.0, 0.0
                if _should("longitudinal_amplitude", self.metrics_to_extract):
                    metrics_out["longitudinal_amplitude"] = float(abs(ampL))
                if _should("longitudinal_wavelength", self.metrics_to_extract):
                    metrics_out["longitudinal_wavelength"] = float(abs(wlL))

            # --- Transverse (Ey - c Bz) ---
            if _should("transverse_amplitude", self.metrics_to_extract) or \
            _should("transverse_wavelength", self.metrics_to_extract):
                xT, yT = self._centerline_transverse()
                poptT, errT, wlT, ampT, _ = self.fit_sine_multi_mode(xT, yT)
                if poptT is None:
                    ampT, wlT = 0.0, 0.0
                if _should("transverse_amplitude", self.metrics_to_extract):
                    metrics_out["transverse_amplitude"] = float(abs(ampT))
                if _should("transverse_wavelength", self.metrics_to_extract):
                    metrics_out["transverse_wavelength"] = float(abs(wlT))

            # --- Energía (haz) ---
            if _should("mean_energy", self.metrics_to_extract) or _should("std_energy", self.metrics_to_extract):
                meanE, stdE = self.calculate_energy_spectrum()
                if _should("mean_energy", self.metrics_to_extract):
                    metrics_out["mean_energy"] = meanE
                if _should("std_energy", self.metrics_to_extract):
                    metrics_out["std_energy"] = stdE

            if _should("mean_energy_hot_electrons", self.metrics_to_extract):
                metrics_out["mean_energy_hot_electrons"] = self.calculate_mean_energy_hot_electrons()

            # --- Twiss / emittances ---
            if any(_should(k, self.metrics_to_extract) for k in ("emittance_x","alpha_x","beta_x","gamma_x")):
                twx = self.calculate_twiss_parameters()  # devuelve emittance_x, beta_x, gamma_x, alpha_x
                if _should("emittance_x", self.metrics_to_extract): metrics_out["emittance_x"] = float(twx["emittance_x"])
                if _should("beta_x", self.metrics_to_extract):      metrics_out["beta_x"] = float(twx["beta_x"])
                if _should("gamma_x", self.metrics_to_extract):     metrics_out["gamma_x"] = float(twx["gamma_x"])
                if _should("alpha_x", self.metrics_to_extract):     metrics_out["alpha_x"] = float(twx["alpha_x"])

            if _should("emittance_y", self.metrics_to_extract):
                metrics_out["emittance_y"] = float(self.calculate_twiss_parameters_y())
            if _should("emittance_xy", self.metrics_to_extract):
                metrics_out["emittance_xy"] = float(self.calculate_emittance_xy())

            # --- Inserción en DB (metrics table) ---
            for name, value in metrics_out.items():
                unit = ALL_METRICS.get(name, {}).get("unit", None)
                self.logger.info(f'Insertando métrica {name}={value} (sim_id={self.sim_id})')
                # Si tu InsertResult ya sabe meter en 'metrics', úsalo;
                # si no, inserta aquí directamente con sqlite3.
                InsertResult(self.sim_id, name, value).execute()

            return True

        except Exception as e:
            self.logger.error(f"Error extrayendo métricas: {e}")
            return False


    def extract_metrics(self):
        """
        Extrae las métricas especificadas en metrics_to_extract.
        
        Returns:
            dict: Diccionario con las métricas extraídas y sus valores.
        """
        # Implementa la lógica para extraer solo las métricas especificadas
        extracted_metrics = {}
        for metric in self.metrics_to_extract:
            # Aquí iría la lógica para extraer cada métrica
            extracted_metrics[metric] = self.extract_specific_metric(metric)
        return extracted_metrics

    def extract_specific_metric(self, metric):
        """
        Extrae una métrica específica.
        
        Args:
            metric (str): Nombre de la métrica a extraer.
            
        Returns:
            float: Valor de la métrica extraída.
        """
        # Lógica para extraer una métrica específica
        # Esto es un ejemplo, deberías implementar la lógica real de extracción
        self.logger.info(f'Extrayendo métrica: {metric}')
        return 0.0  # Valor de ejemplo

# Ejecución del script pasando el ID de la simulación como argumento
if __name__ == "__main__":
    sim_id = int(sys.argv[1])
    # Acepta: (i) sin 2º arg -> todas; (ii) "[]" -> todas; (iii) lista JSON
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        try:
            metrics_to_extract = json.loads(sys.argv[2])
            if not isinstance(metrics_to_extract, list):
                metrics_to_extract = []
        except Exception:
            metrics_to_extract = []
    else:
        metrics_to_extract = []  # por defecto: TODAS

    extractor = ExtractLFMetrics(sim_id, metrics_to_extract)
    success = extractor.execute()
    sys.exit(0 if success else 1)
