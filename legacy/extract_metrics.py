from base_command import Command
from ex_filter import ExFilter
from extract_wakefield_metrics import ExtractWakefieldMetrics
from animate.phase_space import PhaseSpaceAnimation
from scipy.stats import linregress
from insert_result import InsertResult
from get_array_by_id import GetArrayByID
from singletons.config import Config
from singletons.logger import Logger
import numpy as np
import sys
import os

class ExtractMetrics(Command):
    """
    A command for extracting the wakefield metrics and putting them into the database.

    Args:
        sim_id (int): The ID of the simulation.
        logger (Logger): The logger object for logging messages.
        config (Config): The configuration object for accessing configuration settings.

    Returns:
        bool: True if the metrics were successfully extracted and inserted into the database, False otherwise.
    """
    
    def __init__(self, sim_id):
        self.c = 299792458.0
        self.sim_id = sim_id
        self.directory_path = os.getcwd()
        self.logger = Logger().get_logger()
        self.config = Config().get_instance()
        self.db_path = self.config.get_config('Directories', 'db_path')
        
        self.file_type = self.detect_file_type()
        print(f'File type: {self.file_type}')
        
        if self.file_type == 'sdf':
            self.ex_name = 'Electric_Field_Ex'
            self.ey_name = 'Electric_Field_Ey'
            self.bz_name = 'Magnetic_Field_Bz'
        elif self.file_type == 'h5':
            self.ex_name = 'E_z_0_dump'
            self.ey_name = 'E_r_0_dump'
            self.bz_name = 'B_z_0_dump'
            
        self.load_data()
            
    def detect_file_type(self):
        # List all extensions among the files in the directory
        extensions = [file.split('.')[-1] for file in os.listdir(f'{self.directory_path}/RZ')]
        print(f'Extensions in {self.directory_path}/RZ: {extensions}')
        if extensions:
            if 'sdf' in extensions:
                return 'sdf'
            elif 'h5' in extensions:
                return 'h5'
        else:
            return FileNotFoundError('No files found in the directory.')
    
    def load_data(self):
        self.ex_data = GetArrayByID(self.sim_id, self.ex_name).execute()
        print(f'ex_data shape: {self.ex_data.shape}')
        self.ey_data = GetArrayByID(self.sim_id, self.ey_name).execute()
        print(f'ey_data shape: {self.ey_data.shape}')
        self.bz_data = GetArrayByID(self.sim_id, self.bz_name).execute()
        print(f'bz_data shape: {self.bz_data.shape}')
        self.transverse_wakefield_data = self.ey_data - self.c * self.bz_data
        
    def execute(self):
        """
        Executes the extraction of metrics from the database.

        Returns:
            bool: True if the metrics were successfully extracted and inserted into the database, False otherwise.
        """
        try:
            self.logger.info(f'Extracting metrics for simulation {self.sim_id}')
            
            if self.ex_data.size == 0 or self.ey_data.size == 0 or self.bz_data.size == 0:
                self.logger.info('One or more of the retrieved arrays are empty.')
                self.logger.info(f'ex_data shape: {self.ex_data.shape}')
                self.logger.info(f'ey_data shape: {self.ey_data.shape}')
                self.logger.info(f'bz_data shape: {self.bz_data.shape}')
                return False
            
            ex_data_last_frame = self.ex_data[-1, :, :]
            transverse_wakefield_data_last_frame = self.transverse_wakefield_data[-1, :, :]
            
            self.logger.info(f'Filtering data for simulation {self.sim_id}')
            filtered_ex_data = ExFilter().execute(ex_data_last_frame)
            filtered_transverse_wakefield_data = ExFilter().execute(transverse_wakefield_data_last_frame)
            
            self.logger.info(f'filtered_ex_data shape: {filtered_ex_data.shape}')
            self.logger.info(f'filtered_transverse_wakefield_data shape: {filtered_transverse_wakefield_data.shape}')
            
            if filtered_ex_data.size == 0 or filtered_transverse_wakefield_data.size == 0:
                self.logger.info('Filtered data is empty.')
                return False
            
            last_frame_ex_metrics = ExtractWakefieldMetrics().execute(filtered_ex_data)
            last_frame_transverse_wakefield_metrics = ExtractWakefieldMetrics().execute(filtered_transverse_wakefield_data)

            self.logger.info(f'Last frame ex metrics: {last_frame_ex_metrics}')
            self.logger.info(f'Last frame transverse wakefield metrics: {last_frame_transverse_wakefield_metrics}')

            last_frame_ex_metrics['bubble_sizes'] = float(last_frame_ex_metrics['bubble_sizes'])
            last_frame_transverse_wakefield_metrics['bubble_sizes'] = float(last_frame_transverse_wakefield_metrics['bubble_sizes'])
            
            self.logger.info(f'Extracting amplitudes for simulation {self.sim_id}')
            ex_amplitudes = []
            transverse_wakefield_amplitudes = []
            for frame in range(self.ex_data.shape[0]):
                self.logger.info(f'Processing frame {frame}')
                ex_data_frame = self.ex_data[frame, :, :]
                transverse_wakefield = self.transverse_wakefield_data[frame, :, :]
                
                filtered_ex_data_frame = ExFilter().execute(ex_data_frame)
                filtered_transverse_wakefield = ExFilter().execute(transverse_wakefield)
                
                if filtered_ex_data_frame.size == 0 or filtered_transverse_wakefield.size == 0:
                    self.logger.info(f'Filtered data is empty at frame {frame}. Skipping this frame.')
                    continue
                
                try:
                    ex_amplitude = ExtractWakefieldMetrics().extract_amplitude(filtered_ex_data_frame)
                    transverse_wakefield_amplitude = ExtractWakefieldMetrics().extract_amplitude(filtered_transverse_wakefield)
                    
                    ex_amplitudes.append(ex_amplitude)
                    transverse_wakefield_amplitudes.append(transverse_wakefield_amplitude)
                except Exception as e:
                    self.logger.info(f'Error extracting amplitudes at frame {frame}: {e}')
                    continue
            
            self.logger.info(f'ex_amplitudes: {ex_amplitudes}')
            self.logger.info(f'transverse_wakefield_amplitudes: {transverse_wakefield_amplitudes}')
            
            if len(ex_amplitudes) == 0 or len(transverse_wakefield_amplitudes) == 0:
                self.logger.info('No amplitudes could be extracted.')
                return False
            
            self.logger.info(f'Calculating slopes for simulation {self.sim_id}')
            self.logger.info(f'Ex amplitudes: {ex_amplitudes}')
            self.logger.info(f'Transverse wakefield amplitudes: {transverse_wakefield_amplitudes}')
            try:
                if len(ex_amplitudes) > 1 and not np.any(np.isnan(ex_amplitudes)):
                    ex_slope = linregress(range(len(ex_amplitudes)), ex_amplitudes)[0]
                else:
                    ex_slope = float('nan')
            except ValueError as ve:
                self.logger.info(f'Error in calculating ex_slope: {ve}')
                ex_slope = float('nan')
                
            try:
                if len(transverse_wakefield_amplitudes) > 1 and not np.any(np.isnan(transverse_wakefield_amplitudes)):
                    transverse_wakefield_slope = linregress(range(len(transverse_wakefield_amplitudes)), transverse_wakefield_amplitudes)[0]
                else:
                    transverse_wakefield_slope = float('nan')
            except ValueError as ve:
                self.logger.info(f'Error in calculating transverse_wakefield_slope: {ve}')
                transverse_wakefield_slope = float('nan')
                
            phase_space_anim = PhaseSpaceAnimation(self.directory_path, self.logger, self.config, self.sim_id)
            
            h5_files = [os.path.join(self.directory_path, f) for f in sorted(os.listdir(self.directory_path)) if f.endswith('.h5')]
            emitances = {}
            for axis in ['x', 'y', 'z']:
                position, momentum = phase_space_anim.load_data(h5_files[-1])[0:2] if axis == 'x' else phase_space_anim.load_data(h5_files[-1])[2:4] if axis == 'y' else phase_space_anim.load_data(h5_files[-1])[4:6]
                # If there are no particles, we will set the emittance equal to 0
                if len(position) == 0 or len(momentum) == 0:
                    emitances[f'{axis}_emittance'] = 0
                    continue
                emitance = phase_space_anim.calculate_emmitance(position, momentum)
                emitances[f'{axis}_emittance'] = emitance
                self.logger.info(f'Emittance for axis {axis}: {emitance}')
            
            transverse_wakefield_amplitude = transverse_wakefield_amplitudes[-1] if transverse_wakefield_amplitudes else float('nan')
            ex_amplitude = ex_amplitudes[-1] if ex_amplitudes else float('nan')
            
            metrics = {
                'ex_amplitude': ex_amplitude,
                'ex_bubble_sizes': last_frame_ex_metrics['bubble_sizes'],
                'ex_slope': ex_slope,
                'transverse_wakefield_amplitude': transverse_wakefield_amplitudes,
                'transverse_wakefield_bubble_sizes': last_frame_transverse_wakefield_metrics['bubble_sizes'],
                'transverse_wakefield_slope': transverse_wakefield_slope,
                'x_emittance': emitances['x_emittance'],
                'y_emittance': emitances['y_emittance'],
                'z_emittance': emitances['z_emittance'],
                'n_particles_last_frame': len(position)
            }

            for metric_name, metric_value in metrics.items():
                self.logger.info(f'Inserting metric {metric_name} for simulation {self.sim_id} with value {metric_value}')
                InsertResult(self.sim_id, metric_name, metric_value).execute()
            return True
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}")
            return False

# We will pass the id on the slurm script doing the following:
# python ~/Python/parameter_scan/commands/analyze/extract_metrics.py $SLURM_ARRAY_TASK_ID
ExtractMetrics(int(sys.argv[1])).execute()
