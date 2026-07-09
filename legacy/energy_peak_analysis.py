import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import sys

# Obtener el ID de simulación del argumento de línea de comandos
if len(sys.argv) != 2:
    print("Uso: python energy_peak_analysis.py <simulation_id>")
    sys.exit(1)

simulation_id = int(sys.argv[1])
print(f"Analizando simulación {simulation_id}")

initialstep = 0
maxstep = 8000
steps = np.arange(initialstep, maxstep + 100, 100)

# Umbral de energía en MeV (ignorar electrones con energía menor a este valor)
energy_threshold = 5.0  # MeV

def load_data(step):
    """Carga los datos para un paso específico de la simulación"""
    file_path = f'/home/jrope/workspaces/cursor_workspace/simulations/multichannel_lwfa_labuena/combination_{simulation_id}/3D/openpmd_{step:06d}.h5'
    
    with h5py.File(file_path, 'r') as f:
        # Cargar datos de partículas
        species = 'electrons'
        z_dataset = f['data'][f'{step}']['particles'][species]['position']['z']
        pz_dataset = f['data'][f'{step}']['particles'][species]['momentum']['z']
        px_dataset = f['data'][f'{step}']['particles'][species]['momentum']['x']
        py_dataset = f['data'][f'{step}']['particles'][species]['momentum']['y']
        
        # Convertir a arrays
        z_array = np.array(z_dataset)
        pz_array = np.array(pz_dataset)
        px_array = np.array(px_dataset)
        py_array = np.array(py_dataset)
        
    return {
        'z': z_array,
        'pz': pz_array,
        'px': px_array,
        'py': py_array
    }

def calculate_energy(data):
    """Calcula la energía de los electrones en MeV"""
    # Constantes físicas
    m_e = 9.10938356e-31  # kg
    c = 299792458  # m/s
    J_to_MeV = 1/(1.60218e-13)  # Conversión de J a MeV
    
    # Calcular momento total
    p_array = np.sqrt(data['px']**2 + data['py']**2 + data['pz']**2)
    
    # Calcular factor gamma
    gamma = np.sqrt(1 + (p_array/(m_e*c))**2)
    
    # Calcular energía en MeV
    E_MeV = gamma * m_e * c**2 * J_to_MeV
    
    return E_MeV

def analyze_energy_peaks():
    """Analiza los picos de energía en cada frame"""
    stats = {
        'frames': [],
        'mean_energies': [],
        'median_energies': [],
        'mode_energies': [],
        'std_energies': [],
        'num_electrons': [],
        'percent_above_mean': [],
        'total_charge': []  # Carga total en pC
    }
    
    for step in steps:
        data = load_data(step)
        energies = calculate_energy(data)
        
        # Filtrar energías por encima del umbral
        mask = energies >= energy_threshold
        if np.any(mask):
            filtered_energies = energies[mask]
            num_electrons = np.sum(mask)
            
            # Calcular estadísticas
            mean_energy = np.mean(filtered_energies)
            median_energy = np.median(filtered_energies)
            std_energy = np.std(filtered_energies)
            
            # Calcular moda usando histograma
            hist, bin_edges = np.histogram(filtered_energies, bins=50)
            mode_energy = bin_edges[np.argmax(hist)]
            
            # Porcentaje de electrones por encima de la media
            percent_above_mean = np.sum(filtered_energies > mean_energy) / len(filtered_energies) * 100
            
            # Calcular carga total en pC (1.602176634e-19 C por electrón)
            total_charge = num_electrons * 1.602176634e-19 * 1e12  # Convertir a pC
            
            # Guardar estadísticas
            stats['frames'].append(step)
            stats['mean_energies'].append(mean_energy)
            stats['median_energies'].append(median_energy)
            stats['mode_energies'].append(mode_energy)
            stats['std_energies'].append(std_energy)
            stats['num_electrons'].append(num_electrons)
            stats['percent_above_mean'].append(percent_above_mean)
            stats['total_charge'].append(total_charge)
    
    return stats

def plot_energy_peaks(stats):
    """Crea gráficos de las estadísticas de energía"""
    # Crear figura con subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # Gráfico 1: Energías (media, mediana, moda)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(stats['frames'], stats['mean_energies'], 'b-', label='Media')
    ax1.plot(stats['frames'], stats['median_energies'], 'g-', label='Mediana')
    ax1.plot(stats['frames'], stats['mode_energies'], 'r-', label='Moda')
    
    # Encontrar el frame con la energía promedio máxima
    max_energy_idx = np.argmax(stats['mean_energies'])
    max_energy_frame = stats['frames'][max_energy_idx]
    max_energy_value = stats['mean_energies'][max_energy_idx]
    
    ax1.scatter(max_energy_frame, max_energy_value, color='black', 
               label=f'Máximo: {max_energy_value:.2f} MeV en frame {max_energy_frame}')
    
    ax1.set_ylabel('Energía (MeV)')
    ax1.set_title(f'Estadísticas de energía - Simulación {simulation_id}\n(Umbral: {energy_threshold} MeV)')
    ax1.grid(True)
    ax1.legend()
    
    # Gráfico 2: Número de electrones y carga
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(stats['frames'], stats['num_electrons'], 'b-', label='Número de electrones')
    ax2.set_ylabel('Número de electrones')
    ax2.grid(True)
    ax2.legend()
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(stats['frames'], stats['total_charge'], 'r-', label='Carga total')
    ax2_twin.set_ylabel('Carga (pC)')
    ax2_twin.legend(loc='upper right')
    
    # Gráfico 3: Desviación estándar
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(stats['frames'], stats['std_energies'], 'g-', label='Desviación estándar')
    ax3.set_ylabel('Desviación estándar (MeV)')
    ax3.grid(True)
    ax3.legend()
    
    # Gráfico 4: Porcentaje por encima de la media
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(stats['frames'], stats['percent_above_mean'], 'm-', label='% electrones > media')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Porcentaje (%)')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    
    # Guardar el gráfico
    output_dir = '/home/jrope/workspaces/cursor_workspace/simulations/multichannel_lwfa_labuena'
    plt.savefig(f'{output_dir}/energy_peaks_sim{simulation_id}.png')
    plt.close()
    
    # Imprimir estadísticas detalladas
    print(f"\nResultados detallados para simulación {simulation_id}:")
    print(f"Frame con energía promedio máxima: {max_energy_frame}")
    print(f"Energía promedio máxima: {max_energy_value:.2f} MeV")
    print(f"Mediana en el pico: {stats['median_energies'][max_energy_idx]:.2f} MeV")
    print(f"Moda en el pico: {stats['mode_energies'][max_energy_idx]:.2f} MeV")
    print(f"Desviación estándar en el pico: {stats['std_energies'][max_energy_idx]:.2f} MeV")
    print(f"Número de electrones en el pico: {stats['num_electrons'][max_energy_idx]:.0f}")
    print(f"Carga total en el pico: {stats['total_charge'][max_energy_idx]:.2f} pC")
    print(f"Porcentaje de electrones por encima de la media: {stats['percent_above_mean'][max_energy_idx]:.1f}%")
    print(f"Gráfico guardado en: {output_dir}/energy_peaks_sim{simulation_id}.png")

if __name__ == "__main__":
    stats = analyze_energy_peaks()
    plot_energy_peaks(stats) 