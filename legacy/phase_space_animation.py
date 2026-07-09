import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

# Obtener el ID de simulación del argumento de línea de comandos
if len(sys.argv) != 2:
    print("Uso: python phase_space_animation.py <simulation_id>")
    sys.exit(1)

simulation_id = int(sys.argv[1])
print(f"Procesando simulación {simulation_id}")

initialstep = 0
maxstep = 4000
period = 100
steps = np.arange(initialstep, maxstep + period, period)

# Directorio para guardar las animaciones
output_dir = '/home/jrope/workspaces/cursor_workspace/simulations/hexagonal_buenas'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_particle_data(step):
    """Carga solo los datos de partículas para un paso específico de la simulación"""
    file_path = f'/home/jrope/workspaces/cursor_workspace/simulations/hexagonal_buenas/combination_{simulation_id}/3D/openpmd_{step:06d}.h5'
    
    with h5py.File(file_path, 'r') as f:
        # Cargar datos de partículas para electrones
        species = 'electrons'
        z_dataset = f['data'][f'{step}']['particles'][species]['position']['z']
        pz_dataset = f['data'][f'{step}']['particles'][species]['momentum']['z']
        
        # Cargar datos de partículas para electrones ionizados
        ionized_species = 'ionized_electrons'
        z_ionized_dataset = f['data'][f'{step}']['particles'][ionized_species]['position']['z']
        pz_ionized_dataset = f['data'][f'{step}']['particles'][ionized_species]['momentum']['z']
        
        # Convertir a arrays
        z_array = np.array(z_dataset)
        pz_array = np.array(pz_dataset)
        z_ionized_array = np.array(z_ionized_dataset)
        pz_ionized_array = np.array(pz_ionized_dataset)
        
    return {
        'z': z_array,
        'pz': pz_array,
        'z_ionized': z_ionized_array,
        'pz_ionized': pz_ionized_array
    }

def create_phase_space_animation():
    """Crea una animación del espacio de fases z-pz"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        data = load_particle_data(steps[frame])
        
        # Convertir unidades para electrones
        z_mm = data['z'] * 1000  # m a mm
        pz_mev = data['pz'] * 1/(5*10**(-28)*10**6)  # a MeV/c
        
        # Convertir unidades para electrones ionizados
        z_ionized_mm = data['z_ionized'] * 1000  # m a mm
        pz_ionized_mev = data['pz_ionized'] * 1/(5*10**(-28)*10**6)  # a MeV/c
        
        # Crear scatter plot para electrones
        scatter_electrons = ax.scatter(z_mm, pz_mev, s=1, alpha=0.5, label='Electrones', color='blue')
        
        # Crear scatter plot para electrones ionizados
        scatter_ionized = ax.scatter(z_ionized_mm, pz_ionized_mev, s=1, alpha=0.5, label='Electrones Ionizados', color='red')
        
        ax.set_xlabel('z [mm]')
        ax.set_ylabel('pz [MeV/c]')
        ax.set_title(f'Diagrama de espacio de fases z-pz - Paso {steps[frame]}')
        ax.grid(True)
        ax.legend()
        return [scatter_electrons, scatter_ionized]
    
    ani = animation.FuncAnimation(fig, update, frames=len(steps), blit=True)
    
    # Guardar la animación como GIF
    ani.save(f'{output_dir}/phase_space_animation_sim{simulation_id}.gif', writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    print("Creando animación del espacio de fases...")
    create_phase_space_animation()
    print("¡Animación del espacio de fases ha sido creada con éxito!") 